#!/usr/bin/env python3

import os
import random
import numpy as np
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
# os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
import sys

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

# LoRA imports
from peft import LoraConfig, get_peft_model

# Dataset, collate_fn and model import
from uris.dataset import MyDataset, collate_fn
from uris.models.qformer2llama import QformerLlamaModel

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def freeze_llama_except_lora(llama_model):
    """Freeze all parameters except those injected by LoRA."""
    for name, param in llama_model.named_parameters():
        if "lora_" not in name:
            param.requires_grad = False
    return llama_model

def map_label_to_token(lb):
    lb_clean = lb.strip().lower()
    if lb_clean == "neu":
        return "<NEU>"
    elif lb_clean == "ang":
        return "<ANG>"
    elif lb_clean == "hap":
        return "<HAP>"
    elif lb_clean == "sad":
        return "<SAD>"
    return None

def train_lora_sft(
    pt_file,
    qformer_ckpt_path,
    batch_size=4,
    epochs=3,
    lr=1e-4,
    seed=42,
    lora_r=8,
    lora_alpha=32,
    lora_dropout=0.05,
    lora_target_modules="q_proj,v_proj",
    audio_hidden_dim=1024,
    text_hidden_dim=768,
    llama_hidden_size=4096,
    device="cuda"
):
    """
    LoRA SFT training with Q-Former + LLaMA. Randomly assigns an English prompt to each sample.
    Samples with label "fru" are excluded (only neu, ang, hap, sad are used).

    Args:
      pt_file (str): Path to .pt file containing list[dict] with {audio_emb, text_emb, label}.
      qformer_ckpt_path (str): Checkpoint for Q-Former (e.g. best_qformer.pth).
      batch_size (int): Batch size.
      epochs (int): Number of epochs.
      lr (float): Learning rate.
      seed (int): Random seed.
      lora_r (int): LoRA rank.
      lora_alpha (int): LoRA alpha.
      lora_dropout (float): LoRA dropout.
      lora_target_modules (str): Target modules for LoRA, comma-separated.
      audio_hidden_dim (int): Audio embedding dimension.
      text_hidden_dim (int): Text embedding dimension.
      llama_hidden_size (int): LLaMA hidden size.
      device (str): "cuda" or "cpu".
    """
    set_seed(seed)

    # Build DataLoader
    dataset = MyDataset(pt_file)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

    # Initialize Q-Former + LLaMA model
    model = QformerLlamaModel(
        qformer_ckpt_path=qformer_ckpt_path,
        audio_hidden_dim=audio_hidden_dim,
        text_hidden_dim=text_hidden_dim,
        llama_hidden_size=llama_hidden_size
    ).to(device)

    # Configure LoRA
    target_module_list = lora_target_modules.split(",")
    lora_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        target_modules=target_module_list,
        lora_dropout=lora_dropout,
        bias="none",
        task_type="CAUSAL_LM"
    )

    # Inject LoRA and freeze non-LoRA parameters in LLaMA
    model.llama_model = get_peft_model(model.llama_model, lora_config)
    model.llama_model = freeze_llama_except_lora(model.llama_model)
    print("Pre-PEFT vocab size:", model.llama_model.config.vocab_size)
    if hasattr(model.llama_model, "language_model"):
        print("Pre-PEFT LM vocab size:", model.llama_model.language_model.config.vocab_size)

    # Sync vocab size in config
    new_vocab_size = model.llama_model.get_input_embeddings().weight.size(0)
    model.llama_model.config.vocab_size = new_vocab_size
    if hasattr(model.llama_model, "base_model"):
        model.llama_model.base_model.config.vocab_size = new_vocab_size
        if hasattr(model.llama_model.base_model, "language_model"):
            model.llama_model.base_model.language_model.config.vocab_size = new_vocab_size

    print("Post-PEFT top-level vocab size:", model.llama_model.config.vocab_size)
    print("Post-PEFT base_model vocab size:", model.llama_model.base_model.config.vocab_size)
    print("Post-PEFT LM vocab size:", model.llama_model.base_model.language_model.config.vocab_size)

    # Ensure input/output embedding sizes match
    input_emb = model.llama_model.get_input_embeddings()
    output_emb = model.llama_model.get_output_embeddings()
    if input_emb.weight.shape[0] != output_emb.weight.shape[0]:
        print(f"Updating LM head: {input_emb.weight.shape[0]} vs {output_emb.weight.shape[0]}")
        new_weight = torch.zeros(input_emb.weight.shape[0], input_emb.weight.shape[1],
                                   dtype=output_emb.weight.dtype,
                                   device=output_emb.weight.device)
        new_weight[:output_emb.weight.shape[0], :] = output_emb.weight.data
        output_emb.weight = nn.Parameter(new_weight)
        model.llama_model.set_output_embeddings(output_emb)
        model.llama_model.tie_weights()

    # Optimizer
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.AdamW(trainable_params, lr=lr)

    # Tracking
    train_losses = []
    train_accs = []
    best_acc = 0.0
    best_epoch = 0

    # Predefined English prompts
    english_prompts = [
        "From the categories [neu,ang,hap,sad], which single label best matches this audio? Return only one tag.",
        "Please decide if the speaker’s emotion is neu,ang,hap, or sad, and respond with exactly one of these tags only.",
        "Identify the speaker’s emotion strictly from [neu,ang,hap,sad]. Output only your chosen label.",
        "Among [neu,ang,hap,sad], which tag best describes the speaker's emotion in this clip? Provide that tag alone.",
        "Classify the audio using exactly one label from [neu,ang,hap,sad]. No other words allowed.",
        "Please pick one among [neu,ang,hap,sad] that fits the speaker’s mood, returning only that tag.",
        "Which label in [neu,ang,hap,sad] does this audio convey? Answer with just neu,ang,hap, or sad.",
        "Analyze the audio and select the single appropriate emotion from [neu,ang,hap,sad]. Output only that label.",
        "Kindly judge whether this audio is neu,ang,hap, or sad, and give only the corresponding tag.",
        "Please limit your answer to one of [neu,ang,hap,sad], describing the audio’s emotion. Return only that tag.",
        "Which emotion in [neu,ang,hap,sad] best matches the speaker’s expression? Reply with just that label.",
        "Identify the emotion from [neu,ang,hap,sad] in this audio. Respond with exactly one of these four tags.",
        "Out of [neu,ang,hap,sad], which single label does this speaker’s tone indicate? Give no other text beyond that tag.",
        "Determine if the speaker is neu,ang,hap, or sad based on the audio, and provide only the chosen tag.",
        "Select a label from [neu,ang,hap,sad] for the speaker’s emotion in this clip. Do not include any extra words.",
        "Among [neu,ang,hap,sad], please specify the single emotion reflected in the audio. Return only that label.",
        "Please classify the audio’s sentiment using exactly one label: neu,ang,hap, or sad. Output nothing else.",
        "From [neu,ang,hap,sad], pick the emotion that matches the speaker’s mood, and reply with that label alone.",
        "Which label (neu,ang,hap,sad) do you think the speaker conveys in this audio? Provide only that single tag.",
        "Analyze this audio and answer with one from [neu,ang,hap,sad]. No full words, no other emotions.",
        "You must choose the speaker’s emotion solely from [neu,ang,hap,sad]. Return only that one label, please.",
        "Listen to the clip, then respond with exactly one among [neu,ang,hap,sad]. No additional explanation.",
        "Please identify the emotion strictly in [neu,ang,hap,sad]. Provide only the label that best applies.",
        "Determine whether the audio’s emotion is neu,ang,hap, or sad, and give only that single tag.",
        "Kindly judge the audio’s sentiment. The valid labels are [neu,ang,hap,sad]. Output just the chosen label.",
        "Of [neu,ang,hap,sad], which best describes this audio’s emotion? Return only that exact tag.",
        "Select one label from [neu,ang,hap,sad] to classify the speaker’s emotion. Offer no extra words.",
        "Constrain your answer to neu,ang,hap,sad for the speaker’s feeling in this audio. Provide only the chosen tag.",
        "Please choose from [neu,ang,hap,sad] to describe the audio’s emotion, replying with that label alone.",
        "Identify the speaker’s emotion from these four tags [neu,ang,hap,sad], and respond solely with that one tag.",
    ]

    # Training loop
    model.train()
    for epoch in range(epochs):
        total_loss = 0.0
        num_correct = 0
        num_total = 0

        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}", ncols=100)
        for batch in pbar:
            padded_audios, audio_masks, padded_texts, text_masks, label_texts = batch

            # Exclude samples with label "fru"
            keep_indices = [i for i, lb in enumerate(label_texts) if lb != "fru"]
            if not keep_indices:
                continue

            padded_audios  = padded_audios[keep_indices].to(device)
            audio_masks    = audio_masks[keep_indices].to(device)
            padded_texts   = padded_texts[keep_indices].to(device)
            text_masks     = text_masks[keep_indices].to(device)
            label_texts    = [label_texts[i] for i in keep_indices]
            converted_labels = [map_label_to_token(lb) for lb in label_texts if map_label_to_token(lb) is not None]
            if not converted_labels:
                continue

            # Assign a random prompt to each sample
            prompt_texts = [random.choice(english_prompts) for _ in label_texts]

            optimizer.zero_grad()
            loss, outputs = model(
                audio_emb=padded_audios,
                audio_mask=audio_masks,
                text_emb=padded_texts,
                text_mask=text_masks,
                prompt_text=prompt_texts,
                label_text=converted_labels
            )
            loss.backward()
            optimizer.step()

            loss_val = loss.item()
            total_loss += loss_val

            logits = outputs.logits  # [B, seq_len, vocab_size]
            label_ids = model.llama_tokenizer(converted_labels, return_tensors='pt',
                                              padding=True, truncation=True).input_ids.to(device)
            print("Inside forward - Label ids range: min:", label_ids.min().item(),
                  "max:", label_ids.max().item(),
                  "Configured vocab_size:", model.llama_model.config.vocab_size)
            gold = label_ids[:, -1]
            predicted = logits[:, -1, :].argmax(dim=-1)

            batch_correct = (predicted == gold).sum().item()
            batch_size_ = predicted.size(0)

            num_correct += batch_correct
            num_total += batch_size_
            batch_acc = batch_correct / float(batch_size_)

            pbar.set_postfix({"loss": f"{loss_val:.4f}", "acc": f"{batch_acc:.4f}"})

        avg_loss = total_loss / len(dataloader)
        epoch_acc = num_correct / (num_total + 1e-9)
        train_losses.append(avg_loss)
        train_accs.append(epoch_acc)

        print(f"\n[Epoch {epoch+1}/{epochs}] | loss={avg_loss:.4f}, acc={epoch_acc:.4f}")

        # Save latest checkpoint
        latest_ckpt = "/root/uris/pth/llama_latest.pth"
        torch.save({
            'epoch': epoch+1,
            'model_state_dict': model.qformer.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': avg_loss,
            'acc': epoch_acc
        }, latest_ckpt)

        # Save best model if accuracy improves
        if epoch_acc > best_acc:
            best_acc = epoch_acc
            best_epoch = epoch+1
            best_ckpt_path = "/root/uris/pth/llama_best.pth"
            torch.save({
                'epoch': best_epoch,
                'model_state_dict': model.qformer.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
                'acc': best_acc
            }, best_ckpt_path)
            print(f"[*] Best model so far (acc={best_acc:.4f}) -> {best_ckpt_path}")
        torch.cuda.empty_cache()

    # Save final LoRA and Q-Former weights
    torch.save({"model_state_dict": model.qformer.state_dict()},
               "/root/uris/pth/qformer_after_lora.pth")
    model.llama_model.save_pretrained("llama_lora_peft")

    print(f"Done training. Best epoch={best_epoch}, best_acc={best_acc:.4f}")
    print("Train Losses:", train_losses)
    print("Train Acc   :", train_accs)

    # Plot training curves
    epochs_list = range(1, epochs+1)
    plt.figure(figsize=(10,5))
    plt.subplot(1,2,1)
    plt.plot(epochs_list, train_losses, marker='o', label='Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Train Loss')
    plt.legend()
    plt.grid(True)

    plt.subplot(1,2,2)
    plt.plot(epochs_list, [acc*100 for acc in train_accs],
             marker='s', color='green', label='Accuracy(%)')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy(%)')
    plt.title('Train Accuracy')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig("/root/uris/png/loss_acc_curve.png")
    print("[*] Saved training curves to loss_acc_curve.png")
    plt.show()


if __name__ == "__main__":
    train_lora_sft(
        pt_file="/root/uris/output_features.pt",
        qformer_ckpt_path="/root/uris/pth/best_model_triplet.pth",
        # llama_ckpt_path="/root/.llama/checkpoints/Llama3.2-11B-Vision-Instruct",
        batch_size=8,
        epochs=3,
        lr=1e-4,
        seed=42,
        lora_r=8,
        lora_alpha=32,
        lora_dropout=0.05,
        lora_target_modules="q_proj,v_proj",
        audio_hidden_dim=768,
        text_hidden_dim=768,
        llama_hidden_size=4096,
        device="cuda"
    )
