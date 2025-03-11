#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt

# Set project root path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from uris.models.qformer import Blip2QformerAudioText


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class MyEmbeddingDataset(Dataset):
    def __init__(self, data_list, label2idx=None):
        self.data_list = data_list
        if label2idx is None:
            unique_labels = sorted(list(set(d['label'] for d in data_list)))
            self.label2idx = {lab: i for i, lab in enumerate(unique_labels)}
        else:
            self.label2idx = label2idx

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        entry = self.data_list[idx]
        label_str = entry['label']
        label = self.label2idx[label_str]
        audio_emb = entry['audio_emb']  # Audio embedding [Ta, 1024]
        text_emb  = entry['text_emb']   # Text embedding [Tt, 768]
        return audio_emb, text_emb, label


def collate_fn_qformer(batch):
    audio_feats = []
    text_feats  = []
    labels      = []

    for (af, tf, lb) in batch:
        audio_feats.append(af)
        text_feats.append(tf)
        labels.append(lb)

    maxA = max(a.shape[0] for a in audio_feats)
    maxT = max(t.shape[0] for t in text_feats)
    A_dim = audio_feats[0].shape[1]  # 1024
    T_dim = text_feats[0].shape[1]   # 768

    padded_audios = []
    padded_texts  = []
    audio_masks   = []
    text_masks    = []

    for af, tf in zip(audio_feats, text_feats):
        A_len = af.shape[0]
        T_len = tf.shape[0]
        p_a = torch.zeros(maxA, A_dim)
        p_t = torch.zeros(maxT, T_dim)
        p_a[:A_len, :] = af
        p_t[:T_len, :] = tf

        padded_audios.append(p_a)
        padded_texts.append(p_t)

        a_mask = torch.zeros(maxA, dtype=torch.bool)
        a_mask[:A_len] = 1
        t_mask = torch.zeros(maxT, dtype=torch.bool)
        t_mask[:T_len] = 1
        audio_masks.append(a_mask)
        text_masks.append(t_mask)

    padded_audios = torch.stack(padded_audios, dim=0)  # [B, maxA, 1024]
    audio_masks   = torch.stack(audio_masks,   dim=0)  # [B, maxA]
    padded_texts  = torch.stack(padded_texts,  dim=0)   # [B, maxT, 768]
    text_masks    = torch.stack(text_masks,    dim=0)   # [B, maxT]
    labels        = torch.tensor(labels, dtype=torch.long)

    return padded_audios, audio_masks, padded_texts, text_masks, labels


class DeepMLP(nn.Module):
    def __init__(self,
                 audio_dim=768,
                 text_dim=768,
                 hidden_dims=[1024, 512, 256, 128, 64],
                 num_classes=6,
                 dropout=0.0):
        super(DeepMLP, self).__init__()
        input_dim = audio_dim + text_dim  # 1536 total

        layers = []
        prev_dim = input_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, h_dim))
            layers.append(nn.ReLU(inplace=True))
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev_dim = h_dim

        layers.append(nn.Linear(prev_dim, num_classes))
        self.mlp = nn.Sequential(*layers)

    def forward(self, audio_vec, text_vec):
        x = torch.cat([audio_vec, text_vec], dim=1)
        out = self.mlp(x)
        return out


def main(
    batch_size=8,
    num_epochs=50,
    lr=1e-5,
    betas=(0.9, 0.999),           # AdamW betas
    weight_decay=1e-6,            # AdamW weight decay
    club_alpha=0.7,
    club_hidden_size=512,
    is_sampled_version=False,
    lambda_sclub=0.001,
    C=1000
):
    """
    Unified hyperparameter configuration.
    """
    set_seed(42)

    # Load data
    pt_file = "/root/uris/output_features.pt"
    data_list = torch.load(pt_file)
    # Label processing (if needed)
    """
    for d in data_list:
        if d['label'] == 'exc':
            d['label'] = 'hap'
    data_list = [d for d in data_list if d['label'] != 'fru']
    """

    unique_labels = sorted(list(set(d['label'] for d in data_list)))
    print("Unique labels:", unique_labels)
    print("Total samples:", len(data_list))

    # Shuffle and split data
    random.shuffle(data_list)
    n_total = len(data_list)
    n_train = int(n_total * 0.9)
    train_data_list = data_list[:n_train]
    test_data_list  = data_list[n_train:]
    print(f"train={len(train_data_list)}, test={len(test_data_list)}")

    # Create datasets and loaders
    train_dataset = MyEmbeddingDataset(train_data_list)
    test_dataset  = MyEmbeddingDataset(test_data_list, label2idx=train_dataset.label2idx)
    train_loader  = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn_qformer)
    test_loader   = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn_qformer)

    # Initialize Q-Former model
    qformer_model = Blip2QformerAudioText(
        audio_dim=1024,
        text_dim=768,
        hidden_size=768,
        n_audio_q=20,
        n_text_q=20,
        n_mix_q=12,
        cross_attention_freq=2,
        club_hidden_size=club_hidden_size,
        alpha=club_alpha,
        is_sampled_version=is_sampled_version
    )
    # Uncomment to load a checkpoint:
    # ckp = torch.load("/root/uris/pth/best_qformer.pth", map_location="cuda")
    # qformer_model.load_state_dict(ckp["model_state_dict"], strict=False)

    for param in qformer_model.parameters():
        param.requires_grad = True

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    qformer_model.to(device)

    # Initialize MLP classifier
    num_classes = len(train_dataset.label2idx)
    classifier = DeepMLP(
        audio_dim=768,
        text_dim=768,
        hidden_dims=[1024, 512, 256, 128, 64],
        num_classes=num_classes,
        dropout=0.1
    ).to(device)

    criterion = nn.CrossEntropyLoss()

    # Set up optimizer and cosine scheduler with 4% warmup
    optimizer = AdamW(
        list(qformer_model.parameters()) + list(classifier.parameters()),
        lr=lr,
        weight_decay=weight_decay,
        betas=betas
    )

    total_training_steps = len(train_loader) * num_epochs
    warmup_steps = int(total_training_steps * 0.04)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_training_steps
    )

    best_test_acc = 0.0
    train_accs = []
    test_accs  = []

    for epoch in range(num_epochs):
        # Training phase
        qformer_model.train()  
        classifier.train()

        train_correct = 0
        train_total   = 0
        total_loss    = 0.0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]")
        for padded_audios, audio_masks, padded_texts, text_masks, labels in pbar:
            padded_audios = padded_audios.to(device)
            audio_masks   = audio_masks.to(device)
            padded_texts  = padded_texts.to(device)
            text_masks    = text_masks.to(device)
            labels        = labels.to(device)

            # Forward pass: Q-Former and MLP
            out_dict = qformer_model(
                {
                    "audio_emb": padded_audios,
                    "audio_mask": audio_masks,
                    "text_emb": padded_texts,
                    "text_mask": text_masks
                },
                return_embeddings=True
            )
            audio_vec = out_dict["audio_all"]
            text_vec = out_dict["text_all"]

            logits = classifier(audio_vec, text_vec)
            ce_loss = criterion(logits, labels)

            sclub_loss = out_dict["stmil_loss"]
            sclub_loss = torch.clamp(sclub_loss, min=-C, max=C)

            total_loss_batch = ce_loss + (lambda_sclub * sclub_loss)

            # Backward pass and optimization
            optimizer.zero_grad()
            total_loss_batch.backward()
            optimizer.step()
            scheduler.step()

            total_loss += total_loss_batch.item()
            preds = logits.argmax(dim=-1)
            train_correct += (preds == labels).sum().item()
            train_total += labels.size(0)
            current_acc = train_correct / train_total
            pbar.set_postfix({"ce_loss": f"{ce_loss.item():.4f}",
                              "sclub_loss": f"{sclub_loss.item():.4f}",
                              "acc": f"{current_acc:.4f}"})

        epoch_train_acc = train_correct / train_total
        epoch_train_loss = total_loss / len(train_loader)

        # Evaluation phase
        qformer_model.eval()
        classifier.eval()
        test_correct = 0
        test_total = 0

        with torch.no_grad():
            for padded_audios, audio_masks, padded_texts, text_masks, labels in test_loader:
                padded_audios = padded_audios.to(device)
                audio_masks   = audio_masks.to(device)
                padded_texts  = padded_texts.to(device)
                text_masks    = text_masks.to(device)
                labels        = labels.to(device)

                out_dict = qformer_model(
                    {
                        "audio_emb": padded_audios,
                        "audio_mask": audio_masks,
                        "text_emb": padded_texts,
                        "text_mask": text_masks
                    },
                    return_embeddings=True
                )
                audio_vec = out_dict["audio_all"]
                text_vec = out_dict["text_all"]

                logits = classifier(audio_vec, text_vec)
                preds = logits.argmax(dim=-1)
                test_correct += (preds == labels).sum().item()
                test_total += labels.size(0)

        test_acc = test_correct / test_total if test_total > 0 else 0.0
        train_accs.append(epoch_train_acc)
        test_accs.append(test_acc)

        print(f"[Epoch {epoch+1}/{num_epochs}] train_loss={epoch_train_loss:.4f}, "
              f"train_acc={epoch_train_acc:.4f}, test_acc={test_acc:.4f}")

        # Save latest checkpoint
        torch.save({
            'epoch': epoch + 1,
            'qformer_state_dict': qformer_model.state_dict(),
            'classifier_state_dict': classifier.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_acc': epoch_train_acc,
            'test_acc': test_acc
        }, "/root/uris/pth/stmil_latest.pth")

        print(f"[*] Saved checkpoint at epoch {epoch+1} -> stmil_latest.pth")

        if test_acc > best_test_acc:
            best_test_acc = test_acc
            best_epoch = epoch + 1
            torch.save({
                'epoch': best_epoch,
                'qformer_state_dict': qformer_model.state_dict(),
                'classifier_state_dict': classifier.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'test_acc': best_test_acc
            }, "/root/uris/pth/stmil_best.pth")
            print(f"[*] New best model (test_acc={best_test_acc:.4f}) -> stmil_best.pth")

    print("Training done.")
    print("Train acc list:", train_accs)
    print("Test  acc list:", test_accs)
    print(f"Best test_acc={best_test_acc:.4f}")

    # Plot accuracy curves
    epochs_range = range(1, num_epochs+1)
    plt.figure(figsize=(8,6))
    plt.plot(epochs_range, train_accs, label="Train Acc", marker='o')
    plt.plot(epochs_range, test_accs, label="Test Acc", marker='s')
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.title("Train & Test Accuracy")
    plt.grid(True)
    out_fig = "/root/uris/png/acc_curve.png"
    plt.savefig(out_fig)
    plt.show()
    print(f"Saved figure: {out_fig}")


if __name__ == "__main__":
    # Modify hyperparameters as needed
    main(
        batch_size=8,
        num_epochs=50,
        lr=1e-5,
        betas=(0.9, 0.999),
        weight_decay=1e-6,
        club_alpha=0.7,
        club_hidden_size=512,
        is_sampled_version=False,
        lambda_sclub=0.001,
        C=1000
    )





