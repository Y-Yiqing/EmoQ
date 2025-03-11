import os
import sys
import random
import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
import gc

from transformers import get_cosine_schedule_with_warmup

# Set project root and add to sys.path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from uris.dataset import MyDataset, collate_fn
from uris.models.qformer import Blip2QformerAudioText
from uris.models.triplet_loss import compute_triplet_loss_cosine

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

def clear_gpu_memory():
    gc.collect()
    torch.cuda.empty_cache()

clear_gpu_memory()

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def map_label_str_to_id(label_str):
    """Map emotion label string to ID."""
    mapping = {"hap": 0, "exc": 1, "ang": 2, "sad": 3, "neu": 4, "fru": 5}
    return mapping.get(label_str, None)

class LabelEmbeddingModule(nn.Module):
    def __init__(self, num_labels=6, hidden_dim=768):
        super().__init__()
        self.label_emb = nn.Embedding(num_labels, hidden_dim)

    def forward(self, labels):
        return self.label_emb(labels)

class ProjectionMLP(nn.Module):
    """Projection MLP to map concatenated features to 768."""
    def __init__(self, in_dim=1536, hidden_dim=768, out_dim=768):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.PReLU(),
            nn.Linear(hidden_dim, out_dim),
            nn.PReLU()
        )

    def forward(self, x):
        return self.net(x)

class ClassifierMLP(nn.Module):
    """6-class classifier with residual connections."""
    def __init__(self, in_dim=1536, hidden_dim=512, out_dim=6, dropout=0.1):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, out_dim)
        self.res_fc1 = nn.Linear(in_dim, hidden_dim)
        self.res_fc2 = nn.Linear(hidden_dim, out_dim)
        self.activation = nn.PReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, audio_vec, text_vec):
        x = torch.cat([audio_vec, text_vec], dim=1)
        x_res = self.res_fc1(x)
        x = self.fc1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = x + x_res
        x_res = self.res_fc2(x)
        x = self.fc2(x)
        x = self.activation(x)
        x = x + x_res
        return x

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
        label = self.label2idx[entry['label']]
        return entry['audio_emb'], entry['text_emb'], label

def collate_fn_qformer(batch):
    audio_feats, text_feats, labels = [], [], []
    for af, tf, lb in batch:
        audio_feats.append(af)
        text_feats.append(tf)
        labels.append(lb)

    maxA = max(a.shape[0] for a in audio_feats)
    maxT = max(t.shape[0] for t in text_feats)
    A_dim, T_dim = audio_feats[0].shape[1], text_feats[0].shape[1]

    padded_audios, padded_texts, audio_masks, text_masks = [], [], [], []
    for af, tf in zip(audio_feats, text_feats):
        A_len, T_len = af.shape[0], tf.shape[0]
        p_a = torch.zeros(maxA, A_dim)
        p_t = torch.zeros(maxT, T_dim)
        p_a[:A_len] = af
        p_t[:T_len] = tf
        padded_audios.append(p_a)
        padded_texts.append(p_t)
        a_mask = torch.zeros(maxA, dtype=torch.bool)
        t_mask = torch.zeros(maxT, dtype=torch.bool)
        a_mask[:A_len] = 1
        t_mask[:T_len] = 1
        audio_masks.append(a_mask)
        text_masks.append(t_mask)

    return (torch.stack(padded_audios, 0),
            torch.stack(audio_masks, 0),
            torch.stack(padded_texts, 0),
            torch.stack(text_masks, 0),
            torch.tensor(labels, dtype=torch.long))

def train_triplet_and_classify(pt_file,
                               output_dir='/root/uris/pth/',
                               epochs=10,
                               batch_size=8,
                               lr=1e-4,
                               betas=(0.9, 0.999),
                               weight_decay=1e-6,
                               dropout=0.1,
                               margin=0.2,
                               alpha_triplet=1.0,
                               device='cuda'):
    """Train using Triplet Loss and Cross-Entropy."""
    set_seed(42)
    data_list = torch.load(pt_file)
    unique_labels = sorted(list(set(d['label'] for d in data_list)))
    print("Unique labels:", unique_labels)
    print("Total samples:", len(data_list))
    random.shuffle(data_list)
    n_total = len(data_list)
    n_train = int(n_total * 0.9)
    train_data_list = data_list[:n_train]
    test_data_list = data_list[n_train:]
    print(f"train={len(train_data_list)}, test={len(test_data_list)}")
    
    train_dataset = MyEmbeddingDataset(train_data_list)
    test_dataset = MyEmbeddingDataset(test_data_list, label2idx=train_dataset.label2idx)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn_qformer)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn_qformer)

    model = Blip2QformerAudioText().to(device)
    label_embed_module = LabelEmbeddingModule(num_labels=6, hidden_dim=768).to(device)
    proj_mlp = ProjectionMLP(in_dim=1536, hidden_dim=768, out_dim=768).to(device)
    classifier_head = ClassifierMLP(in_dim=1536, hidden_dim=512, out_dim=6, dropout=dropout).to(device)

    for p in model.parameters():
        p.requires_grad = True
    for p in label_embed_module.parameters():
        p.requires_grad = True
    for p in proj_mlp.parameters():
        p.requires_grad = True
    for p in classifier_head.parameters():
        p.requires_grad = True

    all_params = (list(model.parameters()) +
                  list(label_embed_module.parameters()) +
                  list(proj_mlp.parameters()) +
                  list(classifier_head.parameters()))
    optimizer = AdamW(all_params, lr=lr, betas=betas, weight_decay=weight_decay)
    total_training_steps = len(train_loader) * epochs
    warmup_steps = int(total_training_steps * 0.04)
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_training_steps)
    ce_criterion = nn.CrossEntropyLoss()

    best_test_acc = 0.0
    best_epoch = 0
    train_accs = []
    test_accs = []

    for epoch in range(epochs):
        model.train()
        label_embed_module.train()
        proj_mlp.train()
        classifier_head.train()

        train_correct = 0
        train_total = 0
        total_ce_loss = 0.0
        total_trip_loss = 0.0
        total_loss_accum = 0.0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]")
        for padded_audios, audio_masks, padded_texts, text_masks, labels in pbar:
            padded_audios = padded_audios.to(device)
            audio_masks = audio_masks.to(device)
            padded_texts = padded_texts.to(device)
            text_masks = text_masks.to(device)
            labels = labels.to(device)

            out_dict = model({
                "audio_emb": padded_audios,
                "audio_mask": audio_masks,
                "text_emb": padded_texts,
                "text_mask": text_masks
            }, return_embeddings=True)
            audio_vec = out_dict["audio_vec"]
            text_vec = out_dict["text_vec"]
            audio_mix = out_dict["audio_mix"]
            text_mix = out_dict["text_mix"]

            label_vec = label_embed_module(labels)
            triplet_loss = compute_triplet_loss_cosine(
                audio_vec=audio_vec,
                text_vec=text_vec,
                audio_mix=audio_mix,
                text_mix=text_mix,
                labels=labels,
                margin=margin,
                label_emb=label_vec,
                proj_mlp=proj_mlp
            )

            logits_6 = classifier_head(audio_vec, text_vec)
            ce_loss = ce_criterion(logits_6, labels)
            total_loss = alpha_triplet * triplet_loss + ce_loss

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            scheduler.step()

            total_ce_loss += ce_loss.item()
            total_trip_loss += triplet_loss.item()
            total_loss_accum += total_loss.item()

            preds = logits_6.argmax(dim=-1)
            train_correct += (preds == labels).sum().item()
            train_total += labels.size(0)
            current_acc = train_correct / train_total

            pbar.set_postfix({
                "CE": f"{ce_loss.item():.4f}",
                "Triplet": f"{triplet_loss.item():.4f}",
                "Acc": f"{current_acc:.4f}"
            })

        epoch_train_acc = train_correct / train_total
        epoch_ce_loss = total_ce_loss / len(train_loader)
        epoch_trip_loss = total_trip_loss / len(train_loader)
        epoch_train_loss = total_loss_accum / len(train_loader)
        train_accs.append(epoch_train_acc)

        model.eval()
        label_embed_module.eval()
        proj_mlp.eval()
        classifier_head.eval()

        test_correct = 0
        test_total = 0
        with torch.no_grad():
            for padded_audios, audio_masks, padded_texts, text_masks, labels in test_loader:
                padded_audios = padded_audios.to(device)
                audio_masks = audio_masks.to(device)
                padded_texts = padded_texts.to(device)
                text_masks = text_masks.to(device)
                labels = labels.to(device)

                out_dict = model({
                    "audio_emb": padded_audios,
                    "audio_mask": audio_masks,
                    "text_emb": padded_texts,
                    "text_mask": text_masks
                }, return_embeddings=True)
                audio_all = out_dict["audio_all"]
                text_all = out_dict["text_all"]

                logits_6 = classifier_head(audio_all, text_all)
                preds = logits_6.argmax(dim=-1)
                test_correct += (preds == labels).sum().item()
                test_total += labels.size(0)

        test_acc = test_correct / test_total if test_total > 0 else 0.0
        test_accs.append(test_acc)

        print(f"[Epoch {epoch+1}/{epochs}] CE Loss={epoch_ce_loss:.4f}, Triplet Loss={epoch_trip_loss:.4f}, "
              f"Train Loss={epoch_train_loss:.4f}, Train Acc={epoch_train_acc:.4f}, Test Acc={test_acc:.4f}")

        ckpt_latest = {
            'epoch': epoch+1,
            'model_state_dict': model.state_dict(),
            'label_emb_state_dict': label_embed_module.state_dict(),
            'proj_mlp_state_dict': proj_mlp.state_dict(),
            'classifier_state_dict': classifier_head.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_acc': epoch_train_acc,
            'test_acc': test_acc
        }
        torch.save(ckpt_latest, os.path.join(output_dir, "triplet_cls_latest.pth"))
        print(f"[*] Saved latest checkpoint at epoch {epoch+1}")

        if test_acc > best_test_acc:
            best_test_acc = test_acc
            best_epoch = epoch + 1
            ckpt_best = {
                'epoch': best_epoch,
                'model_state_dict': model.state_dict(),
                'label_emb_state_dict': label_embed_module.state_dict(),
                'proj_mlp_state_dict': proj_mlp.state_dict(),
                'classifier_state_dict': classifier_head.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'test_acc': best_test_acc
            }
            torch.save(ckpt_best, os.path.join(output_dir, "triplet_cls_best.pth"))
            print(f"[*] New best model (Test Acc={best_test_acc:.4f})")

    print("\nTraining complete!")
    print(f"Best epoch: {best_epoch} with Test Acc: {best_test_acc:.4f}")
    print("Train acc list:", train_accs)
    print("Test acc list:", test_accs)

    epochs_range = range(1, epochs+1)
    plt.figure(figsize=(8,6))
    plt.plot(epochs_range, train_accs, label="Train Acc", marker='o')
    plt.plot(epochs_range, test_accs, label="Test Acc", marker='s')
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Triplet + CE Accuracy")
    plt.legend()
    plt.grid(True)
    out_fig = os.path.join(output_dir, "triplet_cls_acc_curve.png")
    plt.savefig(out_fig)
    plt.show()
    print(f"[*] Saved accuracy curve: {out_fig}")

if __name__ == '__main__':
    train_triplet_and_classify(
        pt_file="/root/uris/output_features.pt",
        output_dir="/root/uris/pth/",
        epochs=50,
        batch_size=32,
        lr=1e-4,
        betas=(0.9, 0.999),
        weight_decay=1e-6,
        dropout=0.2,
        margin=0.3,
        alpha_triplet=3,
        device='cuda'
    )
