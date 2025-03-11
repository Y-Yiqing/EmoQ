import random
import numpy as np
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim as optim
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
import sys
import os

from transformers import get_cosine_schedule_with_warmup

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from uris.dataset import MyDataset, collate_fn
from uris.models.qformer import Blip2QformerAudioText

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    cudnn.deterministic = True
    cudnn.benchmark = False

def compute_wa_ua(logits, labels, num_classes=4):
    """Compute Weighted Accuracy (WA) and Unweighted Accuracy (UA)."""
    preds = logits.argmax(dim=1)
    correct = (preds == labels).float()
    wa = correct.mean().item()

    class_correct = torch.zeros(num_classes, dtype=torch.float)
    class_count = torch.zeros(num_classes, dtype=torch.float)
    for c in range(num_classes):
        mask = (labels == c)
        c_count = mask.sum().item()
        if c_count > 0:
            class_count[c] = c_count
            class_correct[c] = (preds[mask] == c).sum().item()

    ua_vals = []
    for c in range(num_classes):
        if class_count[c] > 0:
            ua_vals.append(class_correct[c] / class_count[c])
    ua = sum(ua_vals) / len(ua_vals) if ua_vals else 0.0
    return wa, ua

@torch.no_grad()
def evaluate(model, loader, device, num_classes=4, ce_criterion=None):
    """
    Evaluate model: return (avg_loss, WA, UA).
    If ce_criterion is provided, we also compute CE Loss for reference.
    """
    model.eval()
    total_loss = 0.0
    total_wa = 0.0
    total_ua = 0.0
    steps = 0

    for padded_audios, audio_masks, padded_texts, labels in loader:
        padded_audios = padded_audios.to(device)
        audio_masks   = audio_masks.to(device)
        padded_texts  = padded_texts.to(device)
        label_tensor  = labels.to(device)

        samples = {
            "audio_emb": padded_audios,
            "audio_mask": audio_masks,
            "text_emb": padded_texts,
            "label": label_tensor
        }
        out = model(samples)
        loss_val = out["loss_total"]
        logits = out["logits"] if "logits" in out else None

        if logits is not None and ce_criterion is not None:
            # optional: log CE part if you like
            pass

        total_loss += loss_val.item()

        if logits is not None:
            wa, ua = compute_wa_ua(logits, label_tensor, num_classes)
            total_wa += wa
            total_ua += ua
        steps += 1

    avg_loss = total_loss / steps if steps>0 else 0.0
    avg_wa   = total_wa / steps if steps>0 else 0.0
    avg_ua   = total_ua / steps if steps>0 else 0.0
    return avg_loss, avg_wa, avg_ua

def train_one_fold(
    model, train_loader, test_loader, epochs, device,
    lr=1e-4, betas=(0.9,0.999), weight_decay=1e-6,
    num_classes=4
):
    """
    Train one fold, with AdamW + Cosine schedule + CE loss as example.
    """
    # Collect all params
    all_params = list(model.parameters())
    optimizer = AdamW(all_params, lr=lr, betas=betas, weight_decay=weight_decay)

    total_steps = len(train_loader)*epochs
    warmup_steps = int(total_steps*0.04)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
    )
    ce_criterion = nn.CrossEntropyLoss()

    train_losses = []
    train_was = []
    train_uas = []

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        total_wa = 0.0
        total_ua = 0.0
        steps = 0

        data_iter = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", ncols=100)
        for padded_audios, audio_masks, padded_texts, labels in data_iter:
            padded_audios = padded_audios.to(device)
            audio_masks   = audio_masks.to(device)
            padded_texts  = padded_texts.to(device)
            labels        = labels.to(device)

            samples = {
                "audio_emb": padded_audios,
                "audio_mask": audio_masks,
                "text_emb": padded_texts,
                "label": labels
            }

            optimizer.zero_grad()
            out = model(samples)
            loss_val = out["loss_total"]
            logits = out["logits"] if "logits" in out else None

            loss_val.backward()
            optimizer.step()
            scheduler.step()

            total_loss += loss_val.item()
            if logits is not None:
                wa, ua = compute_wa_ua(logits, labels, num_classes)
                total_wa += wa
                total_ua += ua

            steps += 1
            avg_wa = (total_wa/steps)*100
            avg_ua = (total_ua/steps)*100
            data_iter.set_postfix({"WA(%)": f"{avg_wa:.2f}", "UA(%)": f"{avg_ua:.2f}"})

        avg_loss = total_loss / steps if steps>0 else 0.0
        avg_wa   = total_wa / steps if steps>0 else 0.0
        avg_ua   = total_ua / steps if steps>0 else 0.0
        train_losses.append(avg_loss)
        train_was.append(avg_wa)
        train_uas.append(avg_ua)
        print(f"Epoch {epoch+1}/{epochs}: train_loss={avg_loss:.4f}, WA={avg_wa:.2f}%, UA={avg_ua:.2f}%")

    test_loss, test_wa, test_ua = evaluate(model, test_loader, device=device, num_classes=num_classes, ce_criterion=ce_criterion)
    return (test_loss, test_wa, test_ua), (train_losses, train_was, train_uas)

def train_loso_cv(
    pt_file,
    epochs=3,
    batch_size=8,
    lr=1e-4,
    betas=(0.9,0.999),
    weight_decay=1e-6,
    device='cuda',
    num_classes=4
):
    set_seed(42)
    # load data and do label cleaning
    data_list = torch.load(pt_file)
    for d in data_list:
        if d['label'] == 'exc':
            d['label'] = 'hap'
    data_list = [d for d in data_list if d['label'] != 'fru']

    n_sessions = 5
    fold_results = []

    for test_sess in range(1, n_sessions+1):
        train_sess = [s for s in range(1, n_sessions+1) if s != test_sess]
        train_set = MyDataset(data_list=data_list, sessions=train_sess)
        test_set  = MyDataset(data_list=data_list, sessions=[test_sess])

        train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
        test_loader  = DataLoader(test_set, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

        model = Blip2QformerAudioText(num_classes=num_classes).to(device)

        (test_loss, test_wa, test_ua), _ = train_one_fold(
            model=model,
            train_loader=train_loader,
            test_loader=test_loader,
            epochs=epochs,
            lr=lr,
            betas=betas,
            weight_decay=weight_decay,
            device=device,
            num_classes=num_classes
        )
        print(f"[Fold test_sess={test_sess}] => loss={test_loss:.4f}, WA={test_wa*100:.2f}%, UA={test_ua*100:.2f}%")
        fold_results.append((test_loss, test_wa, test_ua))

    all_losses = [r[0] for r in fold_results]
    all_was    = [r[1] for r in fold_results]
    all_uas    = [r[2] for r in fold_results]
    avg_loss   = sum(all_losses)/len(all_losses)
    avg_wa     = sum(all_was)/len(all_was)
    avg_ua     = sum(all_uas)/len(all_uas)

    print("\nLOSOCV results:")
    for i, (l, w, u) in enumerate(fold_results, start=1):
        print(f" Fold {i} => loss={l:.4f}, WA={w*100:.2f}%, UA={u*100:.2f}%")
    print(f" Avg: loss={avg_loss:.4f}, WA={avg_wa*100:.2f}%, UA={avg_ua*100:.2f}%")

if __name__ == "__main__":
    pt_file = "output_features.pt"
    train_loso_cv(
        pt_file=pt_file,
        epochs=5,         # can be changed
        batch_size=16,    # can be changed
        lr=1e-4,          # can be changed
        betas=(0.9,0.999),# can be changed
        weight_decay=1e-6,# can be changed
        device='cuda',
        num_classes=4
    )





