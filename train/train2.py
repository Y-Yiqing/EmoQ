import random
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
import sys
import os
import gc

# Set env var to reduce GPU memory fragmentation
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# Clear GPU memory if needed
def clear_gpu_memory():
    gc.collect()
    torch.cuda.empty_cache()

clear_gpu_memory()

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from uris.dataset import MyDataset, collate_fn
from uris.models.qformer import Blip2QformerAudioText
from uris.models.triplet_loss import compute_triplet_loss_cosine

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Map labels: hap/exc->0, ang->1, sad->2, neu->3; skip 'fru'
def map_label_str_to_id(label_str):
    if label_str in ["hap", "exc"]:
        return 0
    elif label_str == "ang":
        return 1
    elif label_str == "sad":
        return 2
    elif label_str == "neu":
        return 3
    elif label_str == "fru":
        return None
    else:
        return None

class LabelEmbeddingModule(torch.nn.Module):
    def __init__(self, num_labels=4, hidden_dim=768):
        super().__init__()
        self.label_emb = torch.nn.Embedding(num_labels, hidden_dim)

    def forward(self, labels):
        return self.label_emb(labels)

class ProjectionMLP(torch.nn.Module):
    """Two-layer FC with ReLU to map 1536-D to 768-D."""
    def __init__(self, in_dim=1536, hidden_dim=768, out_dim=768):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(in_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, out_dim),
            torch.nn.ReLU()
        )
        
    def forward(self, x):
        return self.net(x)

def train_triplet(pt_file, model_ckpt='best_model.pth', epochs=3, batch_size=8,
                  lr=1e-4, margin=0.2, device='cuda'):
    """Stage 2: Align using triplet loss with emotion labels and save models."""
    set_seed(42)
    
    dataset = MyDataset(pt_file)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    
    model = Blip2QformerAudioText().to(device)
    ckpt = torch.load(model_ckpt, map_location=device)
    model.load_state_dict(ckpt['model_state_dict'], strict=False)
    print(f"[*] Loaded model from {model_ckpt}, epoch={ckpt.get('epoch', '?')}.")
    
    label_embed_module = LabelEmbeddingModule(num_labels=4, hidden_dim=768).to(device)
    proj_mlp = ProjectionMLP(in_dim=1536, hidden_dim=768, out_dim=768).to(device)
    
    for name, param in model.named_parameters():
        param.requires_grad = True
    for p in label_embed_module.parameters():
        p.requires_grad = True
    for p in proj_mlp.parameters():
        p.requires_grad = True
    
    print("[*] Trainable parameters:")
    total_params = 0
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(f"  - {name}: {list(param.shape)}")
            total_params += param.numel()
    for name, param in label_embed_module.named_parameters():
        if param.requires_grad:
            print(f"  - label_embed_module.{name}: {list(param.shape)}")
            total_params += param.numel()
    for name, param in proj_mlp.named_parameters():
        if param.requires_grad:
            print(f"  - proj_mlp.{name}: {list(param.shape)}")
            total_params += param.numel()
    print(f"[*] Total trainable params: {total_params:,}\n{'='*50}\n")
    
    optimizer = optim.Adam(list(model.parameters()) +
                           list(label_embed_module.parameters()) +
                           list(proj_mlp.parameters()), lr=lr)
    
    best_loss = float('inf')
    best_epoch = 0
    train_losses = []
    
    for epoch in range(epochs):
        total_loss = 0.0
        model.train()
        label_embed_module.train()
        proj_mlp.train()
        data_iter = tqdm(dataloader, desc=f"[Triplet] Epoch {epoch+1}/{epochs}", ncols=100)
        
        for batch_idx, (padded_audios, audio_masks, padded_texts, text_masks, label_strs) in enumerate(data_iter):
            mapped_ids = []
            keep_indices = []
            for i, lb_str in enumerate(label_strs):
                mapped_id = map_label_str_to_id(lb_str)
                if mapped_id is not None:
                    mapped_ids.append(mapped_id)
                    keep_indices.append(i)
            if len(keep_indices) == 0:
                continue
            
            padded_audios_ = padded_audios[keep_indices]
            audio_masks_   = audio_masks[keep_indices]
            padded_texts_  = padded_texts[keep_indices]
            text_masks_    = text_masks[keep_indices]
            labels_tensor  = torch.LongTensor(mapped_ids)
            
            padded_audios_ = padded_audios_.to(device)
            audio_masks_ = audio_masks_.to(device)
            padded_texts_ = padded_texts_.to(device)
            text_masks_ = text_masks_.to(device)
            labels_tensor = labels_tensor.to(device)
            
            samples = {
                "audio_emb": padded_audios_,
                "audio_mask": audio_masks_,
                "text_emb": padded_texts_,
                "text_mask": text_masks_
            }
            optimizer.zero_grad()
            out_dict = model(samples, return_embeddings=True)
            label_vec = label_embed_module(labels_tensor)
            
            triplet_loss = compute_triplet_loss_cosine(
                audio_vec=out_dict["audio_vec"],
                text_vec=out_dict["text_vec"],
                audio_mix=out_dict["audio_mix"],
                text_mix=out_dict["text_mix"],
                labels=labels_tensor,
                margin=margin,
                label_emb=label_vec,
                proj_mlp=proj_mlp
            )
            if not triplet_loss.requires_grad:
                dummy = next(label_embed_module.parameters()).sum() * 0.0
                triplet_loss = triplet_loss + dummy
            
            triplet_loss.backward()
            optimizer.step()
            
            loss_val = triplet_loss.item()
            total_loss += loss_val
            data_iter.set_postfix({"batch_loss": f"{loss_val:.6f}"})
        
        avg_epoch_loss = total_loss / len(dataloader)
        train_losses.append(avg_epoch_loss)
        print(f"\n[Epoch {epoch+1}/{epochs}] triplet loss = {avg_epoch_loss:.6f}")
        
        latest_ckpt = {
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'label_emb_state_dict': label_embed_module.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': avg_epoch_loss
        }
        torch.save(latest_ckpt, "model_latest_triplet.pth")
        
        if avg_epoch_loss < best_loss:
            best_loss = avg_epoch_loss
            best_epoch = epoch + 1
            best_ckpt = {
                'epoch': best_epoch,
                'model_state_dict': model.state_dict(),
                'label_emb_state_dict': label_embed_module.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': best_loss
            }
            torch.save(best_ckpt, "best_model_triplet.pth")
            print("  [*] Saved best triplet model to best_model_triplet.pth")
    
    print(f"Done! Best epoch = {best_epoch}, loss = {best_loss:.6f}")
    print("Epoch losses:", train_losses)
    
    plt.figure(figsize=(8,5))
    plt.plot(range(1, epochs+1), train_losses, marker='o', label='Triplet Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Triplet Loss Over Epochs')
    plt.legend()
    plt.grid(True)
    plt.savefig("triplet_loss_curve.png")
    print("[*] Saved loss curve to triplet_loss_curve.png")
    plt.show()

if __name__ == '__main__':
    pt_file = "output_features.pt"
    model_ckpt = "latest_model.pth"  # Load latest or best checkpoint as needed
    set_seed(42)
    train_triplet(pt_file, model_ckpt=model_ckpt, epochs=50, batch_size=8, lr=1e-4, margin=0.2, device='cuda')





