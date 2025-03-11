import torch
import torch.nn.functional as F
import torch.nn as nn

def compute_triplet_loss_cosine(
    audio_vec,
    text_vec,
    audio_mix,
    text_mix,
    labels,
    margin=0.2,
    w_mix2label=0.5,
    w_audio2text=0.2,
    w_audio2label=0.3,
    label_emb=None,
    proj_mlp=None
):
    """
    cosine similarity + ReLU(max(0, ...)) to compute Triplet Loss:
      1) mix2label:  (audio_mix + text_mix) vs. label
      2) audio2text:  audio_vec vs. text_vec
      3) audio2label: audio_vec vs. label

    Args:
      audio_vec:  [B, hidden]
      text_vec:   [B, hidden]
      audio_mix:  [B, hidden]
      text_mix:   [B, hidden]
      labels:     [B] (情感类别ID)
      margin (float): margin
      w_mix2label, w_audio2text, w_audio2label: weight
      label_emb:  [num_labels, hidden] or [B, hidden], 
      proj_mlp:   optional

    Returns:
      total_triplet_loss: weighted sum of (mix2label + audio2text + audio2label)
    """

    B = audio_vec.size(0)

    # ========== (A) label_vec ==========
    if label_emb is not None:
        # [B, hidden]
        label_vec = label_emb
    else:
        raise NotImplementedError("Please provide label_emb or define your own label embedding logic.")

    # ========== (B) proj_mlp==========
    mix_cat = torch.cat([audio_mix, text_mix], dim=-1)  # [B, hidden1+hidden2]
    if proj_mlp is not None:
        mix_vec = proj_mlp(mix_cat)  # [B, 768] 之类
    else:
        mix_vec = audio_mix + text_mix  

    # ========== (C) L2 normalize (eps) ==========
    mix_vec   = F.normalize(mix_vec,   p=2, dim=1, eps=1e-8)
    audio_vec = F.normalize(audio_vec, p=2, dim=1, eps=1e-8)
    text_vec  = F.normalize(text_vec,  p=2, dim=1, eps=1e-8)
    label_vec = F.normalize(label_vec, p=2, dim=1, eps=1e-8)

    device = audio_vec.device

    # ========== (D) ==========
    loss_mix2label   = torch.zeros((), device=device)
    loss_audio2text  = torch.zeros((), device=device)
    loss_audio2label = torch.zeros((), device=device)

    # ========== (E) 2 loops ==========
    for i in range(B):
        label_i = labels[i]

        # anchor
        anchor_mix   = mix_vec[i]    # [hidden]
        anchor_audio = audio_vec[i]
        # pos
        pos_label_i  = label_vec[i]
        pos_text_i   = text_vec[i]

        for j in range(B):
            if j == i:
                continue

            label_j = labels[j]
            if label_j != label_i:
                neg_label_j = label_vec[j]
                neg_text_j  = text_vec[j]

                # --- (1) mix2label ---
                #     ReLU[ cos(anchor_mix, neg_label_j) - cos(anchor_mix, pos_label_i) + margin ]
                cost_pos = F.cosine_similarity(anchor_mix.unsqueeze(0),
                                               pos_label_i.unsqueeze(0))
                cost_neg = F.cosine_similarity(anchor_mix.unsqueeze(0),
                                               neg_label_j.unsqueeze(0))
                diff = cost_neg - cost_pos + margin
                diff = torch.clamp(diff, min=-10., max=10.)  # clamp
                loss_mix2label = loss_mix2label + F.relu(diff)

                # --- (2) audio2text ---
                cost_pos_au = F.cosine_similarity(anchor_audio.unsqueeze(0),
                                                  pos_text_i.unsqueeze(0))
                cost_neg_au = F.cosine_similarity(anchor_audio.unsqueeze(0),
                                                  neg_text_j.unsqueeze(0))
                diff2 = cost_neg_au - cost_pos_au + margin
                diff2 = torch.clamp(diff2, min=-10., max=10.)
                loss_audio2text = loss_audio2text + F.relu(diff2)

                # --- (3) audio2label ---
                cost_pos_au2 = F.cosine_similarity(anchor_audio.unsqueeze(0),
                                                   pos_label_i.unsqueeze(0))
                cost_neg_au2 = F.cosine_similarity(anchor_audio.unsqueeze(0),
                                                   neg_label_j.unsqueeze(0))
                diff3 = cost_neg_au2 - cost_pos_au2 + margin
                diff3 = torch.clamp(diff3, min=-10., max=10.)
                loss_audio2label = loss_audio2label + F.relu(diff3)

    # ========== (F) ==========
    num_neg_pairs = B*(B-1)  # B*(B-1)
    if num_neg_pairs == 0:
        return torch.tensor(0.0, device=device, requires_grad=True)
    loss_mix2label   = loss_mix2label   / num_neg_pairs
    loss_audio2text  = loss_audio2text  / num_neg_pairs
    loss_audio2label = loss_audio2label / num_neg_pairs

    # ========== (G) total_loss==========
    total_loss = (w_mix2label * loss_mix2label
                  + w_audio2text * loss_audio2text
                  + w_audio2label * loss_audio2label)

    return total_loss
