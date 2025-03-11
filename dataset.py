import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader


class MyDataset(Dataset):
    def __init__(self, datalist, sessions=None):
        self.data_list = datalist
        # [{'Audio Embedding':..., 'Text Embedding':..., 'Label':...}, ...]
        for item in self.data_list:
            fn = item['filename']  # e.g. "Ses02F_..."
            sess_id = int(fn[4])   # parse it to int
            item['session_id'] = sess_id

        if sessions is not None:
            filtered = []
            for item in self.data_list:
                if item['session_id'] in sessions:
                    filtered.append(item)
            self.data_list = filtered

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        item = self.data_list[idx]
        audio_emb = item.get('audio_emb')  # shape [A_len, 1024]
        transcription = item.get('text')   # shape [T_len, 768]
        label = item.get('label')            # e.g. 'sad','hap','ang','fru'
        return audio_emb, transcription, label


def collate_fn(batch):
    """
    batch: list of (audio_emb, transcription, label)
    """
    audio_feats = []
    transcriptions = []
    labels = []

    for (af, tf, lb) in batch:
        audio_feats.append(af)
        transcriptions.append(tf)
        labels.append(lb)

    # find the max length
    A_dim = audio_feats[0].shape[1]  # 1024
    max_A_len = max(a.shape[0] for a in audio_feats)

    # padding
    padded_audios = []
    audio_masks = []

    for af in zip(audio_feats, transcriptions):
        A_len = af.shape[0]
        p_a = torch.zeros(max_A_len, A_dim)
        p_a[:A_len, :] = af
        padded_audios.append(p_a)

        # mask：1/0
        a_mask = torch.zeros(max_A_len, dtype=torch.bool)
        a_mask[:A_len] = 1
        audio_masks.append(a_mask)

    padded_audios = torch.stack(padded_audios, dim=0)  # [B, max_A_len, 1024]
    audio_masks = torch.stack(audio_masks,   dim=0)  # [B, max_A_len]

    return padded_audios, audio_masks, transcriptions, labels
