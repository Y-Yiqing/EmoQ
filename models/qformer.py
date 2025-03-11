import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from uris.lavis.models.blip_models.blip_outputs import BlipOutput
from uris.lavis.common.registry import registry

from uris.models.mi_estimators import SymmetricCLUB
from uris.lavis.models.blip2_models.Qformer import BertConfig, BertLMHeadModel

from safetensors.torch import load_file  
from transformers import BertTokenizer  

class Blip2QformerText(nn.Module):
    def __init__(
        self,
        audio_dim=1024,
        hidden_size=768,
        n_audio_q=32,
        cross_attention_freq=2,
        emb_dim=256,
        alpha=0.5,
        is_sampled_version=False,
        max_txt_len=44,
        k=2,
        margin=0.2,
        num_classes=4
    ):
        super().__init__()
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-cased")
        self.tokenizer.add_special_tokens({"bos_token": "[DEC]"})
        self.max_txt_len = max_txt_len
        self.audio_proj2qformer = nn.Linear(audio_dim, hidden_size)

        # ============ 1) Audio Q-Former ============
        self.Qformer, self.audio_query_tokens = self.init_qformer(
            config_path="/root/.cache/huggingface/hub/models--bert-base-cased/snapshots/cd5ef92a9fb2f889e972770a36d4ed042daf221e/config.json",
            ckpt_path="/root/.cache/huggingface/hub/models--bert-base-cased/snapshots/cd5ef92a9fb2f889e972770a36d4ed042daf221e/model.safetensors",
            num_query_token=n_audio_q,
            vision_width=hidden_size,
            cross_attention_freq=cross_attention_freq
        )
        self.Qformer.resize_token_embeddings(len(self.tokenizer))
        state_dict = self.Qformer.state_dict()
        for name, param in self.Qformer.named_parameters():
            if "_query" in name:
                key_orig = name.replace("_query", "")
                param.data.copy_(state_dict[key_orig])


        # Delete LM head
        self.Qformer.cls = None

        # FFN?
        # for layer in self.Qformer.bert.encoder.layer:
        #     layer.output = None
        #     layer.intermediate = None

        self.audio_proj = nn.Linear(self.Qformer.config.hidden_size,emb_dim)
        self.text_proj = nn.Linear(self.Qformer.config.hidden_size, emb_dim)
        self.temp = nn.Parameter(0.07 * torch.ones([]))
        self.ln_a = nn.LayerNorm(audio_dim)
        self.k=k
        self.margin=margin
        self.mha = nn.MultiheadAttention(embed_dim=self.Qformer.config.hidden_size, num_heads=1, batch_first=True)
        self.query = nn.Parameter(torch.randn(1, self.Qformer.config.hidden_size))
        self.classifier = nn.Linear(self.Qformer.config.hidden_size, num_classes)

    def init_qformer(self, config_path, ckpt_path, num_query_token, vision_width, cross_attention_freq=2):
        """
        Initialize BertLMHeadModel as Q-Former.
        """
        # 1) 
        encoder_config = BertConfig.from_pretrained(config_path)
        encoder_config.encoder_width = vision_width
        encoder_config.add_cross_attention = True
        encoder_config.cross_attention_freq = cross_attention_freq
        encoder_config.query_length = num_query_token

        # 2) 
        Qformer = BertLMHeadModel(config=encoder_config)

        # 3) 
        safetensor_weights = load_file(ckpt_path)     
        Qformer.load_state_dict(safetensor_weights, strict=False)

        # 4) query_tokens 
        query_tokens = nn.Parameter(
            torch.zeros(1, num_query_token, encoder_config.hidden_size)
        )
        query_tokens.data.normal_(mean=0.0, std=encoder_config.initializer_range)

        return Qformer, query_tokens

    def forward(self, samples):
        """
        Args:
            samples: dict, 
                - "audio_emb": [B, T_a, audio_dim]
                - "audio_mask": [B, T_a]
                - "text_emb": [B, T_t, text_dim]
                - "text_mask": [B, T_t]
            return_embeddings: bool
        """
        # 1) input
        audio_emb = samples["audio_emb"]
        audio_mask = samples["audio_mask"]
        audio_emb=self.ln_a(audio_emb)
        text=samples["transcriptions"]
        label = samples["label"]

        # 2) proj
        audio_emb = self.audio_proj2qformer(audio_emb)

        # 3) Query
        audio_query_tokens = self.audio_query_tokens.expand(audio_emb.shape[0], -1, -1)

        # 4) Audio Q-Former
        audio_out = self.Qformer.bert(
            query_embeds=audio_query_tokens,
            encoder_hidden_states=audio_emb,
            encoder_attention_mask=audio_mask,
            use_cache=True,
            return_dict=True
        )
        audio_feat =  F.normalize(
            self.audio_proj(audio_out.last_hidden_state), dim=-1
        )
        
        text_tokens = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=self.max_txt_len,
            return_tensors="pt",
        )
        text_output = self.Qformer.bert(
            text_tokens.input_ids,
            attention_mask=text_tokens.attention_mask,
            return_dict=True,
        )
        text_feat = F.normalize(
            self.text_proj(text_output.last_hidden_state[:, 0, :]), dim=-1
        )

        B, n_q, D = audio_feat.shape

        sim_a2t_all = torch.einsum("ikd,jd->ijk", audio_feat, text_feat)  # [B, B, n_q]
        sim_a2t, _ = sim_a2t_all.max(dim=-1)  # [B, B]
        sim_a2t = sim_a2t / self.temp

        #   sim_t2a[i, j] = max_k( text_feat[i] · audio_feat[j, k] )
        sim_t2a_all = torch.einsum("id,jkd->ijk", text_feat, audio_feat)  # [B, B, n_q]
        sim_t2a, _ = sim_t2a_all.max(dim=-1)  # [B, B]
        sim_t2a = sim_t2a / self.temp

        #   label = [0,1,2,...,B-1]
        labels = torch.arange(B, device=sim_a2t.device)
        loss_a2t = F.cross_entropy(sim_a2t, labels)  # audio->text
        loss_t2a = F.cross_entropy(sim_t2a, labels)  # text->audio
        loss_atc = (loss_a2t + loss_t2a) * 0.5

        #begin
        query_tokens_itm = self.audio_query_tokens.expand(text_tokens.input_ids.shape[0], -1, -1)
        query_atts_itm = torch.ones(
            query_tokens_itm.size()[:-1],  # [batch_size, n_query]
            dtype=torch.long,
            device=query_tokens_itm.device
        )

        attention_mask=torch.cat([query_atts_itm, text_tokens.attention_mask], dim=1)
        output_at=self.Qformer.bert(
            text_tokens.input_ids,
            query_embeds=query_tokens_itm,
            attention_mask=attention_mask,
            encoder_hidden_states=audio_emb,
            encoder_attention_mask=audio_mask,
            return_dict=True,
        )
        at_emb=output_at.last_hidden_state[:, :query_tokens_itm.size(1), :]

        B_at = at_emb.size(0)

        q = self.query.expand(B_at, -1, -1)  
        at_out, attn_weights = self.mha(q, at_emb, at_emb) 
        at_out = F.normalize(at_out.squeeze(1), dim=-1)
        B = at_out.size(0)
        sim_mat = torch.matmul(at_out, at_out.t())  # [B,B] dot
        dist_mat = 1.0 - sim_mat                         
        loss_sum = 0.0
        valid_count = 0

        for i in range(B):
            label_i = label[i]
            pos_mask = (label == label_i)
            pos_mask[i] = False
            pos_indices = torch.where(pos_mask)[0]

            neg_mask = (label != label_i)
            neg_indices = torch.where(neg_mask)[0]

            if len(pos_indices) < 1 or len(neg_indices) < 1:
                continue

            # ---------------- Hardest Positive ----------------
            pos_distances = dist_mat[i, pos_indices]  # shape [#pos]
            pos_sorted, _ = torch.sort(pos_distances, descending=True)
            hardest_pos_k = pos_sorted[:self.k]
            pos_mean = hardest_pos_k.mean()

            # ---------------- Hardest Negative ----------------
            neg_distances = dist_mat[i, neg_indices]  # shape [#neg]
            neg_sorted, _ = torch.sort(neg_distances, descending=False)
            hardest_neg_k = neg_sorted[:self.k]
            neg_mean = hardest_neg_k.mean()

            # ---------------- Triplet Margin ----------------
            loss_i = F.relu(self.margin + pos_mean - neg_mean)
            loss_sum += loss_i
            valid_count += 1

        if valid_count > 0:
            loss_triplet = loss_sum / valid_count
        else:
            loss_triplet = torch.tensor(0.0)

        logits = self.classifier(at_out)  # shape [B, num_classes]
        loss_ce = F.cross_entropy(logits, label)
        loss_total = loss_atc + loss_triplet + loss_ce

        return {
            "loss_atc": loss_atc,
            "loss_triplet": loss_triplet,
            "loss_ce": loss_ce,
            "loss_total": loss_total,
            "logits": logits
        }

