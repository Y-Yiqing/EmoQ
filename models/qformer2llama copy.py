import torch
import torch.nn as nn
from transformers import LlamaForCausalLM, LlamaTokenizer
from transformers import MllamaForConditionalGeneration, AutoProcessor
from uris.models.qformer import Blip2QformerAudioText

class QformerLlamaModel(nn.Module):
    def __init__(
        self,
        qformer_ckpt_path,
        #llama_ckpt_path,
        audio_hidden_dim=768,
        text_hidden_dim=768,
        llama_hidden_size=4096,
    ):
        super().__init__()

        # ========== 1) Q-Former ==========
        self.qformer = Blip2QformerAudioText()
        ckpt = torch.load(qformer_ckpt_path, map_location="cuda")
        self.qformer.load_state_dict(ckpt["model_state_dict"], strict=False)

        # ========== 2) LLaMA + tokenizer ==========
        model_id = "meta-llama/Llama-3.2-11B-Vision-Instruct"
        self.llama_model = MllamaForConditionalGeneration.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
        )
        self.llama_tokenizer = AutoProcessor.from_pretrained(model_id).tokenizer

        # special token: <AUDIO>, <TEXT>
        special_tokens_dict = {"additional_special_tokens": ["<AUDIO>", "<TEXT>","<NEU>", "<ANG>", "<HAP>", "<SAD>"]}
        self.llama_tokenizer.add_special_tokens(special_tokens_dict)
        vocab_dict = self.llama_tokenizer.get_vocab()
        new_vocab_size = max(vocab_dict.values()) + 1
        self.llama_model.resize_token_embeddings(new_vocab_size)
        self.llama_model.config.vocab_size = new_vocab_size


        if self.llama_tokenizer.pad_token is None:
            self.llama_tokenizer.pad_token = self.llama_tokenizer.eos_token

        # after tokenizer.add_special_tokens(...) + model.resize_token_embeddings(new_vocab_size)

        # 1) 在
        input_emb = self.llama_model.get_input_embeddings()
        out_emb = self.llama_model.get_output_embeddings()

        in_size, hidden_dim = input_emb.weight.size()
        out_size, hidden_dim2 = out_emb.weight.size()

        if in_size != out_size:
            # enlarge out_emb
            new_weight = torch.zeros(in_size, hidden_dim,
                                    dtype=out_emb.weight.dtype,
                                    device=out_emb.weight.device)
            new_weight[:out_size, :] = out_emb.weight.data
            out_emb.weight = nn.Parameter(new_weight)
            
            # update config
            self.llama_model.config.vocab_size = in_size
            self.llama_model.language_model.config.vocab_size = in_size  # 128263

            

            self.llama_model.set_output_embeddings(out_emb)
            
            # tie_weights 
            self.llama_model.tie_weights()

        # print
        print("input_embeddings:", input_emb.weight.shape)
        print("output_embeddings:", out_emb.weight.shape)
        print("config vocab:", self.llama_model.config.vocab_size)
        print("language_model.config.vocab_size:", self.llama_model.language_model.config.vocab_size)


        # ========== 3) 2MLP audio/text => 4096 ==========
        self.audio2llama = nn.Sequential(
            nn.Linear(audio_hidden_dim, 2048),
            nn.ReLU(),
            nn.Linear(2048, llama_hidden_size),
            #nn.ReLU()  
        )
        self.text2llama = nn.Sequential(
            nn.Linear(text_hidden_dim, 2048),
            nn.ReLU(),
            nn.Linear(2048, llama_hidden_size),
            #nn.ReLU()
        )

    def forward(
        self,
        audio_emb,   # [B, Ta, audio_hidden_dim]
        audio_mask,  # [B, Ta]
        text_emb,    # [B, Tt, text_hidden_dim]
        text_mask,   # [B, Tt]
        prompt_text, # list[str], length=B
        label_text=None
    ):
        device = audio_emb.device

        # ========== A) Q-Former ==========
        out_dict = self.qformer({
            "audio_emb":  audio_emb,
            "audio_mask": audio_mask,
            "text_emb":   text_emb,
            "text_mask":  text_mask
        }, return_embeddings=True)
        audio_vec_768 = out_dict["audio_all"]  # [B,768]
        text_vec_768  = out_dict["text_all"]   # [B,768]

        # ========== B) projection => 4096 ==========
        audio_vec_4096 = self.audio2llama(audio_vec_768).to(torch.bfloat16)
        text_vec_4096  = self.text2llama(text_vec_768).to(torch.bfloat16)

        batch_size = audio_vec_4096.shape[0]
        embedding_layer = self.llama_model.get_input_embeddings()

        # 1) BOS
        bos_tokens = torch.full((batch_size,1),
                                self.llama_tokenizer.bos_token_id,
                                dtype=torch.long, device=device)
        bos_embeds = embedding_layer(bos_tokens)

        # 2) <AUDIO> + audio_vec
        audio_token_id = self.llama_tokenizer.convert_tokens_to_ids("<AUDIO>")
        audio_token = torch.full((batch_size,1), audio_token_id, dtype=torch.long, device=device)
        audio_token_embed = embedding_layer(audio_token)
        audio_embeds = torch.cat([audio_token_embed, audio_vec_4096.unsqueeze(1)], dim=1)

        # 3) <TEXT> + text_vec
        text_token_id = self.llama_tokenizer.convert_tokens_to_ids("<TEXT>")
        text_token = torch.full((batch_size,1), text_token_id, dtype=torch.long, device=device)
        text_token_embed = embedding_layer(text_token)
        text_embeds = torch.cat([text_token_embed, text_vec_4096.unsqueeze(1)], dim=1)

        # 4) prompt_text (list[str]) => token -> embedding
    
        prompt_ids = self.llama_tokenizer(prompt_text, return_tensors='pt',
                                          padding=True, truncation=True).input_ids.to(device)
        prompt_embeds = embedding_layer(prompt_ids)

        # 5) label_text => optional
        if label_text is not None:
            label_ids = self.llama_tokenizer(label_text, return_tensors='pt',
                                             padding=True, truncation=True).input_ids.to(device)
            print("vocab_size =", len(self.llama_tokenizer),"label_ids max =", label_ids.max().item())
            label_embeds = embedding_layer(label_ids)
        else:
            label_ids = None
            label_embeds = None

        # ========== D) embedding ==========
        full_embeds = torch.cat([bos_embeds, audio_embeds, text_embeds, prompt_embeds], dim=1)

        # attention_mask
        bos_mask = torch.ones((batch_size,1), dtype=torch.long, device=device)
        audio_mask_= torch.ones((batch_size,audio_embeds.size(1)), dtype=torch.long, device=device)
        text_mask_ = torch.ones((batch_size,text_embeds.size(1)), dtype=torch.long, device=device)
        prompt_mask= (prompt_ids != self.llama_tokenizer.pad_token_id).long()
        full_mask  = torch.cat([bos_mask,audio_mask_,text_mask_,prompt_mask], dim=1)

        if label_embeds is not None:
            full_embeds = torch.cat([full_embeds, label_embeds], dim=1)
            label_mask_ = (label_ids != self.llama_tokenizer.pad_token_id).long()
            full_mask   = torch.cat([full_mask, label_mask_], dim=1)

        # ========== E) label token ==========
        if label_embeds is not None:
            total_seq_len = full_embeds.size(1)
            labels_tensor = -100*torch.ones((batch_size,total_seq_len),
                                            dtype=torch.long, device=device)
            label_len = label_ids.size(1)
            labels_tensor[:, -label_len:] = label_ids

            outputs = self.llama_model(
                inputs_embeds=full_embeds,
                attention_mask=full_mask,
                labels=labels_tensor
            )
            loss = outputs.loss
            return loss, outputs
        else:
            outputs = self.llama_model(
                inputs_embeds=full_embeds,
                attention_mask=full_mask
            )
            return None, outputs