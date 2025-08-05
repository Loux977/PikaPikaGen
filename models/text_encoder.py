import torch
import torch.nn as nn
from transformers import CLIPModel, CLIPProcessor, BertModel, BertTokenizer

class TextEncoder(nn.Module):
    def __init__(self, encoder_type="clip", embedding_dim=256, device="cuda", num_last_finetune_layers=1):
        super().__init__()
        self.device = device
        self.encoder_type = encoder_type.lower()

        if self.encoder_type == "clip":
            self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(self.device)
            self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32", use_fast=True)
            self.input_dim = self.clip_model.config.text_config.hidden_size # 512 for ViT-B/32

            # Add projection layer to match target embedding dim
            if self.input_dim != embedding_dim:
                self.proj = nn.Linear(self.input_dim, embedding_dim).to(self.device)
                # Ensure the projection layer matches the CLIP model's dtype
                self.proj = self.proj.to(self.clip_model.text_model.encoder.layers[0].self_attn.q_proj.weight.dtype)
            else:
                self.proj = nn.Identity()

            # Freeze all layers first
            for param in self.clip_model.parameters():
                param.requires_grad = False

            # Unfreeze Embedding Layer
            for param in self.clip_model.text_model.embeddings.parameters():
                param.requires_grad = True

            # Unfreeze last N transformer layers of the text model
            total_layers = len(self.clip_model.text_model.encoder.layers)
            for i in range(total_layers - num_last_finetune_layers, total_layers):
                for param in self.clip_model.text_model.encoder.layers[i].parameters():
                    param.requires_grad = True

            # Unfreeze LayerNorms and final projection
            for name, param in self.clip_model.text_model.named_parameters():
                if "layer_norm" in name or "text_projection" in name:
                    param.requires_grad = True

        elif self.encoder_type == "bert":
            self.bert_model = BertModel.from_pretrained("prajjwal1/bert-mini").to(self.device)
            self.bert_tokenizer = BertTokenizer.from_pretrained("prajjwal1/bert-mini")
            self.input_dim = self.bert_model.config.hidden_size

            # Add projection layer to match target embedding dim
            if self.input_dim != embedding_dim:
                self.proj = nn.Linear(self.input_dim, embedding_dim).to(self.device)
            else:
                self.proj = nn.Identity()

            for param in self.bert_model.parameters():
                param.requires_grad = False

            # Unfreeze Embedding Layer
            for param in self.bert_model.embeddings.parameters():
                param.requires_grad = True

            # Unfreeze last N transformer layers of the text model
            total_layers = len(self.bert_model.encoder.layer)
            for i in range(total_layers - num_last_finetune_layers, total_layers):
                for param in self.bert_model.encoder.layer[i].parameters():
                    param.requires_grad = True

            # Unfreeze LayerNorms and final projection
            for name, param in self.bert_model.named_parameters():
                if "LayerNorm" in name or "pooler" in name:
                    param.requires_grad = True

        else:
            raise ValueError(f"Unknown encoder_type: {encoder_type}")

    def forward(self, texts):
        if self.encoder_type == "clip":
            inputs = self.clip_processor.tokenizer(text=texts, return_tensors="pt", padding=True, truncation=True).to(self.device)
            outputs = self.clip_model.text_model(**inputs, output_hidden_states=True, return_dict=True)

            per_token_embeddings = outputs.last_hidden_state
            global_embeddings = self.clip_model.get_text_features(**inputs)

            # Project
            per_token_embeddings = self.proj(per_token_embeddings)
            global_embeddings = self.proj(global_embeddings)

            # Re-normalize after projection
            global_embeddings = global_embeddings / global_embeddings.norm(dim=-1, keepdim=True)

            return per_token_embeddings.float(), global_embeddings.float()

        elif self.encoder_type == "bert":
            tokens = self.bert_tokenizer(texts, return_tensors="pt", padding=True).to(self.device)
            outputs = self.bert_model(**tokens)

            per_token_embeddings = outputs.last_hidden_state
            global_embeddings = outputs.pooler_output

            # Project
            per_token_embeddings = self.proj(per_token_embeddings)
            global_embeddings = self.proj(global_embeddings)

            # Normalize after projection
            global_embeddings = global_embeddings / global_embeddings.norm(dim=-1, keepdim=True)

            return per_token_embeddings, global_embeddings