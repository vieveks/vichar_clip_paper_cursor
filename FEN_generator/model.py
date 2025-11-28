
import torch
import torch.nn as nn
import open_clip

class ChessFENGenerator(nn.Module):
    """
    Encoder-Decoder model for generating FEN strings from chess board images.
    Encoder: Pretrained CLIP Vision Model (ViT)
    Decoder: Transformer Decoder
    """
    def __init__(self, encoder_name="ViT-B-32", pretrained="laion2B-s34B-b79K", 
                 vocab_size=30, d_model=512, nhead=8, num_decoder_layers=6, 
                 dim_feedforward=2048, dropout=0.1, max_len=80):
        super().__init__()
        
        # 1. Encoder (CLIP Vision Model)
        # We load the full CLIP model but only keep the visual part
        clip_model, _, _ = open_clip.create_model_and_transforms(
            model_name=encoder_name,
            pretrained=pretrained
        )
        self.encoder = clip_model.visual
        
        # For ViT-B/32, intermediate features are 768-dim, final pooled is 512-dim
        # We use the intermediate spatial features for better spatial reasoning
        self.encoder_intermediate_dim = 768  # ViT-B/32 hidden dim
        self.encoder_output_dim = self.encoder.output_dim  # 512 (pooled)
        
        # Project intermediate features (768) to d_model (512)
        self.encoder_proj = nn.Linear(self.encoder_intermediate_dim, d_model)
            
        # 2. Decoder
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model, max_len=max_len)
        
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model, 
            nhead=nhead, 
            dim_feedforward=dim_feedforward, 
            dropout=dropout,
            batch_first=True
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_decoder_layers)
        
        # 3. Output Head
        self.fc_out = nn.Linear(d_model, vocab_size)
        
    def forward(self, images, tgt_tokens, tgt_mask=None, tgt_padding_mask=None):
        """
        Args:
            images: [batch_size, 3, 224, 224]
            tgt_tokens: [batch_size, seq_len] (Input tokens for decoder)
            tgt_mask: [seq_len, seq_len] (Causal mask)
            tgt_padding_mask: [batch_size, seq_len] (Padding mask)
        """
        # Extract spatial patch embeddings from CLIP encoder
        # forward_intermediates returns a dict with:
        # - 'image_intermediates': list of [B, 768, 7, 7] features from each layer
        # - 'image_features': [B, 512] pooled features (which we DON'T want)
        
        encoder_output = self.encoder.forward_intermediates(images)
        
        # Get the last layer's spatial features [B, 768, 7, 7]
        spatial_features = encoder_output['image_intermediates'][-1]
        
        # Reshape to sequence: [B, 768, 7, 7] -> [B, 768, 49] -> [B, 49, 768]
        batch_size = spatial_features.size(0)
        spatial_features = spatial_features.view(batch_size, self.encoder_intermediate_dim, -1)  # [B, 768, 49]
        spatial_features = spatial_features.permute(0, 2, 1)  # [B, 49, 768]
        
        # Project to d_model
        memory = self.encoder_proj(spatial_features)  # [B, 49, 512]
        
        # Decode
        tgt_emb = self.embedding(tgt_tokens) * math.sqrt(self.d_model)
        tgt_emb = self.pos_encoder(tgt_emb)
        
        output = self.decoder(
            tgt=tgt_emb, 
            memory=memory, 
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_padding_mask
        )
        
        logits = self.fc_out(output)
        return logits

    def generate(self, images, tokenizer, max_len=80, device='cuda', beam_size=5, min_length=35):
        """
        Beam search generation with minimum length constraint.
        
        Args:
            min_length: Minimum number of tokens to generate before allowing EOS (default 35)
        """
        self.eval()
        with torch.no_grad():
            # Extract spatial features
            encoder_output = self.encoder.forward_intermediates(images)
            spatial_features = encoder_output['image_intermediates'][-1]
            
            # Reshape to sequence
            batch_size = spatial_features.size(0)
            spatial_features = spatial_features.view(batch_size, self.encoder_intermediate_dim, -1)
            spatial_features = spatial_features.permute(0, 2, 1)
            
            # Project
            memory = self.encoder_proj(spatial_features)  # [batch_size, 49, d_model]
            
            # For simplicity, process one image at a time
            all_outputs = []
            for b in range(batch_size):
                # Get memory for this sample and expand for beam
                mem = memory[b:b+1]  # [1, 49, d_model]
                mem = mem.expand(beam_size, -1, -1)  # [beam_size, 49, d_model]
                
                # Initialize beams - all start with SOS
                current_seqs = torch.full((beam_size, 1), tokenizer.sos_token_id, dtype=torch.long, device=device)
                current_scores = torch.zeros(beam_size, device=device)
                current_scores[1:] = -float('inf')  # Only first beam active initially
                
                finished = []
                
                for step in range(max_len):
                    if len(finished) >= beam_size:
                        break
                    
                    # Get embeddings for current sequences
                    tgt_emb = self.embedding(current_seqs) * math.sqrt(self.d_model)
                    tgt_emb = self.pos_encoder(tgt_emb)
                    
                    # Decode
                    output = self.decoder(tgt=tgt_emb, memory=mem)
                    logits = self.fc_out(output[:, -1, :])  # [beam_size, vocab_size]
                    
                    # Mask EOS token if we haven't reached minimum length
                    current_len = current_seqs.size(1) - 1  # -1 for SOS token
                    if current_len < min_length:
                        logits[:, tokenizer.eos_token_id] = -float('inf')
                    
                    log_probs = torch.log_softmax(logits, dim=-1)
                    
                    # Expand scores: add log prob to each beam's score
                    vocab_size = log_probs.size(-1)
                    scores = current_scores.unsqueeze(1) + log_probs  # [beam_size, vocab_size]
                    
                    # For the first step, only use first beam to avoid duplicates
                    if step == 0:
                        scores = scores[0:1]  # [1, vocab_size]
                    
                    # Flatten and get top candidates
                    scores_flat = scores.view(-1)  # [beam_size * vocab_size] or [vocab_size]
                    top_scores, top_indices = torch.topk(scores_flat, min(beam_size * 2, len(scores_flat)))
                    
                    # Track new candidates
                    new_seqs = []
                    new_scores = []
                    
                    for score, idx in zip(top_scores, top_indices):
                        if step == 0:
                            beam_idx = 0
                            token_idx = idx.item()
                        else:
                            beam_idx = idx.item() // vocab_size
                            token_idx = idx.item() % vocab_size
                        
                        # Create new sequence
                        new_seq = torch.cat([current_seqs[beam_idx], torch.tensor([token_idx], device=device)])
                        
                        # Check if EOS (but enforce minimum length)
                        current_len = len(new_seq) - 1  # Subtract 1 for SOS token
                        if token_idx == tokenizer.eos_token_id and current_len >= min_length:
                            finished.append((new_seq, score.item()))
                        elif token_idx == tokenizer.eos_token_id and current_len < min_length:
                            # Reject premature EOS - don't add to finished or new_seqs
                            continue
                        else:
                            new_seqs.append(new_seq)
                            new_scores.append(score.item())
                        
                        # Stop if we have enough candidates
                        if len(new_seqs) >= beam_size:
                            break
                    
                    if len(new_seqs) == 0:
                        break
                    
                    # Pad sequences to same length for batching
                    max_len_seq = max(len(s) for s in new_seqs)
                    current_seqs = torch.stack([
                        torch.cat([s, torch.full((max_len_seq - len(s),), tokenizer.pad_token_id, dtype=torch.long, device=device)])
                        if len(s) < max_len_seq else s
                        for s in new_seqs
                    ])
                    current_scores = torch.tensor(new_scores, device=device)
                
                # Select best sequence
                if finished:
                    best_seq = max(finished, key=lambda x: x[1] / len(x[0]))[0]  # Normalize by length
                else:
                    best_seq = current_seqs[current_scores.argmax()]
                
                all_outputs.append(best_seq.unsqueeze(0))
            
            # Pad all outputs to same length
            max_out_len = max(o.size(1) for o in all_outputs)
            padded = []
            for o in all_outputs:
                if o.size(1) < max_out_len:
                    pad = torch.full((1, max_out_len - o.size(1)), tokenizer.pad_token_id, dtype=torch.long, device=device)
                    o = torch.cat([o, pad], dim=1)
                padded.append(o)
            
            return torch.cat(padded, dim=0)

import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Args:
            x: Tensor, shape [batch_size, seq_len, embedding_dim]
        """
        x = x + self.pe[:x.size(1)].transpose(0, 1)
        return self.dropout(x)
