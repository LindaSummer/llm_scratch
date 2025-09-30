from typing import List
import tiktoken
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from dataclasses import dataclass

@dataclass(frozen=True)
class Config:
    vocab_size: int = 50257
    context_length: int = 1024
    emb_dim: int = 768
    num_heads: int = 12
    num_layers: int = 12
    dropout: float = 0.1
    qkv_bias: bool = False
    bias: bool = False


class DummyTransformerBlock(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        
    def forward(self, x):
        return x
    
class DummyLayerNorm(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        
    def forward(self, x):
        return x

class DummyGPTModel(nn.Module):
    def __init__(self, config: Config, 
                 transformer_cls=DummyTransformerBlock, 
                 norm_layer_cls=DummyLayerNorm):
        super().__init__()
        self._cfg = config
        self._tok_emd = nn.Embedding(config.vocab_size, config.emb_dim)
        self._pos_emd = nn.Embedding(config.context_length, config.emb_dim)
        self._dropout = nn.Dropout(config.dropout)
        self._transformer_blocks = nn.Sequential(
            *[transformer_cls(config) for _ in range(config.num_layers)]
        )
        self._final_norm = norm_layer_cls(config)
        self._out_head = nn.Linear(config.emb_dim, config.vocab_size, bias=False)
        
    def forward(self, in_idx: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len = in_idx.shape
        device = in_idx.device
        tok_embeds = self._tok_emd(in_idx)
        pos_embeds = self._pos_emd(
            torch.arange(seq_len, device=device)
        )
        x = tok_embeds + pos_embeds
        x = self._dropout(x)
        x = self._transformer_blocks(x)
        x = self._final_norm(x)
        return self._out_head(x)
    
class LayerNorm(nn.Module):
    def __init__(self, config: Config, eps=1e-5):
        super().__init__()
        self._eps = eps
        emb_dim = config.emb_dim
        self._scale = nn.Parameter(torch.ones(emb_dim))
        self._shift = nn.Parameter(torch.zeros(emb_dim))
        
    def forward(self, x: torch.Tensor):
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True)
        norm_x = (x - mean) / torch.sqrt(var + self._eps)
        return self._scale * norm_x + self._shift
    
class MultiHeadAttention(nn.Module):
    def __init__(self, config: Config):
        d_in = d_out = config.emb_dim
        assert (d_out % config.num_heads == 0), "d_out must be divisiable by num_heads"
        super().__init__()
        self._d_in = d_in
        self._d_out = d_out
        self._num_heads = config.num_heads
        self._d_head = d_out // config.num_heads
        self._w_q = nn.Linear(self._d_in, self._d_out, bias=config.qkv_bias)
        self._w_k = nn.Linear(self._d_in, self._d_out, bias=config.qkv_bias)
        self._w_v = nn.Linear(self._d_in, self._d_out, bias=config.qkv_bias)
        self._out_proj = nn.Linear(self._d_out, self._d_out)
        self._dropout = nn.Dropout(config.dropout)
        self.register_buffer("mask", torch.triu(torch.ones(config.context_length, config.context_length), diagonal=1))
        
    def forward(self, x):
        import math
        
        q = self._w_q(x)
        k = self._w_k(x)
        v = self._w_v(x)
        
        b, context_length, d_out = q.shape
        split_view = lambda x: x.view(b, context_length, self._num_heads, self._d_head)\
            .transpose(1, 2)
        q = split_view(q)
        k = split_view(k)
        v = split_view(v)
        
        attn_scores = q @ k.transpose(-1, -2) / math.sqrt(k.shape[-1])
        attn_scores.masked_fill_(self.mask.bool()[:context_length, :context_length], -torch.inf)
        weight = torch.softmax(attn_scores, dim=-1)
        weight = self._dropout(weight)
        
        z = weight @ v
        z = z.transpose(1, 2).contiguous().view(b, context_length, d_out)
        z = self._out_proj(z)
        return z
    
class FeedForward(nn.Module):
    def __init__(self,
                 config: Config,
                 ff_mid_dim:int=0,
                 ff_activation=nn.GELU()):
        super().__init__()
        if ff_mid_dim == 0:
            ff_mid_dim = 4 * config.emb_dim
        self._layers = nn.Sequential(
            nn.Linear(config.emb_dim, ff_mid_dim),
            ff_activation,
            nn.Linear(ff_mid_dim, config.emb_dim),
        )
        
    def forward(self, x):
        return self._layers(x)

class TransformerBlock(nn.Module):
    def __init__(self, config: Config, 
                 norm_layer_cls=LayerNorm,
                 feed_fwd_cls=FeedForward,
                 dropouts=None):
        if dropouts is None:
            dropouts = [config.dropout, config.dropout]
        super().__init__()
        self._norm_1 = norm_layer_cls(config=config)
        self._attention = MultiHeadAttention(config=config)
        self._drop_1 = nn.Dropout(dropouts[0])
        self._norm_2 = norm_layer_cls(config=config)
        self._ff = feed_fwd_cls(config=config)
        self._drop_2 = nn.Dropout(dropouts[1])
        
    def forward(self, x):
        shortcut = x
        x = self._norm_1(x)
        x = self._attention(x)
        x = self._drop_1(x)
        x += shortcut
        
        shortcut = x
        x = self._norm_2(x)
        x = self._ff(x)
        x = self._drop_2(x)
        x += shortcut
        
        return x
        
class GPTModel(nn.Module):
    def __init__(self,
                 config: Config,
                 gpt_dropout=-1,
                 transformer_cls=TransformerBlock,
                 norm_layer_cls=LayerNorm):
        super().__init__()
        self._tok_emd = nn.Embedding(config.vocab_size, config.emb_dim)
        self._pos_emd = nn.Embedding(config.context_length, config.emb_dim)
        if gpt_dropout < 0:
            gpt_dropout = config.dropout
        self._dropout = nn.Dropout(gpt_dropout)
        self._transformers = nn.Sequential(
            *[transformer_cls(config=config) for _ in range(config.num_layers)]
        )
        self._final_norm_layer = norm_layer_cls(config=config)
        self._out_head = nn.Linear(config.emb_dim, config.vocab_size, bias=config.bias)
        
    def forward(self, in_idx: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len = in_idx.shape
        token_embeds = self._tok_emd(in_idx)
        pos_embeds = self._pos_emd(torch.arange(seq_len, device=in_idx.device))

        x = token_embeds + pos_embeds
        x = self._dropout(x)
        x = self._transformers(x)
        x = self._final_norm_layer(x)
        return self._out_head(x)
        
def generate_text_trivial(model: nn.Module, 
                          idx: torch.Tensor,
                          max_new_tokens: int,
                          context_size: int):
    for _ in range(max_new_tokens):
        idx_context = idx[:, -context_size:]
        with torch.no_grad():
            logits = model(idx_context)
        new_token_logits = logits[:, -1, :]
        probas = torch.softmax(new_token_logits, dim=-1)
        idx_next = torch.argmax(probas, dim=-1, keepdim=True)
        idx = torch.cat((idx, idx_next), dim=1)
    return idx

class GPTDataSet(Dataset):
    def __init__(self, text, tokenizer, max_length, stride):
        self.input_ids = []
        self.target_ids = []
        
        token_ids = tokenizer.encode(text)
        for i in range(0, len(token_ids) - max_length, stride):
            input_chunk = token_ids[i:i+max_length]
            target_chunk = token_ids[i+1:i+max_length+1]
            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))
            
    def __len__(self):
        return len(self.input_ids)
    
    def __getitem__(self, idx):
        return self.input_ids[idx], self.target_ids[idx]
    

def create_dataloader(text, tokenizer, max_length=128, stride=64, batch_size=32, drop_last=True, shuffle=True, num_workers=0):
    dataset = GPTDataSet(text, tokenizer, max_length, stride)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers,
    )
