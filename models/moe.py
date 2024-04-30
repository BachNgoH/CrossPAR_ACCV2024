import torch.nn as nn
from soft_moe_pytorch import SoftMoE as MoE

class MoETransformer(nn.Module):
    def __init__(self, input_dim, num_heads, num_experts, seq_len=196):
        super(MoETransformer, self).__init__()
        self.multihead_attn = nn.MultiheadAttention(input_dim, num_heads, batch_first=True)
        self.moelayer = MoE(dim=input_dim, seq_len=seq_len, num_experts=num_experts)
        self.norm1 = nn.LayerNorm(input_dim)
        self.norm2 = nn.LayerNorm(input_dim)
    
    def forward(self, q, k, v):
        attn_output, _ = self.multihead_attn(q, k, v)
        
        x = self.norm1(q + attn_output)
        moe_output = self.moelayer(x)
        x = self.norm2(x + moe_output)
        return x
    
class MoEFusionHead(nn.Module):
    def __init__(self, input_dim, num_heads, num_experts):
        super(MoEFusionHead, self).__init__()
        self.transformer = MoETransformer(input_dim, num_heads, num_experts)
        self.pooling = nn.AdaptiveAvgPool1d(1)

    def forward(self, x1, x2):
        outputs = self.transformer(x1, x2, x2)
        outputs = self.pooling(outputs.transpose(1, 2)).squeeze(-1)
        return outputs