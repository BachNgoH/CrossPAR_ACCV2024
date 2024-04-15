import torch.nn as nn
from st_moe_pytorch import MoE

class MoETransformer(nn.Module):
    def __init__(self, input_dim, num_heads, num_experts):
        super(MoETransformer, self).__init__()
        self.multihead_attn = nn.MultiheadAttention(input_dim, num_heads, batch_first=True)
        self.moelayer = MoE(dim=input_dim, num_experts=num_experts)
        self.norm1 = nn.LayerNorm(input_dim)
        self.norm2 = nn.LayerNorm(input_dim)
    
    def forward(self, q, k, v):
        attn_output, _ = self.multihead_attn(q, k, v)
        
        x = self.norm1(q + attn_output)
        moe_output, total_aux_loss, _, _ = self.moelayer(x)
        x = self.norm2(x + moe_output)
        return x, total_aux_loss
    
class MoEFusionHead(nn.Module):
    def __init__(self, input_dim, num_heads, num_experts):
        super(MoEFusionHead, self).__init__()
        self.transformer = MoETransformer(input_dim, num_heads, num_experts)
        self.pooling = nn.AdaptiveAvgPool1d(1)

    def forward(self, x1, x2):
        outputs, aux_loss = self.transformer(x1, x2, x2)
        outputs = self.pooling(outputs.transpose(1, 2)).squeeze(-1)
        return outputs, aux_loss