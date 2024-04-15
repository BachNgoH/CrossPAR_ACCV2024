import torch
import torch.nn as nn
import torch.nn.functional as F

class Expert(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(Expert, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
    
    def forward(self, x):
        return self.network(x)

class GatingNetwork(nn.Module):
    def __init__(self, input_dim, num_experts):
        super(GatingNetwork, self).__init__()
        self.gate = nn.Linear(input_dim, num_experts)
    
    def forward(self, x):
        return F.softmax(self.gate(x), dim=1)

class MoELayer(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_experts):
        super(MoELayer, self).__init__()
        self.experts = nn.ModuleList([Expert(input_dim, hidden_dim, output_dim) for _ in range(num_experts)])
        self.gate = GatingNetwork(input_dim, num_experts)
    
    def forward(self, x):
        gating_weights = self.gate(x)
        expert_outputs = torch.stack([expert(x) for expert in self.experts], dim=1)
        output = torch.einsum('bne,bne->be', gating_weights, expert_outputs)
        return output

class MoETransformer(nn.Module):
    def __init__(self, input_dim, num_heads, hidden_dim, num_experts):
        super(MoETransformer, self).__init__()
        self.multihead_attn = nn.MultiheadAttention(input_dim, num_heads)
        self.moelayer = MoELayer(input_dim, hidden_dim, input_dim, num_experts)
        self.norm1 = nn.LayerNorm(input_dim)
        self.norm2 = nn.LayerNorm(input_dim)
    
    def forward(self, q, k, v):
        attn_output, _ = self.multihead_attn(q, k, v)
        x = self.norm1(x + attn_output)
        moe_output = self.moelayer(x)
        x = self.norm2(x + moe_output)
        return x
    
class MoEFusionHead(nn.Module):
    def __init__(self, input_dim, num_heads, hidden_dim, num_experts):
        super(MoEFusionHead, self).__init__()
        self.transformer = MoETransformer(input_dim, num_heads, hidden_dim, num_experts)
        self.pooling = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(input_dim, 1)

    def forward(self, x1, x2):
        print(x2.shape)
        print(x1.shape)
        outputs = self.pooling(self.transformer(x1, x2, x2))
        return self.fc(outputs)