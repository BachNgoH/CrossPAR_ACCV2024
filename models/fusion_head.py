import torch.nn as nn
import torch

class TransformerFusion(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers=2, num_heads=2, dropout=0.2):
        super(TransformerFusion, self).__init__()
        self.embedding_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
                
        # Transformer encoder
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=input_dim, nhead=num_heads)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)
        
        # Linear layers for fusion
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x1, x2):
        # Concatenate the input embeddings
        x = torch.cat((x1, x2), dim=-1)
                
        # Transformer encoding
        x = self.transformer_encoder(x)
        
        # Fusion through fully connected layers
        x = self.dropout(self.relu(self.fc1(x)))        
        return x

class MoETransformer(nn.Module):
    def __init__(self, input_dim, num_heads, num_experts, use_kan=False):
        super(MoETransformer, self).__init__()
        self.multihead_attn = nn.MultiheadAttention(input_dim, num_heads, batch_first=True)
        if use_kan:
            from models.moe import MoE
        else:
            from st_moe_pytorch import MoE
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
    def __init__(self, input_dim, num_heads, num_experts, use_kan=False):
        super(MoEFusionHead, self).__init__()
        self.transformer = MoETransformer(input_dim, num_heads, num_experts, use_kan=use_kan)
        self.pooling = nn.AdaptiveAvgPool1d(1)

    def forward(self, x1, x2):
        outputs, aux_loss = self.transformer(x1, x2, x2)
        outputs = self.pooling(outputs.transpose(1, 2)).squeeze(-1)
        return outputs, aux_loss