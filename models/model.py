import torch.nn as nn
import timm
import torch
from models.swin_transformer import swin_small_patch4_window7_224
from models.x2vlm import load_pretrained_vision_tower


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

class PARModel(nn.Module):
    def __init__(self, config):

        super(PARModel, self).__init__()

        self.backbone_name = config["backbone"]
        self.fusion_method = config["fuse_method"]

        if config["backbone"] == "SOLIDER":
            self.backbone = swin_small_patch4_window7_224(img_size=(224, 224), drop_path_rate=0.1)
            self.classifier = nn.Linear(config['embed_dim'], config["num_attr"])

        elif config["backbone"] == "x2vlm":
            self.backbone = load_pretrained_vision_tower(config["ckpt"], config)
            self.classifier = nn.Linear(config['embed_dim'], config["num_attr"])

        elif config["backbone"] == "fusion":
            res = config["image_res"]
            self.backbone_1 = swin_small_patch4_window7_224(img_size=(res, res), drop_path_rate=0.1)
            self.backbone_2 = load_pretrained_vision_tower(config["ckpt"], config)


            if self.fusion_method == "concat":
                self.adapter_1 = nn.Linear(self.backbone_1.num_features[-1], 512)
                self.adapter_2 = nn.Linear(config['embed_dim'], 512)
                self.classifier = nn.Linear(config['embed_dim'] * 2, config['num_attr'])

            elif self.fusion_method == "attn":
                self.fusion_layer = TransformerFusion(config["embed_dim"] * 2, config['embed_dim'])
                self.classifier = nn.Linear(config['embed_dim'], config['num_attr'])

            else:
                self.adapter_1 = nn.Linear(self.backbone_1.num_features[-1], 512)
                self.adapter_2 = nn.Linear(config['embed_dim'], 512)
                self.classifier = nn.Linear(config['embed_dim'], config['num_attr'])

        else:
            self.backbone = timm.create_model(config["backbone"], pretrained=True, num_classes=512)        
            self.classifier = nn.Linear(config['embed_dim'], config["num_attr"])



    def forward(self, x):
        if self.backbone_name == "x2vlm":
            x = self.backbone(x)[:, 0, :]
            x = self.classifier(x)
            return x
        elif self.backbone_name == "fusion":
            out1, _ = self.backbone_1(x)
            out2 = self.backbone_2(x)[:, 0, :]

            if self.fusion_method == "concat":
                out = torch.cat((out1, out2), dim=-1)
            elif self.fusion_method == "attn":
                out = self.fusion_layer(out1, out2)
            else:
                out = out1 + out2
            return self.classifier(out)
        else:
            x = self.backbone(x)
            x = self.classifier(x)
            #outputs = self.activation(x)
            return x
