import torch.nn as nn
import timm
import torch
from models.swin_transformer import swin_small_patch4_window7_224, swin_base_patch4_window7_224
from models.x2vlm import load_pretrained_vision_tower
from models.moe import MoEFusionHead

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
        self.config = config

        if config["backbone"] == "SOLIDER":
            self.backbone = swin_small_patch4_window7_224(img_size=config['image_res'], drop_path_rate=0.1)
            self.classifier = nn.Linear(config['embed_dim'], config["num_attr"])

        elif config["backbone"] == "x2vlm":
            self.backbone = load_pretrained_vision_tower(config["ckpt"], config)
            self.classifier = nn.Linear(config['embed_dim'], config["num_attr"])

        elif config["backbone"] == "fusion":
            if config["backbone_1"] == "SOLIDER":
                self.backbone_1 = swin_base_patch4_window7_224(img_size=config["image_res"], drop_path_rate=0.1)
                feature_dim_1 = self.backbone_1.num_features[-1]
            else:
                self.backbone_1 = timm.create_model(config["backbone_1"], img_size=config["image_res"], pretrained=True, num_classes=config["embed_dim"])
                feature_dim_1 = self.backbone_1.num_features
            if config["backbone_2"] == "x2vlm":
                self.backbone_2 = load_pretrained_vision_tower(config["ckpt"], config)
                feature_dim_2 = config['embed_dim']
            else:
                self.backbone_2 = timm.create_model(config["backbone_2"], img_size=config["image_res"], num_classes=config["embed_dim"])
                feature_dim_2 = config['embed_dim']

            if self.fusion_method == "concat":
                self.adapter_1 = nn.Linear(feature_dim_1, 512)
                self.adapter_2 = nn.Linear(config['embed_dim'], 512)
                self.classifier = nn.Linear(config['embed_dim'] * 2, config['num_attr'])

            elif self.fusion_method == "attn":
                self.fusion_layer = TransformerFusion(config["embed_dim"] + self.backbone_1.num_features[-1], config['embed_dim'])
                self.classifier = nn.Linear(config['embed_dim'], config['num_attr'])
            elif self.fusion_method == "moe":
                self.adapter_1 = nn.Linear(feature_dim_1, config["embed_dim"])
                self.fusion_layer = MoEFusionHead(config["embed_dim"], config["embed_dim"] // 64, config["num_experts"])
                self.classifier = nn.Linear(config['embed_dim'], config['num_attr'])
            else:
                self.adapter_1 = nn.Linear(feature_dim_1, config['embed_dim'])
                self.adapter_2 = nn.Linear(feature_dim_2, config['embed_dim'])
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
            if self.config["backbone_1"] == "SOLIDER":
                out1, swin_features = self.backbone_1(x)
                swin_shapes = swin_features[-1].shape
                swin_features = swin_features[-1].reshape(swin_shapes[0], -1, swin_shapes[1])
                swin_features = self.adapter_1(swin_features)
            else:
                swin_features = self.backbone_1.forward_features(x)
                swin_shapes = swin_features.shape
                swin_features = swin_features.reshape(swin_shapes[0], -1, swin_shapes[3])
                swin_features = self.adapter_1(swin_features)
            
            if self.config["backbone_2"] == "x2vlm":
                vlm_features = self.backbone_2(x)
                out2 = vlm_features[:, 0, :]
            else:
                vlm_features = self.backbone_2.forward_features(x)
                out2 = vlm_features[:, 0, :]


            if self.fusion_method == "concat":
                out = torch.cat((out1, out2), dim=-1)
            elif self.fusion_method == "attn":
                out = self.fusion_layer(out1, out2)
            elif self.fusion_method == "moe":
                out, aux_loss = self.fusion_layer(vlm_features, swin_features)
                return self.classifier(out), aux_loss
            else:
                out = out1 + out2
            return self.classifier(out)
        
        else:
            x = self.backbone(x)
            x = self.classifier(x)
            #outputs = self.activation(x)
            return x
