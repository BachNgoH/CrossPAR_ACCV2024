import torch.nn as nn
import timm
import torch
from models.swin_transformer import swin_small_patch4_window7_224, swin_base_patch4_window7_224
from models.x2vlm import load_pretrained_vision_tower
from models.fusion_head import MoEFusionHead, TransformerFusion



class PARModel(nn.Module):
    def __init__(self, config):

        super(PARModel, self).__init__()
        
        if config["use_kan"]:
            from models.kan import KANLinear as Linear
        else:
            from torch.nn import Linear

        self.backbone_name = config["backbone"]
        self.fusion_method = config["fuse_method"]
        self.config = config

        if config["backbone"] == "SOLIDER":
            self.backbone = swin_base_patch4_window7_224(img_size=config['image_res'], drop_path_rate=0.1, ckpt=config["ckpt_1"])
            feature_dim_1 = self.backbone.num_features[-1]

            self.classifier = Linear(feature_dim_1, config["num_attr"])

        elif config["backbone"] == "x2vlm":
            self.backbone = load_pretrained_vision_tower(config["ckpt"], config)
            self.classifier = Linear(config['embed_dim'], config["num_attr"])

        elif config["backbone"] == "fusion":
            if config["backbone_1"] == "SOLIDER":
                self.backbone_1 = swin_base_patch4_window7_224(img_size=config["image_res"], drop_path_rate=0.1, ckpt=config["ckpt_1"])
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
                self.adapter_1 = Linear(feature_dim_1, 512)
                self.adapter_2 = Linear(config['embed_dim'], 512)
                self.classifier = Linear(config['embed_dim'] * 2, config['num_attr'])

            elif self.fusion_method == "attn":
                self.fusion_layer = TransformerFusion(config["embed_dim"] + self.backbone_1.num_features[-1], config['embed_dim'])
                self.classifier = Linear(config['embed_dim'], config['num_attr'])
            elif self.fusion_method == "moe":
                self.adapter_1 = Linear(feature_dim_1, config["embed_dim"])
                self.fusion_layer = MoEFusionHead(config["embed_dim"], config["embed_dim"] // 64, config["num_experts"], use_kan=config["use_kan"])
                self.classifier = Linear(config['embed_dim'], config['num_attr'])
            else:
                self.adapter_1 = Linear(feature_dim_1, config['embed_dim'])
                self.adapter_2 = Linear(feature_dim_2, config['embed_dim'])
                self.classifier = Linear(config['embed_dim'], config['num_attr'])

        else:
            self.backbone = timm.create_model(config["backbone"], pretrained=True, num_classes=512)        
            self.classifier = Linear(config['embed_dim'], config["num_attr"])



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
            if self.config["backbone"] == "SOLIDER":
                x, _ = self.backbone(x)
                x = self.classifier(x)
            else:
                x = self.backbone(x)
                x = self.classifier(x)
            #outputs = self.activation(x)
            return x
