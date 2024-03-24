import torch.nn as nn
import timm
from models.swin_transformer import swin_small_patch4_window7_224
from models.x2vlm import load_pretrained_vision_tower

class PARModel(nn.Module):
    def __init__(self, config):

        super(PARModel, self).__init__()
        if config["backbone"] == "SOLIDER":
            self.backbone = swin_small_patch4_window7_224(img_size=(224, 224), drop_path_rate=0.1)
        if config["backbone"] == "x2vlm":
            self.backbone = load_pretrained_vision_tower(config["ckpt"], config)

        if config["backbone"] == "fusion":
            res = config["image_res"]
            self.backbone_1 = swin_small_patch4_window7_224(img_size=(res, res), drop_path_rate=0.1)
            self.backbone_2 = load_pretrained_vision_tower(config["ckpt"], config)

            self.adapter_1 = nn.Linear(config['embed_dim'], 512)
            self.adapter_2 = nn.Linear(config['embed_dim'], 512)
            self.classifier = nn.Linear(512, config['num_attr'])

        else:
            self.backbone = timm.create_model(config["backbone"], pretrained=True, num_classes=512)        
            self.classifier = nn.Linear(config['embed_dim'], config["num_attr"])

        self.backbone_name = config["backbone"]

    def forward(self, x):
        if self.backbone_name == "x2vlm":
            x = self.backbone(x)[:, 0, :]
            x = self.classifier(x)
            return x
        else:
            x = self.backbone(x)
            x = self.classifier(x)
            #outputs = self.activation(x)
            return x
