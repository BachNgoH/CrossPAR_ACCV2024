import torch.nn as nn
import timm

class PARModel(nn.Module):
    def __init__(self, num_attributes=40, backbone_name='resnet18', pretrained=True):
        super(PARModel, self).__init__()
        
        self.backbone = timm.create_model(backbone_name, pretrained=pretrained, num_classes=num_attributes)        
        #self.activation = nn.Sigmoid()

    def forward(self, x):
        # Forward pass through the backbone
        x = self.backbone(x)        
        #outputs = self.activation(x)
        return x
