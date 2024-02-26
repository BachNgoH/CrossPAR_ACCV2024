import torch.nn as nn
import timm
from graph import GCN

class PARModel(nn.Module):
    def __init__(self, num_attributes=40, backbone_name='resnet18', pretrained=True, num_per_group=None):
        super(PARModel, self).__init__()
        if backbone_name == "SOLIDER":
            self.backbone
        else:
            self.backbone = timm.create_model(backbone_name, pretrained=pretrained, num_classes=512)        
        #self.activation = nn.Sigmoid()
        if num_per_group != None:
            self.classifiers = nn.ModuleList()
            for num in num_per_group:
                # self.classifiers.append(GCN(in_features=512,
                #             edge_features=512,
                #             out_feature=num, 
                #             device='cuda',
                #             ratio=(1,)))
                self.classifiers.append(nn.Linear(512, num))
            self.multi_task=True

        else:
            self.classifier = nn.Linear(512, num_attributes)
            self.multi_task=False

    def forward(self, x):
        # Forward pass through the backbone
        if self.multi_task:
            x = self.backbone(x)
            outputs = []
            # all_edge_sim = []
            for classifier in self.classifiers:
                output = classifier(x)
                outputs.append(output)
                # all_edge_sim.append(edge_sim)
            return outputs
        else:
            x = self.backbone(x)
            x = self.classifier(x)
            #outputs = self.activation(x)
            return x
