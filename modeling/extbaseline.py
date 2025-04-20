# encoding: utf-8
"""
@author:  liaoxingyu
@contact: sherlockliao01@gmail.com
"""

import torch
from torch import nn

from .backbones.resnet import ResNet, BasicBlock, Bottleneck
from .backbones.senet import SENet, SEResNetBottleneck, SEBottleneck, SEResNeXtBottleneck
from .backbones.resnet_ibn_a import resnet50_ibn_a
from .backbones.dinov2 import DinoVisionTransformer, MODEL_DIMS
from .backbones.gat import GraphTransformer


def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_out')
        nn.init.constant_(m.bias, 0.0)
    elif classname.find('Conv') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif classname.find('BatchNorm') != -1:
        if m.affine:
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)


def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.normal_(m.weight, std=0.001)
        if m.bias:
            nn.init.constant_(m.bias, 0.0)


class ExtBaseline(nn.Module):
    in_planes = 2048

    def __init__(self, num_classes, last_stride, model_path, neck, neck_feat, model_name, pretrain_choice, in_channels, out_features):
        super(ExtBaseline, self).__init__()
        if model_name == 'gat_resnet50':
            self.base = ResNet(last_stride=last_stride,
                               block=Bottleneck,
                               layers=[3, 4, 6, 3])
        elif model_name.startswith('gat_dinov2'):
            dinov2_variant = model_name.replace('gat_', '')  # Remove 'gat_' prefix to get the dinov2_variant
            # Get the corresponding dimension from the MODEL_DIMS dictionary in dinov2.py
            self.in_planes = MODEL_DIMS.get(dinov2_variant, 384)  # Default to 384 if not found
            self.base = DinoVisionTransformer(
                last_stride=last_stride,
                model_variant=dinov2_variant,
                pretrained=True
            )
            # self.base.freeze_backbone()  # Freeze the backbone

        if pretrain_choice == 'imagenet':
            self.base.load_param(model_path)
            print('Loading pretrained ImageNet model......')

        # graph transformer
        self.gat = GraphTransformer(
            in_channels=in_channels,
            out_channels=out_features,
            heads=1,
            dropout=0.0,
            edge_dim=384
        )

        self.fc = nn.Linear(in_features=out_features+self.in_planes, out_features=128)

        self.gap = nn.AdaptiveAvgPool2d(1)
        # self.gap = nn.AdaptiveMaxPool2d(1)
        self.num_classes = num_classes
        self.neck = neck
        self.neck_feat = neck_feat

        if self.neck == 'no':
            self.classifier = nn.Linear(128, self.num_classes)
            # self.classifier = nn.Linear(self.in_planes, self.num_classes, bias=False)     # new add by luo
            # self.classifier.apply(weights_init_classifier)  # new add by luo
        elif self.neck == 'bnneck':
            self.bottleneck = nn.BatchNorm1d(128)
            self.bottleneck.bias.requires_grad_(False)  # no shift
            self.classifier = nn.Linear(128, self.num_classes, bias=False)

            self.bottleneck.apply(weights_init_kaiming)
            self.classifier.apply(weights_init_classifier)

    def forward(self, img, sample):

        # img branch
        # torch.Size([64, 3, 252, 126])
        img_feat = self.gap(self.base(img))  # (b, 2048, 1, 1)
        # torch.Size([64, 384, 1, 1])
        img_feat = img_feat.view(img_feat.shape[0], -1)  # flatten to (bs, 2048)
        # torch.Size([64, 384])

        # graph branch

        graph_feat, attention_weights1, attention_weights2 = self.gat(sample)  # (b, 384, 1, 1)


        global_feat = self.fc(torch.cat((img_feat, graph_feat), dim=1))  # (b, 128)

        if self.neck == 'no':
            feat = global_feat
        elif self.neck == 'bnneck':
            feat = self.bottleneck(global_feat)  # normalize for angular softmax

        if self.training:
            cls_score = self.classifier(feat)
            return cls_score, global_feat  # global feature for triplet loss
        else:
            if self.neck_feat == 'after':
                # print("Test with feature after BN")
                return feat
            else:
                # print("Test with feature before BN")
                return global_feat

    def load_param(self, trained_path):
        param_dict = torch.load(trained_path, map_location=torch.device('cpu'))
        
        # Add debug prints
        print(f"Loaded checkpoint keys: {sorted(list(param_dict.keys())) if isinstance(param_dict, dict) else 'not a dict'}")
        
        if isinstance(param_dict, dict):
            # If checkpoint has a 'model' key with model state
            if 'model' in param_dict and isinstance(param_dict['model'], dict):
                param_dict = param_dict['model']
                print(f"Using 'model' key from checkpoint with {len(param_dict)} parameters")
            
            # Create counters to track loading progress
            loaded_count = 0
            total_count = 0
            
            state_dict = self.state_dict()
            for i in state_dict:
                total_count += 1
                
                # Try direct key match
                if i in param_dict and i not in ['classifier.weight', 'classifier.bias']:
                    state_dict[i].copy_(param_dict[i])
                    loaded_count += 1
                # Try removing module prefix (handles DataParallel checkpoints)
                elif i.startswith('module.') and i[7:] in param_dict:
                    state_dict[i].copy_(param_dict[i[7:]])
                    loaded_count += 1
                # Try adding module prefix
                elif 'module.' + i in param_dict:
                    state_dict[i].copy_(param_dict['module.' + i])
                    loaded_count += 1
                # Check for model prefix
                elif 'model.' + i in param_dict:
                    state_dict[i].copy_(param_dict['model.' + i])
                    loaded_count += 1
                # Try removing base prefix
                elif i.startswith('base.') and i[5:] in param_dict:
                    state_dict[i].copy_(param_dict[i[5:]])
                    loaded_count += 1
                # Try replacing base with backbone (common in some frameworks)
                # elif i.replace('base.', 'backbone.') in param_dict:
                #     state_dict[i].copy_(param_dict[i.replace('base.', 'backbone.')])
                #     loaded_count += 1
                else:
                    print(f'Parameter not found in checkpoint: {i}')
            
            print(f"Loaded {loaded_count}/{total_count} parameters from checkpoint")
            if loaded_count < total_count * 0.8:  # Less than 80% loaded
                print("WARNING: Many parameters couldn't be loaded from checkpoint!")
                print("This may indicate a mismatch between model architecture and checkpoint")
                # List first few state dict keys for debugging
                print(f"Model expects parameters: {list(state_dict.keys())[:10]}")
        else:
            print("Checkpoint is not a dictionary, cannot load parameters")
