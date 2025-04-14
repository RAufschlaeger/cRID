# encoding: utf-8
"""
@author:  raufschlaeger
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.hub import load_state_dict_from_url

# DINOv2 model variants with their corresponding pretrained URLs
DINOV2_MODELS = {
    'dinov2_vits14': 'https://dl.fbaipublicfiles.com/dinov2/dinov2_vits14/dinov2_vits14_pretrain.pth',
    'dinov2_vitb14': 'https://dl.fbaipublicfiles.com/dinov2/dinov2_vitb14/dinov2_vitb14_pretrain.pth',
    'dinov2_vitl14': 'https://dl.fbaipublicfiles.com/dinov2/dinov2_vitl14/dinov2_vitl14_pretrain.pth',
    'dinov2_vitg14': 'https://dl.fbaipublicfiles.com/dinov2/dinov2_vitg14/dinov2_vitg14_pretrain.pth',
}

# Feature dimensions for different model sizes
MODEL_DIMS = {
    'dinov2_vits14': 384,  # Fixed - small model has 384 dimensions
    'dinov2_vitb14': 768,  # base model
    'dinov2_vitl14': 1024, # large model
    'dinov2_vitg14': 1536, # giant model
}


class DinoVisionTransformer(nn.Module):
    """
    Vision Transformer backbone using DINOv2 pretrained weights.
    Compatible with ReID strong baseline framework.
    """
    def __init__(self, last_stride=1, model_variant='dinov2_vitb14', pretrained=True):
        super(DinoVisionTransformer, self).__init__()
        
        self.model_variant = model_variant
        
        # Import the necessary libraries for DINOv2
        try:
            from torch.hub import load
            # Import the model from the official repo
            self.backbone = load('facebookresearch/dinov2', model_variant)
        except:
            print(f"Failed to load {model_variant} directly from torch.hub.")
            print("Attempting to use timm library...")
            try:
                import timm
                self.backbone = timm.create_model(model_variant, pretrained=pretrained)
            except:
                raise ImportError(
                    "Could not load DINOv2 model. Please install timm library "
                    "or ensure you have access to torch.hub with Meta's repositories."
                )
        
        # Set the last_stride parameter (for compatibility with the ReID framework)
        self.last_stride = last_stride


    def forward(self, x):
        """
        Forward pass through the DINOv2 backbone.
        Returns feature maps compatible with ReID framework.
        """
        # Get the batch size and image dimensions
        B, C, H, W = x.shape
        
        # Get features from DINOv2
        features = self.backbone.forward_features(x)
        
        # Handle dictionary output format from DINOv2
        if isinstance(features, dict):
            # The backbone returns a dictionary with different token types
            # Use the patch tokens (most likely to contain spatial information)
            if 'x_norm_patchtokens' in features:
                patch_tokens = features['x_norm_patchtokens']
            elif 'x_norm_patchnokens' in features:  # Handle potential typo in key name
                patch_tokens = features['x_norm_patchnokens']
            else:
                # Fallback to another key if patch tokens aren't available
                for key in ['x_prenorm', 'x_norm_regtokens']:
                    if key in features:
                        patch_tokens = features[key]
                        break
                else:
                    raise ValueError(f"Couldn't find appropriate tokens in output. Available keys: {features.keys()}")
        elif isinstance(features, tuple):
            patch_tokens = features[0]  # Take the patch tokens from tuple
        else:
            # Assume it's already the patch tokens tensor
            patch_tokens = features
            # If there's a CLS token, remove it (first token)
            if patch_tokens.ndim == 3 and patch_tokens.size(1) > (H//14)*(W//14):
                patch_tokens = patch_tokens[:, 1:, :]
        
        # Reshape to spatial feature map (assuming 14×14 patches)
        patch_size = 14
        h_patches = H // patch_size
        w_patches = W // patch_size
        
        # Reshape from [B, patches, dim] to [B, dim, h, w]: torch.Size([64, 384, 18, 9])
        features = patch_tokens.reshape(B, h_patches, w_patches, -1).permute(0, 3, 1, 2)
        
        return features

    def load_param(self, model_path):
        """
        Load parameters from a checkpoint file.
        Compatible with ReID framework load method.
        """
        if model_path.startswith(('http://', 'https://')):
            state_dict = load_state_dict_from_url(model_path, progress=True, map_location='cpu')
        else:
            state_dict = torch.load(model_path, map_location='cpu')
        
        # Handle different checkpoint formats
        if 'model' in state_dict:
            state_dict = state_dict['model']
        elif 'state_dict' in state_dict:
            state_dict = state_dict['state_dict']
        
        # Don't try to load directly into self.backbone, which causes key mismatches
        # Instead, properly load the state dict into the backbone's state dict
        backbone_state_dict = self.backbone.state_dict()
        
        # Filter the state dict to only include keys that are in the backbone
        filtered_state_dict = {k: v for k, v in state_dict.items() 
                              if k in backbone_state_dict}
        
        # Load the filtered state dict
        self.backbone.load_state_dict(filtered_state_dict, strict=False)
        print(f"Loaded DINOv2 parameters from {model_path}")
        print(f"Loaded {len(filtered_state_dict)} / {len(backbone_state_dict)} parameters")

    def freeze_backbone(self):
        """
        Freeze the DINOv2 backbone to make it non-trainable.
        """
        for param in self.backbone.parameters():
            param.requires_grad = False
        print("DINOv2 backbone has been frozen.")
