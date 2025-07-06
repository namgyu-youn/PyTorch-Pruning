import timm

MODEL_CONFIGS = {
    'vit': 'vit_base_patch16_224',
    'efficientnet': 'efficientnet_b4',
    'convnext': 'convnext_small',
    'swin': 'swin_base_patch4_window7_224'
}

def load_model(model_name, device='cuda'):
    """Load pretrained model"""
    model = timm.create_model(MODEL_CONFIGS[model_name], pretrained=True, num_classes=10)
    return model.to(device).eval()