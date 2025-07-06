import timm

MODEL_CONFIGS = {
    'c': 'convnext_large',
    'e': 'efficientnet_b4',
    's': 'swin_large_patch4_window7_224',
    'v': 'vit_large_patch16_224',
    'm': 'mobilenetv4_conv_large',
    'r': 'resnet152d',
    'x': 'resnext101_32x8d',
}

def load_model(model_name, device='cuda'):
    """Load pretrained model"""
    model = timm.create_model(MODEL_CONFIGS[model_name], pretrained=True, num_classes=10)

    return model.to(device).eval()