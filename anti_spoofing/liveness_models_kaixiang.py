import torch


SUPPORTED_MODELS = ("mobilenetv2", "efficientnetb0")


def build_liveness_model(model_name: str, pretrained_backbone: bool = False):
    """Rebuild a Kaixiang liveness model for checkpoint loading.

    Model-specific heads live in the two training entry files so each model's
    architecture and tuning settings are easy to inspect in one place.
    """

    model_name = model_name.lower()

    if model_name == "mobilenetv2":
        from anti_spoofing.train_liveness_mobilenetv2_kaixiang import (
            build_mobilenetv2_liveness_model,
        )

        return build_mobilenetv2_liveness_model(pretrained_backbone)

    if model_name == "efficientnetb0":
        from anti_spoofing.train_liveness_efficientnetb0_kaixiang import (
            build_efficientnetb0_liveness_model,
        )

        return build_efficientnetb0_liveness_model(pretrained_backbone)

    supported = ", ".join(SUPPORTED_MODELS)
    raise ValueError(f"Unsupported model '{model_name}'. Choose one of: {supported}")


def load_checkpoint(checkpoint_path, device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model_name = checkpoint["model_name"]
    model = build_liveness_model(model_name, pretrained_backbone=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    return model, checkpoint, device
