"""Model factories.

Segmentation: U-Net (Ronneberger et al., 2015) from segmentation_models_pytorch
with an ImageNet-pretrained ResNet-34 encoder. The first convolution is adapted
to any number of input bands (smp copies/averages the RGB filters), which is
the standard way to reuse ImageNet features on multispectral imagery.

Classification: ResNet-18 pretrained with MoCo self-supervision on SSL4EO-S12
(1M Sentinel-2 L1C scenes, all 13 bands) via TorchGeo. Using satellite-native
pretraining instead of ImageNet gives better accuracy with less labelled data.
"""

from __future__ import annotations

import os

import torch

# Set GSTP_OFFLINE=1 to skip downloading pretrained weights (used by the offline smoke tests).
OFFLINE = os.environ.get("GSTP_OFFLINE") == "1"


def build_unet(in_channels: int, encoder: str = "resnet34", pretrained: bool = True):
    import segmentation_models_pytorch as smp

    pretrained = pretrained and not OFFLINE
    return smp.Unet(
        encoder_name=encoder,
        encoder_weights="imagenet" if pretrained else None,
        in_channels=in_channels,
        classes=1,  # one logit: probability of the positive class (water / burn / forest)
    )


# Channel order of the SSL4EO-S12 Sentinel-2 weights. Inputs MUST follow this order.
SSL4EO_S2_BANDS = ["B01", "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B09", "B10", "B11", "B12"]


class _Upsample(torch.nn.Module):
    """Bilinear resize before the backbone: SSL4EO weights were pretrained on
    224-px crops, so 64-px EuroSAT patches are enlarged to keep feature scales
    closer to pretraining. Part of the model, so inference and ONNX include it."""

    def __init__(self, backbone, size):
        super().__init__()
        self.backbone, self.size = backbone, size

    def forward(self, x):
        import torch.nn.functional as F

        return self.backbone(F.interpolate(x, size=(self.size, self.size), mode="bilinear", align_corners=False))


def build_s2_classifier(num_classes: int, drop_bands: list[str] | None = None, pretrained: bool = True,
                        input_size: int | None = 128):
    """ResNet-18 with SSL4EO-S12 Sentinel-2 (13-band, MoCo) weights.

    drop_bands removes input channels from the pretrained first convolution,
    e.g. ["B10"] so the model accepts Sentinel-2 L2A (12 bands). Remaining
    inputs must follow SSL4EO_S2_BANDS order.
    """
    import timm
    import torch

    drop_band_idx = [SSL4EO_S2_BANDS.index(b) for b in (drop_bands or [])]
    in_chans = 13 - len(drop_band_idx or [])
    model = timm.create_model("resnet18", in_chans=13, num_classes=num_classes)
    if pretrained and not OFFLINE:
        from torchgeo.models import ResNet18_Weights

        w = ResNet18_Weights.SENTINEL2_ALL_MOCO
        state = w.get_state_dict(progress=True)
        state = {k: v for k, v in state.items() if not k.startswith("fc.")}
        missing, unexpected = model.load_state_dict(state, strict=False)
        assert all(k.startswith("fc.") for k in missing), f"Unexpected missing keys: {missing}"
    if drop_band_idx:
        keep = [i for i in range(13) if i not in set(drop_band_idx)]
        old = model.conv1
        new = torch.nn.Conv2d(in_chans, old.out_channels, old.kernel_size, old.stride, old.padding, bias=False)
        with torch.no_grad():
            new.weight.copy_(old.weight[:, keep])
        model.conv1 = new
    return _Upsample(model, input_size) if input_size else model


def classifier_bands(drop_bands=("B10",)) -> list[str]:
    """Band list, in the order the classifier expects."""
    return [b for b in SSL4EO_S2_BANDS if b not in set(drop_bands)]


def count_parameters(model) -> int:
    return sum(p.numel() for p in model.parameters())


def export_onnx(model, in_channels: int, size: int, path: str, opset: int = 17):
    """Export to ONNX — the hand-off format for a later TensorRT build on Jetson."""
    import torch

    model = model.eval().cpu()
    dummy = torch.zeros(1, in_channels, size, size)
    kwargs = dict(input_names=["image"], output_names=["logits"], opset_version=opset,
                  dynamic_axes={"image": {0: "batch"}, "logits": {0: "batch"}})
    try:
        # TorchScript-based exporter: mature, TensorRT-friendly graphs.
        torch.onnx.export(model, dummy, path, dynamo=False, **kwargs)
    except TypeError:  # older torch without the `dynamo` argument
        torch.onnx.export(model, dummy, path, **kwargs)
    _check_onnx(model, dummy, path)
    return path


def _check_onnx(model, dummy, path, tol=1e-3):
    """Numerically compare the ONNX graph with PyTorch (if onnxruntime is installed)."""
    try:
        import onnxruntime as ort
    except ImportError:
        return
    import numpy as np
    import torch

    x = torch.randn_like(dummy)
    with torch.no_grad():
        ref = model(x).numpy()
    out = ort.InferenceSession(path, providers=["CPUExecutionProvider"]).run(None, {"image": x.numpy()})[0]
    err = float(np.max(np.abs(out - ref)) / max(1.0, float(np.max(np.abs(ref)))))
    if err > tol:
        raise RuntimeError(f"ONNX output differs from PyTorch: max relative error {err:.2e}")
    print(f"ONNX export verified ({path}): max relative error {err:.1e}")
