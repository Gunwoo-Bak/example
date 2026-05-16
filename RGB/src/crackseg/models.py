from __future__ import annotations

import torch


def _patch_torch_compiler() -> None:
    """Compatibility shim for older PyTorch + segmentation_models_pytorch."""
    if not hasattr(torch, "compiler"):
        class _DummyCompiler:
            @staticmethod
            def is_compiling() -> bool:
                return False

        torch.compiler = _DummyCompiler()  # type: ignore[attr-defined]
    elif not hasattr(torch.compiler, "is_compiling"):
        torch.compiler.is_compiling = lambda: False  # type: ignore[attr-defined]


_patch_torch_compiler()

import segmentation_models_pytorch as smp  # noqa: E402


def create_model(
    model: str,
    encoder_name: str,
    num_classes: int,
    encoder_weights: str | None = None,
) -> torch.nn.Module:
    if model == "deeplabv3plus":
        return smp.DeepLabV3Plus(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=3,
            classes=num_classes,
        )
    if model == "segformer_b2":
        return smp.Segformer(
            encoder_name="mit_b2",
            encoder_weights=encoder_weights,
            in_channels=3,
            classes=num_classes,
        )
    raise ValueError(f"Unknown model: {model}")

