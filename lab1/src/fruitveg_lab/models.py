from __future__ import annotations

import math

import torch
from torch import nn
from torchvision import models
from torchvision.models import ViT_B_16_Weights, ViT_B_32_Weights


def create_torchvision_resnet(
    model_name: str,
    *,
    num_classes: int,
    weights: str | None = None,
) -> nn.Module:
    """Create a torchvision ResNet classifier with a new output head."""

    factories = {
        "resnet18": models.resnet18,
        "resnet34": models.resnet34,
    }
    if model_name not in factories:
        raise ValueError(f"Unsupported ResNet model: {model_name}")

    model = factories[model_name](weights=weights)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def create_torchvision_vit(
    model_name: str,
    *,
    num_classes: int,
    image_size: int,
    weights: str | object | None = None,
) -> nn.Module:
    """Create a torchvision Vision Transformer with a new output head."""

    factories = {
        "vit_b_16": models.vit_b_16,
        "vit_b_32": models.vit_b_32,
    }
    weight_enums = {
        "vit_b_16": ViT_B_16_Weights.IMAGENET1K_V1,
        "vit_b_32": ViT_B_32_Weights.IMAGENET1K_V1,
    }
    if model_name not in factories:
        raise ValueError(f"Unsupported ViT model: {model_name}")

    if weights == "imagenet":
        weights = weight_enums[model_name]

    try:
        model = factories[model_name](weights=weights, image_size=image_size)
    except TypeError:
        if image_size != 224:
            raise ValueError(
                "This torchvision version does not accept a custom image_size for ViT. "
                "Use image_size=224 or upgrade torchvision."
            )
        model = factories[model_name](weights=weights)

    if hasattr(model.heads, "head"):
        model.heads.head = nn.Linear(model.heads.head.in_features, num_classes)
    else:
        last = len(model.heads) - 1
        model.heads[last] = nn.Linear(model.heads[last].in_features, num_classes)
    return model


def configure_vit_finetuning(
    model: nn.Module,
    *,
    mode: str,
    trainable_blocks: int = 2,
) -> nn.Module:
    """Configure which parts of a torchvision ViT are trainable."""

    if mode == "full":
        for parameter in model.parameters():
            parameter.requires_grad = True
        return model

    if mode not in {"head", "last_blocks"}:
        raise ValueError(f"Unknown ViT fine-tuning mode: {mode}")

    for parameter in model.parameters():
        parameter.requires_grad = False

    _set_trainable(model.heads, True)

    if mode == "last_blocks":
        if hasattr(model, "class_token"):
            model.class_token.requires_grad = True
        if hasattr(model.encoder, "pos_embedding"):
            model.encoder.pos_embedding.requires_grad = True
        if hasattr(model.encoder, "ln"):
            _set_trainable(model.encoder.ln, True)

        encoder_layers = list(model.encoder.layers.children())
        for block in encoder_layers[-trainable_blocks:]:
            _set_trainable(block, True)

    return model


def count_trainable_parameters(model: nn.Module) -> tuple[int, int]:
    """Return trainable and total parameter counts."""

    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    return trainable, total


def _set_trainable(module: nn.Module, value: bool) -> None:
    for parameter in module.parameters():
        parameter.requires_grad = value


class ResidualUnit(nn.Module):
    """Two-convolution residual block for the custom CNN."""

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1) -> None:
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
        )
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.shortcut = nn.Identity()
        self.activation = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.activation(self.body(x) + self.shortcut(x))


class ProduceResidualCNN(nn.Module):
    """A compact residual CNN implemented without torchvision model blocks."""

    def __init__(
        self,
        num_classes: int,
        channels: tuple[int, int, int, int] = (32, 64, 128, 256),
        dropout: float = 0.25,
    ) -> None:
        super().__init__()
        c1, c2, c3, c4 = channels
        self.stem = nn.Sequential(
            nn.Conv2d(3, c1, kernel_size=5, stride=2, padding=2, bias=False),
            nn.BatchNorm2d(c1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )
        self.stage1 = nn.Sequential(ResidualUnit(c1, c1), ResidualUnit(c1, c1))
        self.stage2 = nn.Sequential(ResidualUnit(c1, c2, stride=2), ResidualUnit(c2, c2))
        self.stage3 = nn.Sequential(ResidualUnit(c2, c3, stride=2), ResidualUnit(c3, c3))
        self.stage4 = nn.Sequential(ResidualUnit(c3, c4, stride=2), ResidualUnit(c4, c4))
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(c4, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        return self.head(x)


class PatchEmbedding(nn.Module):
    """Split an image into patches and project them into token embeddings."""

    def __init__(self, image_size: int, patch_size: int, embed_dim: int) -> None:
        super().__init__()
        if image_size % patch_size != 0:
            raise ValueError("image_size must be divisible by patch_size")
        self.num_patches = (image_size // patch_size) ** 2
        self.projection = nn.Conv2d(3, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.projection(x)
        return x.flatten(2).transpose(1, 2)


class MultiHeadSelfAttention(nn.Module):
    """Manual multi-head self-attention used by the custom ViT."""

    def __init__(self, embed_dim: int, num_heads: int, dropout: float) -> None:
        super().__init__()
        if embed_dim % num_heads != 0:
            raise ValueError("embed_dim must be divisible by num_heads")
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.qkv = nn.Linear(embed_dim, embed_dim * 3)
        self.attn_drop = nn.Dropout(dropout)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.proj_drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, tokens, dim = x.shape
        qkv = self.qkv(x).reshape(batch, tokens, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        query, key, value = qkv[0], qkv[1], qkv[2]
        attention = (query @ key.transpose(-2, -1)) * self.scale
        attention = attention.softmax(dim=-1)
        attention = self.attn_drop(attention)
        x = attention @ value
        x = x.transpose(1, 2).reshape(batch, tokens, dim)
        return self.proj_drop(self.proj(x))


class TransformerEncoderLayer(nn.Module):
    """Transformer encoder block with pre-normalization."""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        mlp_ratio: float,
        dropout: float,
    ) -> None:
        super().__init__()
        hidden_dim = int(embed_dim * mlp_ratio)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attention = MultiHeadSelfAttention(embed_dim, num_heads, dropout)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attention(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class TinyVisionTransformer(nn.Module):
    """A small Vision Transformer suitable for a 36-class produce dataset."""

    def __init__(
        self,
        *,
        num_classes: int,
        image_size: int = 128,
        patch_size: int = 16,
        embed_dim: int = 192,
        depth: int = 4,
        num_heads: int = 4,
        mlp_ratio: float = 3.0,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.patch_embed = PatchEmbedding(image_size, patch_size, embed_dim)
        self.class_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.position = nn.Parameter(torch.zeros(1, self.patch_embed.num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(dropout)
        self.blocks = nn.Sequential(
            *[
                TransformerEncoderLayer(embed_dim, num_heads, mlp_ratio, dropout)
                for _ in range(depth)
            ]
        )
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)
        self._init_parameters()

    def _init_parameters(self) -> None:
        nn.init.trunc_normal_(self.position, std=0.02)
        nn.init.trunc_normal_(self.class_token, std=0.02)
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.trunc_normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Conv2d):
                fan_out = module.kernel_size[0] * module.kernel_size[1] * module.out_channels
                nn.init.normal_(module.weight, mean=0.0, std=math.sqrt(2.0 / fan_out))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch = x.shape[0]
        x = self.patch_embed(x)
        class_token = self.class_token.expand(batch, -1, -1)
        x = torch.cat((class_token, x), dim=1)
        x = self.pos_drop(x + self.position)
        x = self.blocks(x)
        x = self.norm(x)
        return self.head(x[:, 0])
