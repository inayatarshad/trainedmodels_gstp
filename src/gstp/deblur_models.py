"""Learned deblurring models.

  ConvAutoencoder : plain encoder-decoder WITHOUT skip connections. Simple AI
                    baseline: everything must pass through a small bottleneck,
                    so fine detail is lost - shows why skip connections matter.
  NAFNet          : encoder-decoder WITH skip connections that predicts only the
                    correction to the blurry input (residual). Chen et al.,
                    "Simple Baselines for Image Restoration", ECCV 2022.
                    Built small here so it fits the Jetson.

Both work on ONE band at a time (1 channel in, 1 out), so the same network can
run on any band of the camera. Inputs are reflectance in [0, 1]; the wrapper
scales them internally, so an exported ONNX file takes reflectance directly.
Large images are processed in overlapping tiles (`tiled`), which keeps memory
small on the Jetson and matches the patch size used in training.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

INPUT_SCALE = 4.0  # reflectance x 4 -> typical land values land around 0.2-1.2


# ───────────────────────────────────────────────────────────── NAFNet parts
class LayerNorm2d(nn.Module):
    def __init__(self, c, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(1, c, 1, 1))
        self.bias = nn.Parameter(torch.zeros(1, c, 1, 1))
        self.eps = eps

    def forward(self, x):
        dt = x.dtype
        x = x.float()
        mu = x.mean(1, keepdim=True)
        var = (x - mu).pow(2).mean(1, keepdim=True)
        x = (x - mu) / torch.sqrt(var + self.eps)
        return (x * self.weight.float() + self.bias.float()).to(dt)


class SimpleGate(nn.Module):
    def forward(self, x):
        a, b = x.chunk(2, dim=1)
        return a * b


class NAFBlock(nn.Module):
    def __init__(self, c, dw_expand=2, ffn_expand=2):
        super().__init__()
        dw = c * dw_expand
        self.conv1 = nn.Conv2d(c, dw, 1)
        self.conv2 = nn.Conv2d(dw, dw, 3, padding=1, groups=dw)
        self.conv3 = nn.Conv2d(dw // 2, c, 1)
        self.sca = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Conv2d(dw // 2, dw // 2, 1))
        self.sg = SimpleGate()
        ffn = c * ffn_expand
        self.conv4 = nn.Conv2d(c, ffn, 1)
        self.conv5 = nn.Conv2d(ffn // 2, c, 1)
        self.norm1, self.norm2 = LayerNorm2d(c), LayerNorm2d(c)
        self.beta = nn.Parameter(torch.zeros(1, c, 1, 1))
        self.gamma = nn.Parameter(torch.zeros(1, c, 1, 1))

    def forward(self, inp):
        x = self.sg(self.conv2(self.conv1(self.norm1(inp))))
        x = self.conv3(x * self.sca(x))
        y = inp + x * self.beta
        x = self.conv5(self.sg(self.conv4(self.norm2(y))))
        return y + x * self.gamma


class NAFNet(nn.Module):
    def __init__(self, img_channel=1, width=24, middle_blk_num=2, enc_blk_nums=(1, 1, 2, 2), dec_blk_nums=(1, 1, 1, 1)):
        super().__init__()
        self.intro = nn.Conv2d(img_channel, width, 3, padding=1)
        self.ending = nn.Conv2d(width, img_channel, 3, padding=1)
        self.encoders, self.downs = nn.ModuleList(), nn.ModuleList()
        self.decoders, self.ups = nn.ModuleList(), nn.ModuleList()
        chan = width
        for n in enc_blk_nums:
            self.encoders.append(nn.Sequential(*[NAFBlock(chan) for _ in range(n)]))
            self.downs.append(nn.Conv2d(chan, 2 * chan, 2, 2))
            chan *= 2
        self.middle = nn.Sequential(*[NAFBlock(chan) for _ in range(middle_blk_num)])
        for n in dec_blk_nums:
            self.ups.append(nn.Sequential(nn.Conv2d(chan, chan * 2, 1, bias=False), nn.PixelShuffle(2)))
            chan //= 2
            self.decoders.append(nn.Sequential(*[NAFBlock(chan) for _ in range(n)]))
        self.multiple = 2 ** len(enc_blk_nums)

    def forward(self, inp):
        x = self.intro(inp)
        skips = []
        for enc, down in zip(self.encoders, self.downs):
            x = enc(x)
            skips.append(x)
            x = down(x)
        x = self.middle(x)
        for dec, up, skip in zip(self.decoders, self.ups, skips[::-1]):
            x = dec(up(x) + skip)
        return self.ending(x) + inp  # predict the correction only


# ───────────────────────────────────────────────────── plain autoencoder
class ConvAutoencoder(nn.Module):
    """Encoder (3 stride-2 downsamplings) -> bottleneck -> decoder. No skips, no residual."""

    def __init__(self, img_channel=1, width=32):
        super().__init__()
        w = width
        act = lambda: nn.ReLU(inplace=True)  # noqa: E731
        self.encoder = nn.Sequential(
            nn.Conv2d(img_channel, w, 3, padding=1), act(),
            nn.Conv2d(w, 2 * w, 3, stride=2, padding=1), act(),
            nn.Conv2d(2 * w, 4 * w, 3, stride=2, padding=1), act(),
            nn.Conv2d(4 * w, 4 * w, 3, stride=2, padding=1), act(),
            nn.Conv2d(4 * w, 4 * w, 3, padding=1), act(),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(4 * w, 4 * w, 4, stride=2, padding=1), act(),
            nn.ConvTranspose2d(4 * w, 2 * w, 4, stride=2, padding=1), act(),
            nn.ConvTranspose2d(2 * w, w, 4, stride=2, padding=1), act(),
            nn.Conv2d(w, img_channel, 3, padding=1),
        )
        self.multiple = 8

    def forward(self, x):
        return self.decoder(self.encoder(x))


# ───────────────────────────────────────────────────────────────── wrapper
class Deblurrer(nn.Module):
    """Takes reflectance (N, 1, H, W) of any size, returns deblurred reflectance."""

    def __init__(self, net: nn.Module, scale: float = INPUT_SCALE):
        super().__init__()
        self.net = net
        self.register_buffer("scale", torch.tensor(float(scale)))

    def forward(self, x):
        h, w = x.shape[-2:]
        m = self.net.multiple
        ph, pw = (m - h % m) % m, (m - w % m) % m
        if ph or pw:
            x = F.pad(x, (0, pw, 0, ph), mode="reflect")
        y = self.net(x * self.scale) / self.scale
        return y[..., :h, :w]


def build(name: str) -> Deblurrer:
    if name == "nafnet":
        return Deblurrer(NAFNet())
    if name == "autoencoder":
        return Deblurrer(ConvAutoencoder())
    raise ValueError(name)


def count_parameters(model) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


@torch.no_grad()
def tiled(model, x, tile=256, overlap=32, batch=16):
    """Run a single-band model over a (C, H, W) or (H, W) tensor in overlapping tiles, band by band.

    Tiles are blended with a smooth window so seams do not show."""
    squeeze = x.dim() == 2
    if squeeze:
        x = x[None]
    C, H, W = x.shape
    if H <= tile and W <= tile:
        out = model(x[:, None])[:, 0]
        return out[0] if squeeze else out
    step = tile - overlap
    ys = list(range(0, max(H - tile, 0) + 1, step))
    xs = list(range(0, max(W - tile, 0) + 1, step))
    if ys[-1] + tile < H:
        ys.append(H - tile)
    if xs[-1] + tile < W:
        xs.append(W - tile)
    ramp = torch.ones(tile, device=x.device)
    r = torch.linspace(0.05, 1, overlap, device=x.device)
    ramp[:overlap], ramp[-overlap:] = r, r.flip(0)
    win = ramp[:, None] * ramp[None, :]
    acc = torch.zeros_like(x)
    wsum = torch.zeros(H, W, device=x.device)
    coords = [(yy, xx) for yy in ys for xx in xs]
    for i in range(0, len(coords), max(1, batch // C)):
        cs = coords[i: i + max(1, batch // C)]
        tiles = torch.cat([x[:, yy: yy + tile, xx: xx + tile] for yy, xx in cs])[:, None]
        pred = model(tiles)[:, 0].float().reshape(len(cs), C, tile, tile)
        for (yy, xx), p in zip(cs, pred):
            acc[:, yy: yy + tile, xx: xx + tile] += p * win
            wsum[yy: yy + tile, xx: xx + tile] += win
    out = acc / wsum
    return out[0] if squeeze else out
