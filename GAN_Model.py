import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights

EPSILON = 1e-6
class DSConv(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, stride: int = 1):
        super().__init__()
        self.dw = nn.Conv2d(in_ch, in_ch, 3, stride=stride,
                            padding=1, groups=in_ch, bias=False)
        self.pw = nn.Conv2d(in_ch, out_ch, 1, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.ReLU6(inplace=True)

    def forward(self, x):
        return self.act(self.bn(self.pw(self.dw(x))))

class LightASPP(nn.Module):

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        mid = out_ch // 2

        self.b0 = nn.Sequential(
            nn.Conv2d(in_ch, mid, 1, bias=False),
            nn.BatchNorm2d(mid), nn.ReLU6(inplace=True),
        )
        self.b1 = nn.Sequential(
            nn.Conv2d(in_ch, mid, 3, padding=6, dilation=6,
                      groups=in_ch, bias=False),
            nn.Conv2d(mid, mid, 1, bias=False),
            nn.BatchNorm2d(mid), nn.ReLU6(inplace=True),
        )
        self.b2 = nn.Sequential(
            nn.Conv2d(in_ch, mid, 3, padding=12, dilation=12,
                      groups=in_ch, bias=False),
            nn.Conv2d(mid, mid, 1, bias=False),
            nn.BatchNorm2d(mid), nn.ReLU6(inplace=True),
        )
        self.pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_ch, mid, 1, bias=False),
            nn.ReLU6(inplace=True),
        )
        self.proj = nn.Sequential(
            nn.Conv2d(mid * 4, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch), nn.ReLU6(inplace=True),
            nn.Dropout2d(0.1),
        )

    def forward(self, x):
        B, _, H, W = x.shape
        pooled = self.pool(x).expand(B, -1, H, W)
        return self.proj(torch.cat([self.b0(x), self.b1(x),
                                    self.b2(x), pooled], dim=1))

class LightDecoderHead(nn.Module):
    def __init__(self, bottleneck_ch=96, skip_chs=(48, 24), dec_ch=64):
        super().__init__()
        self.up1   = DSConv(bottleneck_ch, dec_ch)
        self.skip1 = nn.Conv2d(skip_chs[0], dec_ch, 1, bias=False)
        self.fuse1 = DSConv(dec_ch * 2, dec_ch)

        self.up2   = DSConv(dec_ch, dec_ch // 2)
        self.skip2 = nn.Conv2d(skip_chs[1], dec_ch // 2, 1, bias=False)
        self.fuse2 = DSConv(dec_ch, dec_ch // 2)

        self.final = nn.Sequential(
            DSConv(dec_ch // 2, dec_ch // 4),
            nn.Conv2d(dec_ch // 4, 1, 1),
            nn.Sigmoid(),
        )

    def forward(self, bottleneck, skip2, skip3, out_size):
        x = F.interpolate(bottleneck, size=skip2.shape[2:],
                          mode="bilinear", align_corners=False)
        x = self.up1(x)
        x = torch.cat([x, self.skip1(skip2)], dim=1)
        x = self.fuse1(x)

        x = F.interpolate(x, size=skip3.shape[2:],
                          mode="bilinear", align_corners=False)
        x = self.up2(x)
        x = torch.cat([x, self.skip2(skip3)], dim=1)
        x = self.fuse2(x)

        x = F.interpolate(x, size=out_size, mode="bilinear", align_corners=False)
        return self.final(x)

class ConfidenceGenerator(nn.Module):
    def __init__(self):
        super().__init__()
        base    = mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.DEFAULT)
        features = base.features

        old = features[0][0]
        new = nn.Conv2d(5, 16, 3, stride=2, padding=1, bias=False)
        with torch.no_grad():
            new.weight[:, :3] = old.weight
            new.weight[:, 3:] = old.weight[:, :2]
        features[0][0] = new
        self.stage_early  = nn.Sequential(*features[:2])
        self.stage_s3     = nn.Sequential(*features[2:4])
        self.stage_s2     = nn.Sequential(*features[4:9])
        self.stage_bottle = nn.Sequential(*features[9:])

        self.bottleneck = LightASPP(in_ch=96, out_ch=96)

        self.head1 = LightDecoderHead(bottleneck_ch=96,
                                      skip_chs=(48, 24), dec_ch=64)
        self.head2 = LightDecoderHead(bottleneck_ch=96,
                                      skip_chs=(48, 24), dec_ch=64)

    def forward(self, rgb: torch.Tensor,d1: torch.Tensor,d2: torch.Tensor):

        H, W = rgb.shape[2], rgb.shape[3]
        x = torch.cat([rgb, d1, d2], dim=1)

        x     = self.stage_early(x)
        skip3 = self.stage_s3(x)
        skip2 = self.stage_s2(skip3)
        enc   = self.stage_bottle(skip2)

        bottle = self.bottleneck(enc)

        c1 = self.head1(bottle, skip2, skip3, (H, W))
        c2 = self.head2(bottle, skip2, skip3, (H, W))

        d_fused = (c1 * d1 + c2 * d2) / (c1 + c2 + EPSILON)

        return c1, c2, d_fused


class PatchDiscriminator(nn.Module):
    def __init__(self, in_ch: int = 4, ndf: int = 32):
        super().__init__()

        def block(ci, co, stride=2, norm=True):
            layers = [nn.Conv2d(ci, co, 4, stride=stride,
                                padding=1, bias=not norm)]
            if norm:
                layers.append(nn.InstanceNorm2d(co, affine=True))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return nn.Sequential(*layers)

        self.net = nn.Sequential(
            block(in_ch,    ndf,     norm=False),
            block(ndf,      ndf * 2),
            block(ndf * 2,  ndf * 4),
            block(ndf * 4,  ndf * 4, stride=1),
            nn.Conv2d(ndf * 4, 1, 4, padding=1),
        )

    def forward(self, rgb: torch.Tensor, depth: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([rgb, depth], dim=1))

def gradient_smoothness_loss(depth: torch.Tensor) -> torch.Tensor:
    dx = depth[:, :, :, 1:] - depth[:, :, :, :-1]
    dy = depth[:, :, 1:, :] - depth[:, :, :-1, :]
    return dx.abs().mean() + dy.abs().mean()


def confidence_sum_loss(c1: torch.Tensor, c2: torch.Tensor) -> torch.Tensor:
    return (c1 + c2 - 1.0).abs().mean()


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)