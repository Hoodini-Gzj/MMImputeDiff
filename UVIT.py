import torch

import torch
import torch.nn as nn
import math
from Diffusiondir.timm_pri import trunc_normal_, Mlp
import einops
import torch.utils.checkpoint
from torch.nn import ConvTranspose3d
# import xformers
# import xformers.ops
from torch.nn import functional as F
# if hasattr(torch.nn.functional, 'scaled_dot_product_attention'):
ATTENTION_MODE = 'math'
# else:
#     try:
#         import xformers
#         import xformers.ops
#         ATTENTION_MODE = 'xformers'
#     except:
#         ATTENTION_MODE = 'math'
print(f'attention mode is {ATTENTION_MODE}')


def timestep_embedding(timesteps, dim, max_period=10000):
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
    ).to(device=timesteps.device)
    args = timesteps[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding


def patchify(imgs, patch_size):
    x = einops.rearrange(imgs, 'B C (h p1) (w p2) -> B (h w) (p1 p2 C)', p1=patch_size, p2=patch_size)
    return x


def unpatchify(x, channels=3):
    patch_size = 8
    # h = w = d = 8 112, 128, 160
    h = 12
    w = 12
    d = 12
    x = einops.rearrange(x, 'B (h w d) (p1 p2 p3 C) -> B C (h p1) (w p2) (d p3)',
                         h=h, w=w, d=d, p1=patch_size, p2=patch_size, p3=patch_size)
    return x

def unpatchify_xray(x, channels=3):
    patch_size = 8
    # h = w = d = 8
    h = 24
    w = 24
    x = einops.rearrange(x, 'B (h w) (p1 p2 C) -> B C (h p1) (w p2)',
                         h=h, w=w, p1=patch_size, p2=patch_size)
    return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, L, C = x.shape

        qkv = self.qkv(x)
        # ATTENTION_MODE = 'flash'
        if ATTENTION_MODE == 'flash':
            qkv = einops.rearrange(qkv, 'B L (K H D) -> K B H L D', K=3, H=self.num_heads).float()
            q, k, v = qkv[0], qkv[1], qkv[2]  # B H L D
            x = torch.nn.functional.scaled_dot_product_attention(q, k, v)
            x = einops.rearrange(x, 'B H L D -> B L (H D)')
        elif ATTENTION_MODE == 'xformers':
            qkv = einops.rearrange(qkv, 'B L (K H D) -> K B L H D', K=3, H=self.num_heads)
            q, k, v = qkv[0], qkv[1], qkv[2]  # B L H D
            x = xformers.ops.memory_efficient_attention(q, k, v)
            x = einops.rearrange(x, 'B L H D -> B L (H D)', H=self.num_heads)
        elif ATTENTION_MODE == 'math':
            qkv = einops.rearrange(qkv, 'B L (K H D) -> K B H L D', K=3, H=self.num_heads)
            q, k, v = qkv[0], qkv[1], qkv[2]  # B H L D
            attn = (q @ k.transpose(-2, -1)) * self.scale
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            x = (attn @ v).transpose(1, 2).reshape(B, L, C)
        # else:
        #     raise NotImplemented

        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm, skip=False, use_checkpoint=False):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale)
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer)
        self.skip_linear = nn.Linear(2 * dim, dim) if skip else None
        self.use_checkpoint = use_checkpoint

    def forward(self, x, skip=None):
        if self.use_checkpoint:
            return torch.utils.checkpoint.checkpoint(self._forward, x, skip)
        else:
            return self._forward(x, skip)

    def _forward(self, x, skip=None):
        if self.skip_linear is not None:
            x = self.skip_linear(torch.cat([x, skip], dim=-1))
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x

class TimeEmbedding(nn.Module):
    def __init__(self, T, d_model, dim):
        assert d_model % 2 == 0
        super().__init__()
        emb = torch.arange(0, d_model, step=2) / d_model * math.log(10000)
        emb = torch.exp(-emb)
        pos = torch.arange(T).float()
        emb = pos[:, None] * emb[None, :]
        assert list(emb.shape) == [T, d_model // 2]
        emb = torch.stack([torch.sin(emb), torch.cos(emb)], dim=-1)
        assert list(emb.shape) == [T, d_model // 2, 2]
        emb = emb.view(T, d_model)

        self.timembedding = nn.Sequential(
            nn.Embedding.from_pretrained(emb),
            nn.Linear(d_model, dim),
            Swish(),
            nn.Linear(dim, dim),
        )
    def forward(self, t):
        emb = self.timembedding(t)
        return emb

class ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, tdim, dropout):
        super().__init__()
        self.block1 = nn.Sequential(
            nn.GroupNorm(8, in_ch),
            Swish(),
            nn.Conv3d(in_ch, out_ch, 3, stride=1, padding=1),
        )
        self.temb_proj = nn.Sequential(
            Swish(),
            nn.Linear(tdim, out_ch),
        )
        self.block2 = nn.Sequential(
            nn.GroupNorm(8, out_ch),
            Swish(),
            nn.Dropout(dropout),
            nn.Conv3d(out_ch, out_ch, 3, stride=1, padding=1),
        )

    def forward(self, x, temb):
        h = self.block1(x)
        temp = self.temb_proj(temb)
        temp = temp[:, :, None, None, None]
        h += temp
        h = self.block2(h)

        return h


class UpSample(nn.Module):
    def __init__(self,):
        super().__init__()
        self.main = nn.Conv3d(1, 1, 3, stride=1, padding=1)

    def forward(self, x):
        _, _, H, W, D = x.shape
        x = F.interpolate(
            x, scale_factor=2, mode='trilinear')
        x = self.main(x)
        return x

class PatchEmbed(nn.Module):
    def __init__(self, patch_size, in_chans=1, embed_dim=768):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv3d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        B, C, H, W, D = x.shape
        assert H % self.patch_size[0] == 0 and W % self.patch_size[1] == 0 and D % self.patch_size[2] == 0
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x

class PatchEmbedXray(nn.Module):
    def __init__(self, patch_size, in_chans=1, embed_dim=768):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        B, C, H, W = x.shape
        assert H % self.patch_size == 0 and W % self.patch_size == 0
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x

class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)

class UViT(nn.Module):
    def __init__(self, img_size=(128, 128, 128), img_size_xray=(192, 192), patch_size=8, patch_size_3d=(8, 8, 8), in_chans=1, embed_dim=1032, depth=12, num_heads=12, mlp_ratio=4.,
                 qkv_bias=False, qk_scale=None, norm_layer=nn.LayerNorm, mlp_time_embed=False, num_classes=-1,
                 use_checkpoint=False, conv=True, skip=True):
        super().__init__()
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        self.num_classes = num_classes
        self.in_chans = in_chans

        self.patch_embed = PatchEmbed(patch_size=patch_size_3d, in_chans=1, embed_dim=embed_dim)
        self.patch_embed_xray = PatchEmbedXray(patch_size=patch_size, in_chans=1, embed_dim=embed_dim)

        num_patches = (img_size[0] // patch_size_3d[0]) * (img_size[1] // patch_size_3d[1]) * (img_size[2] // patch_size_3d[2])
        num_patches_xray = (img_size_xray[0] // patch_size) * (img_size_xray[1] // patch_size)


        self.time_embed = nn.Sequential(
            nn.Linear(embed_dim, 4 * embed_dim),
            nn.SiLU(),
            nn.Linear(4 * embed_dim, embed_dim),
        ) if mlp_time_embed else nn.Identity()

        if self.num_classes > 0:
            self.label_emb = nn.Embedding(self.num_classes, embed_dim)
            self.extras = 2
        else:
            self.extras = 1

        self.pos_embed = nn.Parameter(torch.zeros(1, self.extras + num_patches + num_patches_xray, embed_dim))

        self.in_blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                norm_layer=norm_layer, use_checkpoint=use_checkpoint)
            for _ in range(depth // 2)])

        self.mid_block = Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                norm_layer=norm_layer, use_checkpoint=use_checkpoint)

        self.out_blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                norm_layer=norm_layer, skip=skip, use_checkpoint=use_checkpoint)
            for _ in range(depth // 2)])

        self.norm = norm_layer(embed_dim)

        self.patch_dim = patch_size_3d[0] * patch_size_3d[1] * patch_size_3d[2] * in_chans
        self.patch_dim_xray = patch_size ** 2 * in_chans

        self.decoder_pred = nn.Linear(embed_dim, self.patch_dim, bias=True)
        self.decoder_pred_xray = nn.Linear(embed_dim, self.patch_dim_xray, bias=True)

        self.final_layer = nn.Conv3d(self.in_chans, 1, 3, padding=1) if conv else nn.Identity()
        self.final_layer_xray = nn.Conv2d(self.in_chans, 1, 3, padding=1) if conv else nn.Identity()

        self.temb_proj = nn.Sequential(
            Swish(),
            nn.Linear(32, 32),
        )
        self.in_conv1 = nn.Conv3d(self.in_chans, 1, 3, stride=1, padding=1)
        self.in_conv2 = nn.Conv3d(16, 32, 3, stride=2, padding=1)

        self.out_conv1 = nn.ConvTranspose3d(32, 16, 3, stride=2, padding=1, output_padding=1)
        self.out_conv2 = nn.Conv3d(16, 1, 3, stride=1, padding=1)

        self.head = nn.Conv3d(1, 128, kernel_size=3, stride=1, padding=1)

        self.CNN = ResBlock(
            in_ch=128, out_ch=128, tdim=128 * 4,
            dropout=0.15)
        self.CNN2 = ResBlock2(
            in_ch=128, out_ch=128, tdim=128 * 4,
            dropout=0.15)

        self.time_embedding = TimeEmbedding(1000, 128, 128 * 4)
        self.upconv = UpSample()

        trunc_normal_(self.pos_embed, std=.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed'}

    def forward(self, xray, ct, timesteps):


        ct = self.patch_embed(ct)
        xray = self.patch_embed_xray(xray)

        B, L, D = ct.shape
        Bx, Lx, Dx = xray.shape

        time_token = self.time_embed(timestep_embedding(timesteps, self.embed_dim))
        time_token = time_token.unsqueeze(dim=1)

        input = torch.cat((xray, ct), dim=1)
        input = torch.cat((time_token, input), dim=1)

        input = input + self.pos_embed

        skips = []
        for blk in self.in_blocks:
            input = blk(input)
            skips.append(input)

        input = self.mid_block(input)

        for blk in self.out_blocks:
            input = blk(input, skips.pop())

        input = self.norm(input)
        assert input.size(1) == self.extras + L + Lx
        input = input[:, self.extras:, :]
        xray, ct = torch.split(input, [576, 1728], dim=1)#[256, 512]1728  4096
        xray_copy = xray
        ct_copy = ct
        # xray, ct = self.crossattention(xray, ct)
        ct = self.decoder_pred(ct)
        xray = self.decoder_pred_xray(xray)
        # input = input[:, self.extras:, :]
        # xray, ct = torch.split(input, [256, 512], dim=1)
        ct = unpatchify(ct, self.in_chans)
        ct = self.final_layer(ct)
        xray = unpatchify_xray(xray, self.in_chans)
        xray = self.final_layer_xray(xray)

        # xray, ct = self.crossattention(xray, ct, xray_copy, ct_copy)
        # x = self.upconv(x)
        # x = self.out_conv1(x)
        # x = self.out_conv2(x)
        # x = self.out_conv3(x)

        return xray, ct
