import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from timm.models.layers import DropPath, to_2tuple, trunc_normal_


def make_model(args, parent=False):
    """
    与主框架对接的模型构造函数，外部会调用:
        model = make_model(args)
    这里直接返回 swinOIR(args) 的实例。
    """
    return swinOIR(args)

# ---------------------------------------------------------
# 1. 基础组件： Mlp, WindowAttention, SwinTransformerBlock
# ---------------------------------------------------------
class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


def window_partition(x, window_size):
    """
    把 feature map (B, H, W, C) 切分成不重叠的 window，每个 window 大小为 window_size x window_size。
    返回形状: (num_windows*B, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows


def window_reverse(windows, window_size, H, W):
    """
    把分块后的 window 重新拼回原图大小 (B, H, W, C)
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


class WindowAttention(nn.Module):
    """
    SwinTransformer 中的窗口多头自注意力 (Window-based Multi-head Self-Attention)
    """
    def __init__(self, dim, window_size, num_heads,
                 qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()

        self.dim = dim
        self.window_size = window_size  # (window_height, window_width)
        self.num_heads = num_heads

        # 计算相对位置偏置表 (relative position bias table)
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads)
        )

        # 计算 relative_position_index，用于检索 bias
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # shape=(2,wh,ww)
        coords_flatten = torch.flatten(coords, 1)                  # shape=(2, wh*ww)

        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # (2, wh*ww, wh*ww)
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()            # (wh*ww, wh*ww, 2)

        # 偏移，使 index 范围 >= 0
        relative_coords[:, :, 0] += self.window_size[0] - 1
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= (2 * self.window_size[1] - 1)
        relative_position_index = relative_coords.sum(-1)
        self.register_buffer("relative_position_index", relative_position_index)

        # qkv
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        # 注意力相关
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

        # 额外：在 SwinTransformer 原论文中，qk_scale = head_dim**-0.5，若没有显式给出，就内部自行计算
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim**-0.5

    def forward(self, x, mask=None):
        """
        x: (B_, N, C)
        mask: (nW, N, N) or None
        """
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))  # shape=(B_, num_heads, N, N)

        # 加入相对位置偏置
        relative_position_bias = self.relative_position_bias_table[
            self.relative_position_index.view(-1)
        ].view(self.window_size[0]*self.window_size[1], self.window_size[0]*self.window_size[1], -1)
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # (num_heads, N, N)
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            # mask.shape[0] = nW, number of windows
            nW = mask.shape[0]
            attn = attn.view(B_//nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

    def flops(self, N):
        """
        粗略计算 FLOPs
        """
        flops = 0
        # qkv
        flops += N * self.dim * 3 * self.dim
        # q * k
        flops += self.num_heads * N * (self.dim // self.num_heads) * N
        # attn * v
        flops += self.num_heads * N * N * (self.dim // self.num_heads)
        # proj
        flops += N * self.dim * self.dim
        return flops


class SwinTransformerBlock(nn.Module):
    """
    典型的 Swin Transformer Block，包含 WindowAttention + MLP，支持 shift window。
    """
    def __init__(self, dim, input_resolution, num_heads, window_size=7, shift_size=0,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution  # (H, W)
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio

        # 若当前分辨率比 window_size 小，则不做 shift
        if min(self.input_resolution) <= self.window_size:
            self.shift_size = 0
            self.window_size = min(self.input_resolution)
        assert 0 <= self.shift_size < self.window_size, "shift_size >= 0 && < window_size"

        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(
            dim, window_size=to_2tuple(self.window_size),
            num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop=attn_drop, proj_drop=drop
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        # 预先计算好 shift window 的 mask
        if self.shift_size > 0:
            attn_mask = self.calculate_mask(self.input_resolution)
        else:
            attn_mask = None
        self.register_buffer("attn_mask", attn_mask)

    def calculate_mask(self, x_size):
        """
        计算带有 shift_size 时，对每个 window 的遮挡（mask），保证不在同一个窗口内的像素不互相注意。
        """
        H, W = x_size
        img_mask = torch.zeros((1, H, W, 1))
        h_slices = (
            slice(0, -self.window_size),
            slice(-self.window_size, -self.shift_size),
            slice(-self.shift_size, None)
        )
        w_slices = (
            slice(0, -self.window_size),
            slice(-self.window_size, -self.shift_size),
            slice(-self.shift_size, None)
        )

        cnt = 0
        for h in h_slices:
            for w in w_slices:
                img_mask[:, h, w, :] = cnt
                cnt += 1

        mask_windows = window_partition(img_mask, self.window_size)  # (nW, wsize, wsize, 1)
        mask_windows = mask_windows.view(-1, self.window_size*self.window_size)
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        return attn_mask

    def forward(self, x, x_size):
        """
        x: (B, L, C), L=H*W
        x_size: (H, W)
        """
        H, W = x_size
        B, L, C = x.shape
        shortcut = x

        # 1) LN
        x = self.norm1(x)
        # 2) reshape -> (B, H, W, C)
        x = x.view(B, H, W, C)

        # 3) shift
        if self.shift_size > 0:
            shifted_x = torch.roll(
                x, shifts=(-self.shift_size, -self.shift_size),
                dims=(1, 2)
            )
        else:
            shifted_x = x

        # 4) window partition
        x_windows = window_partition(shifted_x, self.window_size)  # (nW*B, wsize, wsize, C)
        x_windows = x_windows.view(-1, self.window_size*self.window_size, C)

        # 5) W-MSA or SW-MSA
        if self.input_resolution == x_size:
            # 若 input_resolution 与当前 x_size 匹配，就用事先的 attn_mask
            attn_windows = self.attn(x_windows, mask=self.attn_mask)
        else:
            # 否则重新计算 mask
            cur_mask = self.calculate_mask(x_size).to(x.device)
            attn_windows = self.attn(x_windows, mask=cur_mask)

        # 6) window reverse
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        shifted_x = window_reverse(attn_windows, self.window_size, H, W)

        # 7) reverse shift
        if self.shift_size > 0:
            x = torch.roll(
                shifted_x, shifts=(self.shift_size, self.shift_size),
                dims=(1, 2)
            )
        else:
            x = shifted_x

        # 8) reshape -> (B, L, C)
        x = x.view(B, H*W, C)

        # 9) FFN + drop_path
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x

    def flops(self):
        """
        估计 FLOPs
        """
        flops = 0
        H, W = self.input_resolution
        # LN1
        flops += self.dim * H * W
        # WindowAttention
        nW = H * W / self.window_size / self.window_size
        flops += nW * self.attn.flops(self.window_size * self.window_size)
        # MLP
        flops += 2 * H * W * self.dim * self.dim * self.mlp_ratio
        # LN2
        flops += self.dim * H * W
        return flops

# ---------------------------------------------------------
# 2. 下采样/合并模块 & BasicLayer & IDSTB
# ---------------------------------------------------------
class PatchMerging(nn.Module):
    """
    Swin 中的 patch merging，用于缩小分辨率，通道数加倍
    """
    def __init__(self, input_resolution, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.reduction = nn.Linear(4*dim, 2*dim, bias=False)
        self.norm = norm_layer(4*dim)

    def forward(self, x):
        """
        x: (B, H*W, C)
        """
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H*W, "input feature has wrong size"
        assert H % 2 == 0 and W % 2 == 0, f"x size ({H},{W}) not even."

        x = x.view(B, H, W, C)
        # 将相邻 2x2 区域 concat 在通道维
        x0 = x[:, 0::2, 0::2, :]
        x1 = x[:, 1::2, 0::2, :]
        x2 = x[:, 0::2, 1::2, :]
        x3 = x[:, 1::2, 1::2, :]
        x = torch.cat([x0, x1, x2, x3], dim=-1)
        x = x.view(B, -1, 4*C)

        x = self.norm(x)
        x = self.reduction(x)
        return x

    def flops(self):
        H, W = self.input_resolution
        flops = H*W * self.dim
        flops += (H//2)*(W//2)*4*self.dim * 2*self.dim
        return flops


class BasicLayer(nn.Module):
    """
    由多个 SwinTransformerBlock 组成的一层 (stage)
    可选 downsample (PatchMerging)
    """
    def __init__(self, dim, input_resolution, depth, num_heads, window_size,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0.,
                 attn_drop=0., drop_path=0., norm_layer=nn.LayerNorm,
                 downsample=None, use_checkpoint=False):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.use_checkpoint = use_checkpoint

        # 构建多个 SwinTransformerBlock
        if isinstance(drop_path, float):
            drop_path = [drop_path]*depth  # 若传进来是单值，则对每个 block 都用相同值
        assert len(drop_path) == depth, "drop_path 的长度应与 depth 相同"

        self.blocks = nn.ModuleList([
            SwinTransformerBlock(
                dim=dim,
                input_resolution=input_resolution,
                num_heads=num_heads,
                window_size=window_size,
                shift_size=0 if (i % 2 == 0) else window_size // 2,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop, attn_drop=attn_drop,
                drop_path=drop_path[i],
                norm_layer=norm_layer
            )
            for i in range(depth)
        ])

        # 如果需要下采样 (patch merging)
        if downsample is not None:
            self.downsample = downsample(input_resolution, dim=dim, norm_layer=norm_layer)
        else:
            self.downsample = None

    def forward(self, x, x_size):
        for blk in self.blocks:
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x, x_size)
            else:
                x = blk(x, x_size)

        if self.downsample is not None:
            x = self.downsample(x)
        return x

    def flops(self):
        flops = 0
        for blk in self.blocks:
            flops += blk.flops()
        if self.downsample is not None:
            flops += self.downsample.flops()
        return flops


class IDSTB(nn.Module):
    """
    这一层对 BasicLayer 再做一个封装，并带有 residual connection (conv + embed/unembed).
    """
    def __init__(self, dim, input_resolution, depth, num_heads, window_size,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, downsample=None,
                 use_checkpoint=False, img_size=224, patch_size=4, resi_connection='1conv'):
        super(IDSTB, self).__init__()

        self.dim = dim
        self.input_resolution = input_resolution

        self.residual_group = BasicLayer(
            dim=dim, input_resolution=input_resolution, depth=depth,
            num_heads=num_heads, window_size=window_size, mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias, qk_scale=qk_scale, drop=drop, attn_drop=attn_drop,
            drop_path=drop_path, norm_layer=norm_layer,
            downsample=downsample, use_checkpoint=use_checkpoint
        )

        # 残差连接的卷积
        if resi_connection == '1conv':
            self.conv = nn.Conv2d(dim, dim, 3, 1, 1)
        elif resi_connection == '3conv':
            # 更复杂的3卷积形式
            self.conv = nn.Sequential(
                nn.Conv2d(dim, dim//4, 3, 1, 1),
                nn.LeakyReLU(negative_slope=0.2, inplace=True),
                nn.Conv2d(dim//4, dim//4, 1, 1, 0),
                nn.LeakyReLU(negative_slope=0.2, inplace=True),
                nn.Conv2d(dim//4, dim, 3, 1, 1)
            )
        else:
            raise ValueError("resi_connection must be '1conv' or '3conv'")

        # 用于把 (B, HW, C) <-> (B, C, H, W) 来回变换
        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size,
            in_chans=0, embed_dim=dim, norm_layer=None
        )
        self.patch_unembed = PatchUnEmbed(
            img_size=img_size, patch_size=patch_size,
            in_chans=0, embed_dim=dim, norm_layer=None
        )

    def forward(self, x, x_size):
        """
        x: (B, L, C)
        x_size: (H, W)
        """
        # 先过 BasicLayer
        out = self.residual_group(x, x_size)  # (B, L, C)
        # unembed -> conv -> embed
        out_img = self.patch_unembed(out, x_size)  # (B, C, H, W)
        out_img = self.conv(out_img)
        out_feat = self.patch_embed(out_img)       # (B, HW, C)

        return out_feat + x  # residual

    def flops(self):
        flops = 0
        flops += self.residual_group.flops()
        H, W = self.input_resolution
        # conv: kernel_size=3 => 9
        flops += H * W * self.dim * self.dim * 9
        flops += self.patch_embed.flops()
        flops += self.patch_unembed.flops()
        return flops

# ---------------------------------------------------------
# 3. PatchEmbed, PatchUnEmbed, Upsample
# ---------------------------------------------------------
class PatchEmbed(nn.Module):
    """
    将特征 (B, C, H, W) 展平 -> (B, HW, C)，可选做 LN
    """
    def __init__(self, img_size=224, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        patches_resolution = [img_size[0]//patch_size[0], img_size[1]//patch_size[1]]

        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0]*patches_resolution[1]
        self.in_chans = in_chans
        self.embed_dim = embed_dim

        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        # x: (B, C, H, W)，或者 (B, in_chans, H, W)
        # 先 flatten: (B, C, H*W) -> 再转置: (B, H*W, C)
        x = x.flatten(2).transpose(1, 2)
        if self.norm is not None:
            x = self.norm(x)
        return x

    def flops(self):
        flops = 0
        H, W = self.img_size
        # 如果有 LN
        if self.norm is not None:
            flops += H*W * self.embed_dim
        return flops


class PatchUnEmbed(nn.Module):
    """
    将 (B, HW, C) 还原成 (B, C, H, W)
    """
    def __init__(self, img_size=224, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        patches_resolution = [img_size[0]//patch_size[0], img_size[1]//patch_size[1]]

        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0]*patches_resolution[1]
        self.in_chans = in_chans
        self.embed_dim = embed_dim

        # norm_layer 暂时不处理

    def forward(self, x, x_size):
        """
        x: (B, HW, C)
        x_size: (H, W)
        """
        B, HW, C = x.shape
        # reshape -> (B, C, H, W)
        x = x.transpose(1, 2).view(B, self.embed_dim, x_size[0], x_size[1])
        return x

    def flops(self):
        return 0


class Upsample(nn.Sequential):
    """
    常规的 PixelShuffle 上采样。
    若 scale=4，就重复两次 PixelShuffle(2)
    """
    def __init__(self, scale, num_feat):
        m = []
        # 判断是否是 2^n 或 3
        if (scale & (scale - 1)) == 0:  # 2,4,8,...
            for _ in range(int(math.log(scale, 2))):
                m.append(nn.Conv2d(num_feat, 4*num_feat, 3, 1, 1))
                m.append(nn.PixelShuffle(2))
        elif scale == 3:
            m.append(nn.Conv2d(num_feat, 9*num_feat, 3, 1, 1))
            m.append(nn.PixelShuffle(3))
        else:
            raise ValueError(f'Unsupported scale {scale}. Only 2^n or 3 are valid.')
        super(Upsample, self).__init__(*m)


# ---------------------------------------------------------
# 4. 主体: swinOIR
# ---------------------------------------------------------
class swinOIR(nn.Module):
    """
    最终的超分网络：将以上模块组合起来。
    """
    def __init__(self,
                 # --------------------- 关键超参 ---------------------
                 img_size=64,
                 patch_size=1,
                 in_chans=3,
                 embed_dim=96,
                 num_heads=[6,6,6,6],
                 window_size=8,
                 mlp_ratio=4.,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.1,   # 用于随机深度
                 qkv_bias=True,
                 qk_scale=None,
                 norm_layer=nn.LayerNorm,
                 ape=False,
                 patch_norm=True,
                 upscale=4,
                 img_range=1.,
                 upsampler='pixelshuffle',
                 resi_connection='1conv',
                 depths=[4,4,4,4],    # 每个stage的层数
                 use_checkpoint=False,
                 # --------------------- 其他可选 ---------------------
                 **kwargs):
        super(swinOIR, self).__init__()
        self.img_range = img_range
        self.upscale = upscale
        self.upsampler = upsampler
        self.window_size = window_size
        self.ape = ape
        self.patch_norm = patch_norm
        self.num_features = embed_dim
        self.mlp_ratio = mlp_ratio

        num_in_ch = in_chans
        num_out_ch = in_chans
        num_feat = 64

        # 均值
        if in_chans == 3:
            rgb_mean = (0.4488, 0.4371, 0.4040)
            self.mean = torch.Tensor(rgb_mean).view(1, 3, 1, 1)
        else:
            self.mean = torch.zeros(1, 1, 1, 1)

        # 首个卷积
        self.conv_first = nn.Conv2d(num_in_ch, embed_dim, 3, 1, 1)

        # patch embed/unembed
        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=embed_dim,
            embed_dim=embed_dim, norm_layer=norm_layer if patch_norm else None
        )
        self.patch_unembed = PatchUnEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=embed_dim,
            embed_dim=embed_dim, norm_layer=norm_layer if patch_norm else None
        )
        patches_resolution = self.patch_embed.patches_resolution
        self.patches_resolution = patches_resolution  # (H//psize, W//psize)

        # 绝对位置编码
        num_patches = self.patch_embed.num_patches
        if self.ape:
            self.absolute_pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
            trunc_normal_(self.absolute_pos_embed, std=.02)

        self.pos_drop = nn.Dropout(p=drop_rate)

        # -------------------------------------------------------------
        # 计算 drop_path 的分布 (dpr)，给每个 block 分配一个不同的 drop_path 概率
        # -------------------------------------------------------------
        total_depth = sum(depths)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, total_depth)]

        # 构建 4 个阶段 (layer1 ~ layer4)，这里用 IDSTB 封装
        self.layer1 = IDSTB(
            dim=embed_dim,
            input_resolution=(patches_resolution[0], patches_resolution[1]),
            depth=depths[0],
            num_heads=num_heads[0],
            window_size=window_size,
            mlp_ratio=self.mlp_ratio,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            drop=drop_rate,
            attn_drop=attn_drop_rate,
            drop_path=dpr[sum(depths[:0]):sum(depths[:1])],  # depth[0] 个block
            norm_layer=norm_layer,
            downsample=None,
            use_checkpoint=use_checkpoint,
            img_size=img_size,
            patch_size=patch_size,
            resi_connection=resi_connection
        )

        self.layer2 = IDSTB(
            dim=embed_dim,
            input_resolution=(patches_resolution[0], patches_resolution[1]),
            depth=depths[1],
            num_heads=num_heads[1],
            window_size=window_size,
            mlp_ratio=self.mlp_ratio,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            drop=drop_rate,
            attn_drop=attn_drop_rate,
            drop_path=dpr[sum(depths[:1]):sum(depths[:2])],  # depth[1] 个block
            norm_layer=norm_layer,
            downsample=None,
            use_checkpoint=use_checkpoint,
            img_size=img_size,
            patch_size=patch_size,
            resi_connection=resi_connection
        )

        self.layer3 = IDSTB(
            dim=embed_dim,
            input_resolution=(patches_resolution[0], patches_resolution[1]),
            depth=depths[2],
            num_heads=num_heads[2],
            window_size=window_size,
            mlp_ratio=self.mlp_ratio,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            drop=drop_rate,
            attn_drop=attn_drop_rate,
            drop_path=dpr[sum(depths[:2]):sum(depths[:3])],  # depth[2]
            norm_layer=norm_layer,
            downsample=None,
            use_checkpoint=use_checkpoint,
            img_size=img_size,
            patch_size=patch_size,
            resi_connection=resi_connection
        )

        self.layer4 = IDSTB(
            dim=embed_dim,
            input_resolution=(patches_resolution[0], patches_resolution[1]),
            depth=depths[3],
            num_heads=num_heads[3],
            window_size=window_size,
            mlp_ratio=self.mlp_ratio,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            drop=drop_rate,
            attn_drop=attn_drop_rate,
            drop_path=dpr[sum(depths[:3]):sum(depths[:4])],  # depth[3]
            norm_layer=norm_layer,
            downsample=None,
            use_checkpoint=use_checkpoint,
            img_size=img_size,
            patch_size=patch_size,
            resi_connection=resi_connection
        )

        # 将4个layer加到 self.layers (便于 flops 统计)
        self.layers = nn.ModuleList([self.layer1, self.layer2, self.layer3, self.layer4])

        self.norm = norm_layer(self.num_features)

        # residual conv
        if resi_connection == '1conv':
            self.conv_after_body = nn.Conv2d(embed_dim, embed_dim, 3, 1, 1)
        elif resi_connection == '3conv':
            # 如果你改成 '3conv'，需要同步修改
            self.conv_after_body = nn.Sequential(
                nn.Conv2d(embed_dim, embed_dim//4, 3, 1, 1),
                nn.LeakyReLU(negative_slope=0.2, inplace=True),
                nn.Conv2d(embed_dim//4, embed_dim//4, 1, 1, 0),
                nn.LeakyReLU(negative_slope=0.2, inplace=True),
                nn.Conv2d(embed_dim//4, embed_dim, 3, 1, 1)
            )
        else:
            raise ValueError("resi_connection must be '1conv' or '3conv'")

        # 上采样模块
        if self.upsampler == 'pixelshuffle':
            self.conv_before_upsample = nn.Sequential(
                nn.Conv2d(embed_dim, num_feat, 3, 1, 1),
                nn.LeakyReLU(inplace=True)
            )
            self.upsample = Upsample(upscale, num_feat)
            self.conv_last = nn.Conv2d(num_feat, num_out_ch, 3, 1, 1)
        elif self.upsampler == 'pixelshuffledirect':
            # 另一种写法, 省略中间的 conv_before_upsample
            self.upsample = Upsample(upscale, embed_dim)
            self.conv_last = nn.Conv2d(embed_dim, num_out_ch, 3, 1, 1)
        else:
            raise ValueError("upsampler should be 'pixelshuffle' or 'pixelshuffledirect'")

        # 初始化
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def check_image_size(self, x):
        """
        保证输入分辨率能被 window_size 整除，不够则反射填充
        """
        _, _, h, w = x.size()
        mod_pad_h = (self.window_size - h % self.window_size) % self.window_size
        mod_pad_w = (self.window_size - w % self.window_size) % self.window_size
        x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h), mode='reflect')
        return x

    def forward_features(self, x):
        """
        Swin 主干。x: (B, C, H, W) -> patch_embed -> (B, HW, C)
        再经过若干层 IDSTB，最后 LN + patch_unembed 恢复为 (B, C, H, W)
        """
        x_size = (x.shape[2], x.shape[3])  # (H, W)
        x = self.patch_embed(x)
        if self.ape:
            x = x + self.absolute_pos_embed
        x = self.pos_drop(x)

        # ===== 下面是你自定义的 “dense 乘法” 方式 =====
        block1 = self.layer1(x, x_size)
        block2 = self.layer2(block1, x_size)
        b2_dense = block1 * block2

        block3 = self.layer3(b2_dense, x_size)
        b3_dense = block1 * block2 * block3

        block4 = self.layer4(b3_dense, x_size)
        # 你这里 b4_dense 少了 block2? 可能故意或笔误
        b4_dense = block1 * block3 * block4  
        # 如果你想全部乘: b4_dense = block1 * block2 * block3 * block4

        x = self.norm(b4_dense)
        x = self.patch_unembed(x, x_size)
        return x

    def forward(self, x):
        """
        整个超分流程
        """
        H, W = x.shape[2:]
        # 1) 保证 (H,W) 能整除 window_size
        x = self.check_image_size(x)
        # 2) 减去均值
        self.mean = self.mean.type_as(x)
        x = (x - self.mean) * self.img_range

        # 3) 首先卷积
        feat = self.conv_first(x)

        # 4) 走主干 forward_features
        body_feat = self.forward_features(feat)
        body_feat = self.conv_after_body(body_feat) + feat  # Residual

        # 5) 上采样
        if self.upsampler == 'pixelshuffle':
            body_feat = self.conv_before_upsample(body_feat)
            body_feat = self.upsample(body_feat)
            out = self.conv_last(body_feat)
        elif self.upsampler == 'pixelshuffledirect':
            body_feat = self.upsample(body_feat)
            out = self.conv_last(body_feat)
        else:
            raise NotImplementedError

        # 6) 加回均值
        out = out / self.img_range + self.mean
        # 7) 裁剪回到原始的放大尺寸
        return out[:, :, :H*self.upscale, :W*self.upscale]

    def flops(self):
        """
        估算模型的 FLOPs
        """
        flops = 0
        H, W = self.patches_resolution
        # conv_first
        flops += H*W*3*self.embed_dim*9
        # patch_embed
        flops += self.patch_embed.flops()
        # 4 个阶段
        for layer in self.layers:
            flops += layer.flops()
        # conv_after_body
        flops += H*W*self.embed_dim*self.embed_dim*9
        # upsample
        if hasattr(self, 'upsample'):
            # 这里仅仅估计 PixelShuffle 卷积部分
            # 如果要精确，需要把 upsample 里的 Conv2d 都算进来
            pass
        return flops
