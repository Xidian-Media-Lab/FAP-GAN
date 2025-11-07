import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from networkx.utils.misc import groups
import torch.nn.utils.spectral_norm as SpectralNorm
import matplotlib.pyplot as plt
import numpy as np
import cv2

# Helper Modules #
class DepthwiseSeparableConv(nn.Module):
    """轻量化深度可分离卷积"""
    def __init__(self, in_ch, out_ch, kernel_size=3, stride=1, padding=1, use_spectral_norm = False):
        super().__init__()
        self.depthwise = nn.Conv2d(in_ch, in_ch, kernel_size, 
                                  stride, padding, groups=in_ch)
        self.pointwise = nn.Conv2d(in_ch, out_ch, 1)
        
    def forward(self, x):
        return self.pointwise(self.depthwise(x))
    
class global_MDTA(nn.Module):
    def __init__(self, dim, num_heads = 4, bias = False):
        super().__init__()
        self.num_heads = num_heads
        self.dim = dim
        assert  dim % num_heads ==0
        self.head_dim = dim//num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv = nn.Conv2d(dim, dim*3, kernel_size=1, bias=bias)
        self.dwconv = nn.Conv2d(dim*3, dim*3, kernel_size=3, stride=1, padding=1, groups=dim*3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias = bias)

    def forward(self, x_q, x_kv):
        """
        x_q = query(comp) [B,C,H,W]
        x_kv = key/value(ref)[B,C,H,W]
        """
        B, C, H, W = x_q.shape
        assert C == self.dim, f"x_q C channel and  MDTA_dim dismatch"
        assert x_kv.shape == (B,C,H,W), f"x_q and x_kv dismatch"

        qkv = self.qkv(torch.cat([x_q, x_kv], dim = 0)) #"[2B, C, H, W] -> [2B, 3C, H, W] "

        qkv = self.dwconv(qkv) # [2B, 3C, H, W] -> [2B, 3C, H, W] 

        q, k, v = torch.chunk(qkv, 3, dim=1) # [2B, 3C, H, W] -> [2B, C, H, W],[2B, C, H, W],[2B, C, H, W]

        q = q[:B]; k = k[B:]; v = v[B:] #3, [B,C,H,W]


        q = q.reshape(B, self.num_heads, self.head_dim, H*W).transpose(2,3) #[B, num_heads, h*W, head_dim]
        k = k.reshape(B, self.num_heads, self.head_dim, H*W).transpose(2,3) #[B, num_heads, h*W, head_dim]
        v = v.reshape(B, self.num_heads, self.head_dim, H*W).transpose(2,3) #[B, num_heads, h*W, head_dim]



        attn = torch.einsum("bhnd, bhmd->bhnm", q, k) * self.temperature

        attn = F.softmax(attn, dim=-1)


        out = torch.einsum("bhnm, bhmd->bhnd", attn, v)

        out = out.transpose(2,3).reshape(B, self.dim, H, W)


        return self.project_out(out)

        
class WindowShiftedMDTA(nn.Module):
    """专为Decoder设计的分块移位注意力模块（降低内存消耗）"""
    def __init__(self, dim, num_heads=4, window_size=8, shift_size=4, bias=False):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = [window_size, window_size]  # 窗口尺寸 (Win_H, Win_W)
        self.shift_size = [shift_size, shift_size]    # 窗口移位步长 (Shift_H, Shift_W)
        self.head_dim = dim // num_heads
        assert self.head_dim * num_heads == dim, "dim必须能被num_heads整除"
        assert window_size >= shift_size, "窗口尺寸需大于等于移位步长"

        # 可学习的温度参数（控制注意力尖锐度）
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        # QKV投影（1x1卷积，保持通道数不变）
        self.q = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.k = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.v = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

        # 窗口移位的掩码（用于消除移位后的边界重叠）
        self.register_buffer("shift_mask", self._create_shift_mask())

    def _create_shift_mask(self):
        """生成窗口移位的掩码（区分原始窗口和移位窗口的重叠区域）"""
        Win_H, Win_W = self.window_size
        Shift_H, Shift_W = self.shift_size

        # 原始窗口索引（无移位）
        img_mask = torch.zeros((1, 1, Win_H, Win_W))  # [1,1,Win_H,Win_W]
        h_slices = (slice(0, -Shift_H),
                    slice(-Shift_H, None))
        w_slices = (slice(0, -Shift_W),
                    slice(-Shift_W, None))
        cnt = 0
        for h in h_slices:
            for w in w_slices:
                img_mask[:, :, h, w] = cnt
                cnt += 1

        # 移位后的窗口索引（重叠部分标记为-1）
        shifted_img_mask = torch.roll(img_mask, shifts=(-Shift_H, -Shift_W), dims=(2, 3))
        overlap_mask = (img_mask != shifted_img_mask).float()  # 重叠区域为1，非重叠为0
        return overlap_mask

    def window_partition(self, x):
        """将特征图划分为多个窗口（B,C,H,W）→ (B*num_windows, C, Win_H, Win_W)"""
        B, C, H, W = x.shape
        win_H, win_W = self.window_size
        assert H % win_H ==0 and W % win_W==0, f"can not division"
        x = x.view(B, C, H // win_H, win_H, W//win_W, win_W)
        x = x.permute(0, 2, 4, 1, 3, 5).contiguous()  # [B, H', W', C, Win_H, Win_W]
        x = x.view(-1, C, win_H, win_W)  # [B*H'*W', C, Win_H, Win_W]
        return x

    def window_merge(self, x, H, W):
        """将窗口特征合并回原特征图（B*num_windows, C, Win_H, Win_W）→ (B,C,H,W)"""
        Win_H, Win_W = self.window_size
        B = x.shape[0] // ((H // Win_H) * (W // Win_W))
        x = x.view(B, H//Win_H, W//Win_W, self.dim, Win_H, Win_W)
        x = x.permute(0, 3, 1, 4, 2, 5).contiguous()  # [B, C, H, Win_H, W, Win_W]
        x = x.view(B, self.dim, H, W)  # [B, C, H, W]
        return x
    def _window_attention(self, q, k, v):
        """窗口内自注意力计算（带温度参数）"""
        B_win, C, Win_H, Win_W = q.shape
        num_heads = self.num_heads
        head_dim = self.head_dim
        N = Win_H * Win_W

        # Reshape为多头形式（B_win, num_heads, Win_H*Win_W, head_dim）
        q = q.reshape(B_win, num_heads, head_dim, N).transpose(2, 3)  # [B_win, num_heads, N, head_dim]（N=Win_H*Win_W）
        k = k.reshape(B_win, num_heads, head_dim, N).transpose(2, 3)  # [B_win, num_heads, N, head_dim]
        v = v.reshape(B_win, num_heads, head_dim, N).transpose(2, 3)  # [B_win, num_heads, N, head_dim]

        # 计算注意力矩阵（N, N）
        attn = torch.einsum("bhnd, bhmd->bhnm", q, k) * self.temperature  # [B_win, num_heads, N, N]
        attn = F.softmax(attn, dim=-1)  # 空间归一化

        # 聚合value（B_win, num_heads, N, head_dim）
        out = torch.einsum("bhnm, bhmd->bhnd", attn, v)
        out = out.transpose(2, 3).reshape(B_win, C, Win_H, Win_W)  # [B_win, C, Win_H, Win_W]

        return out

    def forward(self, x_q, x_kv):
        """
        x_q: [B, C, H, W] 当前特征（query）
        x_kv: [B, C, H, W] 参考特征（key/value）
        输出: [B, C, H, W] 增强后的特征
        """
        B, C, H, W = x_q.shape
        assert C == self.dim, f"x_q通道数({C})与模块dim({self.dim})不匹配！"
        assert x_kv.shape == (B, C, H, W), "x_kv与x_q形状必须一致！"



        # -------------------- 步骤1：窗口划分 --------------------
        Win_H, Win_W = self.window_size
        Shift_H, Shift_W = self.shift_size

        # 划分原始窗口（无移位）
        x_q_win = self.window_partition(x_q)  # [B*H'*W', C, Win_H, Win_W]
        x_kv_win = self.window_partition(x_kv)  # [B*H'*W', C, Win_H, Win_W]



        # -------------------- 步骤2：原始窗口内注意力 --------------------
        # QKV投影（原始窗口）
        q = self.q(x_q_win)
        k = self.k(x_kv_win)
        v = self.v(x_kv_win)

        # 计算窗口内注意力（局部）
        attn_win = self._window_attention(q,k,v)  # [B*H'*W', C, Win_H, Win_W]

        out_win = self.project_out(attn_win)  # [B*H'*W', C, Win_H, Win_W]

        out_win = self.window_merge(out_win, H, W)

        # -------------------- 步骤3：窗口移位+跨窗口注意力 --------------------
        # 移位窗口（偏移Shift_H和Shift_W）
        x_q_shift = torch.roll(x_q, shifts=(-Shift_H, -Shift_W), dims=(2, 3))  # [B, C, H, W]
        x_kv_shift = torch.roll(x_kv, shifts=(-Shift_H, -Shift_W), dims=(2, 3))  # [B, C, H, W]


        # 划分移位后的窗口
        x_q_shift_win = self.window_partition(x_q_shift)  # [B*H'*W', C, Win_H, Win_W]
        x_kv_shift_win = self.window_partition(x_kv_shift)  # [B*H'*W', C, Win_H, Win_W]


        # 计算移位窗口内注意力（跨窗口）
        attn_shift_win = self._window_attention(self.q(x_q_shift_win), self.k(x_kv_shift_win), self.v(x_kv_shift_win))
        out_shift_win = self.project_out(attn_shift_win)  # [B*H'*W', C, Win_H, Win_W]


        # -------------------- 步骤4：合并原始与移位窗口特征 --------------------
        # 恢复窗口到原分辨率
        # 在窗口级别应用 mask
        mask = self.shift_mask.to(x_q.device)  # [1,1,Win_H,Win_W]
        out_shift_win = out_shift_win * (1 - mask)


        # 合并回原图
        out_shift_win = self.window_merge(out_shift_win, H, W)

        out_shift_win = torch.roll(out_shift_win, shifts=(Shift_H, Shift_W), dims=(2, 3))
 

        # -------------------- 步骤4：合并 --------------------
        out_combined = out_win + out_shift_win
        return out_combined

   




    
class SimpleFusion(nn.Module):
    def __init__(self, ch_qp, ch_ref = None):
        super().__init__()
        ch_ref = ch_qp or ch_qp
        self.ch_ref = ch_ref
        self.ch_qp = ch_qp

        if ch_ref != ch_qp:
            self.ref_adapter = nn.Conv2d(ch_ref, ch_qp, 1)
        else:
            self.ref_adapter = nn.Identity()

        self.fusion = nn.Sequential(
            nn.Conv2d(ch_qp*2, ch_qp, 1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(ch_qp, ch_qp, 3, padding=1),
            nn.Sigmoid()
        )

    def forward(self, feat_qp, feat_ref):
        adapted_ref = self.ref_adapter(feat_ref)
        fusion_mask = self.fusion(torch.cat([feat_qp, adapted_ref], dim=1))
        return feat_qp * fusion_mask + adapted_ref * (1-fusion_mask)

    
class MultiScaleEncoder(nn.Module):
    """多尺度特征提取器，支持参考图引导"""
    def __init__(self, in_ch=3, base_ch=32):
        super().__init__()
        # 压缩图分支
        self.conv1_qp = DepthwiseSeparableConv(in_ch, base_ch)
        self.down1_qp = DepthwiseSeparableConv(base_ch, base_ch*2, stride=2)
        self.down2_qp = DepthwiseSeparableConv(base_ch*2, base_ch*4, stride=2)
        
        # 参考图分支
        self.conv1_ref = DepthwiseSeparableConv(in_ch, base_ch)
        self.down1_ref = DepthwiseSeparableConv(base_ch, base_ch*2, stride=2)
        self.down2_ref = DepthwiseSeparableConv(base_ch*2, base_ch*4, stride=2)
        
        self.act = nn.LeakyReLU(0.2, inplace=True)
    
    def forward(self, qp_img, ref_img):
        # Level 1
        f_qp1 = self.act(self.conv1_qp(qp_img))
        f_ref1 = self.act(self.conv1_ref(ref_img))
        # Level 2
        f_qp2 = self.act(self.down1_qp(f_qp1))
        f_ref2 = self.act(self.down1_ref(f_ref1))

        # Level 3
        f_qp3 = self.act(self.down2_qp(f_qp2))
        f_ref3 = self.act(self.down2_ref(f_ref2))
        
        return f_qp1, f_qp2, f_qp3, f_ref1, f_ref2, f_ref3

class ResidualBlock(nn.Module):
    """轻量化残差块"""
    def __init__(self, ch):
        super().__init__()
        self.conv = nn.Sequential(
            DepthwiseSeparableConv(ch, ch),
            nn.LeakyReLU(0.2),
            DepthwiseSeparableConv(ch,ch)
        )
        
    def forward(self, x):
        return x + self.conv(x)

class EnhancedFFTEnhance(nn.Module):
    """优化频域融合 - 解决空洞关键 (完整修复版)"""
    def __init__(self, ch):
        super().__init__()
        # 幅度融合模块
        self.weight = nn.Parameter(torch.tensor(0.5))
    
    def forward(self, qp_feat, ref_feat):
        # FFT变换 (使用正交归一化)
        qp_feat = qp_feat.float()
        ref_feat = ref_feat.float()
        
        qp_fft = torch.fft.fft2(qp_feat, norm="ortho")
        ref_fft = torch.fft.fft2(ref_feat, norm="ortho")

        fused_fft = self.weight * ref_fft + (1-self.weight) * qp_fft

        enhanced = torch.fft.ifft2(fused_fft, norm="ortho").real

        
        # 残差连接 (带可学习缩放)
        return qp_feat + enhanced

# 在残差注入前使用
class MultiScaleDecoder(nn.Module):
    """多尺度特征解码器"""
    def __init__(self, base_ch=128):
        super().__init__()
        # 上采样路径
        self.up1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            DepthwiseSeparableConv(base_ch*4, base_ch*2)
        )
        self.up2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            DepthwiseSeparableConv(base_ch*2, base_ch)
        )

        self.reduce1 = DepthwiseSeparableConv(base_ch * 4, base_ch * 2)
        self.reduce2 = DepthwiseSeparableConv(base_ch * 2, base_ch)
        
        # 特征融合块
        self.ref_fuse1 = SimpleFusion(base_ch*2)
        self.ref_fuse2 = SimpleFusion(base_ch)

        self.res_block1 = ResidualBlock(base_ch * 2)
        self.res_block2 = ResidualBlock(base_ch)

        self.mdta_decoder = WindowShiftedMDTA(dim=base_ch, num_heads=2, window_size=8, shift_size=4)

        self.out_conv = nn.Conv2d(base_ch, 3, kernel_size=3, padding=1)

    def forward(self, x, feat1, feat2, ref1, ref2, input_img=None):
        # 第一次上采样和特征融合
        x = self.up1(x)
        x = torch.cat([x, feat2], dim=1)
        x = self.reduce1(x)
        x = self.ref_fuse1(x, ref2)
        x = self.res_block1(x)

        x = self.up2(x)
        x = torch.cat([x, feat1], dim=1)
        x = self.reduce2(x)
        x = self.ref_fuse2(x, ref1)
        x = self.res_block2(x)

        x = x + self.mdta_decoder(x, ref1)

        out = self.out_conv(x)
        if input_img is not None:
            input_img = F.interpolate(input_img, size=out.shape[2:], mode='bilinear', align_corners=False)
            out = input_img + out
            out = torch.sigmoid(out)
        return out
    

   
class RefGuidedRestorationNet(nn.Module):
    def __init__(self, base_ch=32):
        super().__init__()
        self.encoder = MultiScaleEncoder(base_ch=base_ch)
        self.mdta_fft = global_MDTA(base_ch*4, num_heads=4)
        self.fft = EnhancedFFTEnhance(ch=base_ch*4)
        self.core = nn.Sequential(
            ResidualBlock(base_ch*4),
            ResidualBlock(base_ch*4),
        )
        self.decoder = MultiScaleDecoder(base_ch=base_ch)

    def forward(self, qp_img, ref_img):
        # 多尺度特征提取

        f_qp1, f_qp2, f_qp3, f_ref1, f_ref2, f_ref3 = self.encoder(qp_img, ref_img)

        f_qp3 = f_qp3 + self.mdta_fft(f_qp3, f_ref3)

        fused = self.fft(f_qp3, f_ref3)
        fused = self.core(fused)

        # 解码重建
        out = self.decoder(fused, f_qp1, f_qp2, f_ref1, f_ref2, input_img=qp_img)
        return out


#Discriminator"""FaceRestorationDiscriminator: 接收四路单通道输入 (B×4×H×W)，通过 1×1 → 多层 3×3卷积 + SpectralNorm + LeakyReLU 下采样，最后展平进入 2 层全连接输出 1 个概率。
#该结构类似于 PatchGAN 与 SRGAN 中鉴别器的经典骨干

class LightweightDiscriminator(nn.Module):
    """增强型轻量化判别器 - 改进细节感知能力"""
    def __init__(self, in_ch=6, base_ch=64, return_features=True):
        super().__init__()
        self.return_features = return_features
        
        # 改进的特征提取网络 - 增加深度和宽度
        self.down1 = nn.Sequential(
            DepthwiseSeparableConv(in_ch, base_ch, stride=2, use_spectral_norm=True),  # 128
            nn.LeakyReLU(0.2),
            DepthwiseSeparableConv(base_ch, base_ch, stride=1, use_spectral_norm=True),
            nn.LeakyReLU(0.2)
        )
        
        self.down2 = nn.Sequential(
            DepthwiseSeparableConv(base_ch, base_ch*2, stride=2, use_spectral_norm=True),  # 64
            nn.LeakyReLU(0.2),
            DepthwiseSeparableConv(base_ch*2, base_ch*2, stride=1, use_spectral_norm=True),
            nn.LeakyReLU(0.2)
        )
        
        self.down3 = nn.Sequential(
            DepthwiseSeparableConv(base_ch*2, base_ch*4, stride=2, use_spectral_norm=True),  # 32
            nn.LeakyReLU(0.2),
            DepthwiseSeparableConv(base_ch*4, base_ch*4, stride=1, use_spectral_norm=True),
            nn.LeakyReLU(0.2)
        )
        
        self.down4 = nn.Sequential(
            DepthwiseSeparableConv(base_ch*4, base_ch*8, stride=2, use_spectral_norm=True),  # 16
            nn.LeakyReLU(0.2),
            DepthwiseSeparableConv(base_ch*8, base_ch*8, stride=1, use_spectral_norm=True),
            nn.LeakyReLU(0.2)
        )
        
        self.down5 = nn.Sequential(
            DepthwiseSeparableConv(base_ch*8, base_ch*16, stride=2, use_spectral_norm=True),  # 8
            nn.LeakyReLU(0.2),
            DepthwiseSeparableConv(base_ch*16, base_ch*16, stride=1, use_spectral_norm=True),
            nn.LeakyReLU(0.2)
        )
        
        # 分类头
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            SpectralNorm(nn.Linear(base_ch*16, base_ch*8)),
            nn.LeakyReLU(0.2),
            SpectralNorm(nn.Linear(base_ch*8, 1))
        )
        self.gp_weight = 10.0
    
    def forward(self, qp_img, restored_img):
        # 拼接输入
        x = torch.cat([qp_img, restored_img], dim=1)
        
        # 多尺度特征提取
        feat1 = self.down1(x)   # [B, 64, 256, 256]
        feat2 = self.down2(feat1) # [B, 128, 128, 128]
        feat3 = self.down3(feat2) # [B, 256, 64, 64]
        feat4 = self.down4(feat3) # [B, 512, 32, 32]
        feat5 = self.down5(feat4) # [B, 1024, 16, 16]
        
        # 最终分类
        pred = self.head(feat5)
        
        if self.return_features:
            # 返回多层特征用于特征匹配损失
            features = [feat1, feat2, feat3, feat4, feat5]
            return pred, features
        else:
            return pred
    
    def gradient_penalty(self, qp_img, real_img, fake_img):
        """计算梯度惩罚 (WGAN-GP)"""
        alpha = torch.rand(qp_img.size(0), 1, 1, 1).to(qp_img.device)
        interpolated = alpha * real_img + (1 - alpha) * fake_img
        
        # 计算判别器对插值样本的输出
        disc_interpolated, _ = self(qp_img, interpolated)
        
        # 计算梯度
        grad = torch.autograd.grad(
            outputs=disc_interpolated,
            inputs=interpolated,
            grad_outputs=torch.ones_like(disc_interpolated),
            create_graph=True,
            retain_graph=True,
            only_inputs=True
        )[0]
        
        # 计算梯度惩罚
        grad_norm = grad.view(grad.size(0), -1).norm(2, dim=1)
        gp = torch.mean((grad_norm - 1.0) ** 2)
        
        return gp
        
    

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    comp_face = torch.randn(2,3,512,512,device = device)
    ref_face = torch.randn(2,3,512,512,device = device)
    GT = torch.randn(2,3,512,512,device = device)

    gen = RefGuidedRestorationNet().to(device)

    gen.eval()
    with torch.no_grad():
        restored = gen(comp_face, ref_face)
    print(f"[Generator] restored shape: {restored.shape}")  # 期望 (B,1,256,256)  # 期望 (B,64,256,256)

    # 3. 实例化鉴别器，输入通道=4（comp, ref, hf, restored），base_channels=64
    disc = LightweightDiscriminator(in_ch=6, base_ch = 16, return_features=False).to(device)
    disc.eval()

    # 将 comp_face, ref_face, hf_feat, restored 作为四路输入传给鉴别器
    with torch.no_grad():
        prob = disc(comp_face, restored)

    # 检查鉴别器输出维度
    print(f"[Discriminator] output prob shape: {prob.shape}")  # 期望 (B,1)
