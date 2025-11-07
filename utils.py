import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
import cv2
from torchvision.utils import save_image


class LCC(nn.Module):
    """
    local (over window) normalized cross correlation (square)
    """

    def __init__(self, win=[9, 9], eps=1e-5):
        super(LCC, self).__init__()
        self.win = win
        self.eps = eps

    def forward(self, I, J):
        I2 = I.pow(2)
        J2 = J.pow(2)
        IJ = I * J

        filters = Variable(torch.ones(1, 1, self.win[0], self.win[1]))
        if I.is_cuda:  # gpu
            filters = filters.cuda()
        padding = (self.win[0] // 2, self.win[1] // 2)

        I_sum = F.conv2d(I, filters, stride=1, padding=padding)
        J_sum = F.conv2d(J, filters, stride=1, padding=padding)
        I2_sum = F.conv2d(I2, filters, stride=1, padding=padding)
        J2_sum = F.conv2d(J2, filters, stride=1, padding=padding)
        IJ_sum = F.conv2d(IJ, filters, stride=1, padding=padding)

        win_size = self.win[0] * self.win[1]

        u_I = I_sum / win_size
        u_J = J_sum / win_size

        cross = IJ_sum - u_J * I_sum - u_I * J_sum + u_I * u_J * win_size
        I_var = I2_sum - 2 * u_I * I_sum + u_I * u_I * win_size
        J_var = J2_sum - 2 * u_J * J_sum + u_J * u_J * win_size

        cc = cross * cross / (I_var * J_var + self.eps)  # np.finfo(float).eps
        lcc = -1.0 * torch.mean(cc) + 1
        return lcc


class SSIM(nn.Module):
    """Layer to compute the weighted SSIM and L1 loss between a pair of images"""
    def __init__(self, ssim_weight=0.85):
        super().__init__()
        self.a = ssim_weight
        self.b = 1 - ssim_weight

        self.pool = nn.AvgPool2d(3, 1)
        self.refl = nn.ReflectionPad2d(1)

        self.C1 = 0.01 ** 2
        self.C2 = 0.03 ** 2

    def forward(self, pred, target):
        l1_loss = torch.abs(target - pred).mean(1, keepdim=True)

        pred, target = self.refl(pred), self.refl(target)
        mu_pred, mu_target = self.pool(pred), self.pool(target)

        sigma_pred = self.pool(pred**2) - mu_pred**2
        sigma_target = self.pool(target**2) - mu_target**2
        sigma_pt = self.pool(pred*target) - mu_pred*mu_target

        ssim_n = (2 * mu_pred * mu_target + self.C1) * (2 * sigma_pt + self.C2)
        ssim_d = (mu_pred**2 + mu_target**2 + self.C1) * (sigma_pred + sigma_target + self.C2)

        sim = torch.clamp((1 - ssim_n/ssim_d) / 2, min=0, max=1)
        sim = sim.mean(1, keepdim=True)

        loss = self.a*sim + self.b*l1_loss
        return loss

def compute_loss(fusion, img_cat, img_dark, img_rgb,img_nir, put_type='right', balance=0.01):

    # lossfunc = nn.MSELoss()
    # loss1 = lossfunc(fusion,img_rgb)
    loss1 = torch.norm(fusion - img_rgb, 2)
    # loss2_1 = structure_loss(fusion, img_nir, img_dark,usedark = True).cuda()
    # loss2_2 = structure_loss(fusion, img_2, img_dark,usedark = False).cuda()
    # loss2 = loss2_2*0.4 +loss2_1 * 0.6
    loss2 = structure_loss(fusion, img_cat,img_dark,usedark = False).cuda()
    regis = SSIM()
    l_ssim = regis(fusion,img_rgb)
    index = l_ssim.argmin(dim=1, keepdim=True)
    loss3 = l_ssim.gather(dim=1, index=index).mean()
    # loss3 = Cosineloss(fusion[:,1:3,:,:], img_2[:,1:3,:,:])
    return loss1, balance * loss2, 0.9*loss3
class Gradient_Net(nn.Module):
  def __init__(self):
    super(Gradient_Net, self).__init__()
    kernel_x = [[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]]
    kernel_x = torch.FloatTensor(kernel_x).unsqueeze(0).unsqueeze(0).cuda()

    kernel_y = [[-1., -2., -1.], [0., 0., 0.], [1., 2., 1.]]
    kernel_y = torch.FloatTensor(kernel_y).unsqueeze(0).unsqueeze(0).cuda()

    self.weight_x = nn.Parameter(data=kernel_x, requires_grad=False)
    self.weight_y = nn.Parameter(data=kernel_y, requires_grad=False)

  def forward(self, x):
    grad_x = F.conv2d(x, self.weight_x)
    grad_y = F.conv2d(x, self.weight_y)
    gradient = torch.abs(grad_x) + torch.abs(grad_y)
    return gradient

def compute_warp_loss(img_rgb,img_nir,pred):
    regis = SSIM()
    l_ssim = regis(pred,img_rgb)
    index = l_ssim.argmin(dim=1, keepdim=True)
    loss1 = l_ssim.gather(dim=1, index=index).mean()
    regis2 = LCC()
    loss2 = 1-regis2(img_rgb,pred)
    Grad = Gradient_Net()
    # print(pred.shape)
    g_y = Grad(img_rgb)
    g_nir = Grad(pred)
    # y_tset = g_y[1,:,:,:]
    # nir_test = g_nir[1,:,:,:]
    # save_image(y_tset.cpu().unsqueeze(0), 'grad_y.png')
    # save_image(nir_test.cpu().unsqueeze(0), 'grad_nir.png')

    # print(g_nir.shape)
    # print(g_y.shape)
    loss3 = torch.norm(g_nir-g_y)
    return loss1,loss2,0.5*loss3



def Cosineloss(pred,gt):
    func = nn.CosineEmbeddingLoss()
    target = torch.tensor([[[[1]]]], dtype=torch.float).cuda()
    value =func(pred,gt,target)
    return value
def create_putative(in1, in2, put_type):

    if put_type == 'mean':
        iput = (in1 + in2) / 2
    elif put_type == 'left':
        iput = in1
    elif put_type == 'right':
        iput = in2
    else:
        raise EOFError('No supported type!')

    return iput

def intensity_loss(fusion, img_1, img_2, put_type):

    inp = create_putative(img_1, img_2, put_type)

    # L2 norm
    loss = torch.norm(fusion - inp, 2)

    return loss

def gradient(x):

    H, W = x.shape[2], x.shape[3]

    left = x
    right = F.pad(x, [0, 1, 0, 0])[:, :, :, 1:]
    top = x
    bottom = F.pad(x, [0, 0, 0, 1])[:, :, 1:, :]

    dx, dy = right - left, bottom - top 

    dx[:, :, :, -1] = 0
    dy[:, :, -1, :] = 0
    return dx, dy


def create_structure(inputs):

    B, C, H, W = inputs.shape[0], inputs.shape[1], inputs.shape[2], inputs.shape[3]

    dx, dy = gradient(inputs)

    structure = torch.zeros(B, 4, H, W) # Structure tensor = 2 * 2 matrix

    a_00 = dx.pow(2)
    a_01 = a_10 = dx * dy
    a_11 = dy.pow(2)

    structure[:,0,:,:] = torch.sum(a_00,dim=1)
    structure[:,1,:,:] = torch.sum(a_01,dim=1)
    structure[:,2,:,:] = torch.sum(a_10,dim=1)
    structure[:,3,:,:] = torch.sum(a_11,dim=1)

    return structure.cuda()

def structure_loss(fusion, img_cat, dark,usedark =False):
    st_fusion = create_structure(fusion)
    st_input = create_structure(img_cat)
    B, C, H, W = st_fusion.shape[0], st_fusion.shape[1], st_fusion.shape[2], st_fusion.shape[3]
    # dark = dark.expand([B,C,H,W])
    # E = torch.ones([B,C,H,W]).cuda()
    # dark = dark + E
    # dark = torch.pow(dark,2)

    # Frobenius norm
    if usedark:
        dark = dark.expand([B, C, H, W])
        # E = torch.ones([B, C, H, W]).cuda()
        # dark = dark + E
        dark = torch.pow(dark, 2)
        loss = torch.norm(torch.mul(dark, (st_fusion - st_input)))
    else:
        loss = torch.norm(st_fusion - st_input)
    return loss

def downsample(img1, img2, maxSize = 256):
    _,channels,H,W = img1.shape
    #channels,H,W = img1.shape
    f = int(max(1,np.round(min(H,W)/maxSize)))
    if f>1:
        aveKernel = (torch.ones(channels,1,f,f)/f**2).to(img1.device)
        img1 = F.conv2d(img1, aveKernel, stride=f, padding = 0, groups = channels)
        img2 = F.conv2d(img2, aveKernel, stride=f, padding = 0, groups = channels)
    return img1, img2


if __name__ == "__main__":
    fusion = torch.rand(5,1,4,4).cuda()
    img_1 = torch.rand(5,1,4,4).cuda()
    img_2 = torch.rand(5,1,4,4).cuda()
    dark = torch.rand(5,1,4,4).cuda()
    img_cat = torch.cat([img_1,img_2],dim=1)

    loss = compute_loss(fusion, img_cat, dark, img_2,img_1, put_type='right', balance=0.0001)
    print(loss)
