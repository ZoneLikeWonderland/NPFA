from torch.nn.utils.parametrizations import spectral_norm
import torch.nn as nn
import torch
import torch.nn.functional as F
from pytorch_msssim import MS_SSIM, SSIM
import sys
import os
import torchvision
sys.path.append(os.path.dirname(__file__))
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
if True:
    from utils import *
    from parts import *


class ContentLoss(nn.Module):
    def __init__(self, l1=1, mse=1, ssim=2, channel=3):
        super().__init__()
        self.criterion_L1 = nn.L1Loss()
        self.criterion_MSE = nn.MSELoss()
        # self.criterion_SSIM = MS_SSIM(1, channel=channel)
        self.criterion_SSIM = SSIM(1, channel=channel)

        self.l1 = l1
        self.mse = mse
        self.ssim = ssim

    def forward(self, x, y, scale_x=False):
        if scale_x:
            y_mean = torch.mean(y, dim=(2, 3), keepdim=True).detach()
            x_mean = torch.mean(x, dim=(2, 3), keepdim=True).detach()
            x = x / x_mean * y_mean
        loss_L1 = self.criterion_L1(x, y)
        loss_MSE = self.criterion_MSE(x, y)
        loss_SSIM = 1 - self.criterion_SSIM(x, y)
        return self.l1 * loss_L1 + self.mse * loss_MSE + self.ssim * loss_SSIM


class SimilarityLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, y, dim=1):
        return 1 - torch.mean(F.cosine_similarity(x, y, dim=dim))


class LoadableNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.pn = PixelNorm()
        self.eps = None

    def load(self, path):
        try:
            saving = torch.load(path, map_location=torch.device("cpu"))
            is_changed = self.on_load_checkpoint(saving)
            self.load_state_dict(
                saving,
                False
            )
        except Exception as e:
            print(f"[ WARNING ] ignore loading @ due to {e}")
            return False
        return not is_changed

    def on_load_checkpoint(self, state_dict: dict) -> None:
        model_state_dict = self.state_dict()
        is_changed = False
        for k in state_dict:
            if k in model_state_dict:
                if state_dict[k].shape != model_state_dict[k].shape:
                    print(f"Skip loading parameter: {k}, "
                          f"required shape: {model_state_dict[k].shape}, "
                          f"loaded shape: {state_dict[k].shape}")
                    state_dict_mix = model_state_dict[k].clone()

                    if len(state_dict_mix.shape) == len(state_dict[k].shape):
                        ss = []
                        for i in range(len(state_dict[k].shape)):
                            ss.append(slice(0, min(state_dict[k].shape[i], state_dict_mix.shape[i])))
                        state_dict_mix[ss] = state_dict[k][ss]
                        print("weight copied")

                    state_dict[k] = state_dict_mix
                    is_changed = True
            else:
                print(f"Dropping parameter {k}")
                is_changed = True

        if is_changed:
            model_state_dict.pop("optimizer_states", None)
        return is_changed

    def set_requires_grad(self, requires_grad):
        for param in self.parameters():
            param.requires_grad = requires_grad

    def set_other_requires_grad(self, net, requires_grad):
        for param in net.parameters():
            param.requires_grad = requires_grad

    def set_params_requires_grad(self, params, requires_grad):
        for param in params:
            param.requires_grad = requires_grad

    def sample(self, para_encode, ch=None, sigma_coeff=1):
        if ch is None:
            ch = self.ae_ch

        mu = para_encode[:, :ch]
        log_sigma = para_encode[:, ch:]
        # std = torch.exp(sigma_coeff * log_sigma)
        std = (F.celu(sigma_coeff * log_sigma - 10, alpha=1) + 1) * np.exp(10)
        # std = torch.minimum(std, log_sigma * 1e2)

        kld = torch.mean(-log_sigma + mu**2 + std)

        if self.eps is not None:
            if self.eps is True:
                self.eps = torch.randn_like(mu)
            eps = self.eps
        else:
            eps = torch.randn_like(mu)

        latent = eps * std + mu
        assert torch.isfinite(std).all(), std[~torch.isfinite(std)]
        assert torch.isfinite(mu).all(), mu

        return latent, kld


class ImageEncoder(nn.Module):
    def __init__(self, input_channel, ch, ae_ch, inter_layers=1):
        super().__init__()
        self.to128 = ConvBlock_7_2(input_channel, ch * 1, False)
        self.to64 = ConvBlock_5_2(ch * 1, ch * 2, False)
        self.to32 = ConvBlock_5_2(ch * 2, ch * 4, False)
        self.to16 = ConvBlock_5_2(ch * 4, ch * 8, False)
        self.to8 = ConvBlock_5_2(ch * 8, ch * 16, False)
        self.to4 = ConvBlock_5_2(ch * 16, ch * 32, False)
        self.fc = nn.Sequential(
            FullyConnect(ch * 32 * 4**2, ch * 32),
            *sum([
                [nn.LeakyReLU(0.1, inplace=True),
                 FullyConnect(ch * 32, ch * 32)]
                for i in range(inter_layers)
            ], []),
            nn.LeakyReLU(0.1, inplace=True),
            FullyConnect(ch * 32, ae_ch),
        )

    def forward(self, x):
        x = self.to128(x)
        x = self.to64(x)
        x = self.to32(x)
        x = self.to16(x)
        x = self.to8(x)
        x = self.to4(x)
        x = self.fc(x)
        return x


class ImageEncoder2(nn.Module):
    def __init__(self, input_channel, ch, ae_ch, inter_layers=1):
        super().__init__()
        self.to128 = ConvBlock_7_2(input_channel, ch * 1, False)
        self.to64 = ConvBlock_5_2(ch * 1, ch * 2, False)
        self.to32 = ConvBlock_5_2(ch * 2, ch * 4, False)
        self.to16 = ConvBlock_5_2(ch * 4, ch * 8, False)
        self.to8 = ConvBlock_5_2(ch * 8, ch * 16, False)
        self.to4 = ConvBlock_5_2(ch * 16, ch * 32, False)
        self.fc = nn.Sequential(
            FullyConnect(ch * 32 * 4**2, ch * 32 * 2),
            *sum([
                [nn.LeakyReLU(0.1, inplace=True),
                 FullyConnect(ch * 32 * 2, ch * 32 * 2)]
                for i in range(inter_layers)
            ], []),
            nn.LeakyReLU(0.1, inplace=True),
            FullyConnect(ch * 32 * 2, ae_ch),
        )

    def forward(self, x):
        x = self.to128(x)
        x = self.to64(x)
        x = self.to32(x)
        x = self.to16(x)
        x = self.to8(x)
        x = self.to4(x)
        x = self.fc(x)
        return x


class ImageDecoder(nn.Module):
    def __init__(self, input_channel, ae_ch, ch):
        super().__init__()
        self.to16 = UpscaleBlock(ae_ch + 2, ch * 16)
        self.to32 = UpscaleBlock(ch * 16 + 2, ch * 16)
        self.to64 = UpscaleBlock(ch * 16 + 2, ch * 8)
        self.to128 = UpscaleBlock(ch * 8 + 2, ch * 4)
        self.to256 = UpscaleBlock(ch * 4 + 2, ch * 2)
        self.final = nn.Sequential(
            nn.Conv2d(ch * 2, ch * 1, 1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(ch * 1, input_channel, 1),
            nn.LeakyReLU(0.1, inplace=True),
        )

    def forward(self, x, indicator=0, indicator2=0):
        x = torch.cat([x, torch.ones((x.shape[0], 1, x.shape[2], x.shape[3]), dtype=torch.float32, device=x.device) * indicator], dim=1)
        x = torch.cat([x, torch.ones((x.shape[0], 1, x.shape[2], x.shape[3]), dtype=torch.float32, device=x.device) * indicator2], dim=1)
        x = self.to16(x)
        x = torch.cat([x, torch.ones((x.shape[0], 1, x.shape[2], x.shape[3]), dtype=torch.float32, device=x.device) * indicator], dim=1)
        x = torch.cat([x, torch.ones((x.shape[0], 1, x.shape[2], x.shape[3]), dtype=torch.float32, device=x.device) * indicator2], dim=1)
        x = self.to32(x)
        x = torch.cat([x, torch.ones((x.shape[0], 1, x.shape[2], x.shape[3]), dtype=torch.float32, device=x.device) * indicator], dim=1)
        x = torch.cat([x, torch.ones((x.shape[0], 1, x.shape[2], x.shape[3]), dtype=torch.float32, device=x.device) * indicator2], dim=1)
        x = self.to64(x)
        x = torch.cat([x, torch.ones((x.shape[0], 1, x.shape[2], x.shape[3]), dtype=torch.float32, device=x.device) * indicator], dim=1)
        x = torch.cat([x, torch.ones((x.shape[0], 1, x.shape[2], x.shape[3]), dtype=torch.float32, device=x.device) * indicator2], dim=1)
        x = self.to128(x)
        x = torch.cat([x, torch.ones((x.shape[0], 1, x.shape[2], x.shape[3]), dtype=torch.float32, device=x.device) * indicator], dim=1)
        x = torch.cat([x, torch.ones((x.shape[0], 1, x.shape[2], x.shape[3]), dtype=torch.float32, device=x.device) * indicator2], dim=1)
        x = self.to256(x)
        x = self.final(x)
        return x


class MeshDecoder(nn.Module):
    def __init__(self, input_channel, ae_ch, ch):
        super().__init__()
        self.to16 = UpscaleBlock(ae_ch, ch * 16)
        self.to32 = UpscaleBlock(ch * 16, ch * 16)
        self.to64 = UpscaleBlock(ch * 16, ch * 8)
        self.to128 = UpscaleBlock(ch * 8, ch * 4)
        self.to256 = UpscaleBlock(ch * 4, ch * 2)
        self.final = nn.Sequential(
            nn.Conv2d(ch * 2, ch * 1, 3, padding=1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(ch * 1, input_channel, 3, padding=1),
            # nn.LeakyReLU(0.1, inplace=True),
        )
        # self.act = nn.LeakyReLU(0.1, inplace=True) if act else None

    def forward(self, x, indicator=0):
        x = self.to16(x)
        x = self.to32(x)
        x = self.to64(x)
        x = self.to128(x)
        x = self.to256(x)
        x = self.final(x)
        # if self.act is not None:
        #     x = self.act(x)
        return x


class TexDecoder(nn.Module):
    def __init__(self, input_channel, ae_ch, ch):
        super().__init__()
        self.to16 = UpscaleBlock(ae_ch, ch * 16)
        self.to32 = UpscaleBlock(ch * 16, ch * 16)
        self.to64 = UpscaleBlock(ch * 16, ch * 8)
        self.to128 = UpscaleBlock(ch * 8, ch * 4)
        self.to256 = UpscaleBlock(ch * 4, ch * 2)
        self.to512 = UpscaleBlock(ch * 2, ch * 1)
        self.final = nn.Sequential(
            nn.Conv2d(ch * 1, ch * 1, 3, padding=1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(ch * 1, input_channel, 3, padding=1),
        )

    def forward(self, x, indicator=0):
        x = self.to16(x)
        x = self.to32(x)
        x = self.to64(x)
        x = self.to128(x)
        x = self.to256(x)
        x = self.to512(x)
        x = self.final(x)
        return x


class MLP(nn.Module):
    def __init__(self, input_dim, output_dim, dim, n_blk, norm='none', activ='relu'):

        super(MLP, self).__init__()
        self.model = []
        self.model += [LinearBlock(input_dim, dim, norm=norm, activation=activ)]
        for i in range(n_blk - 2):
            self.model += [LinearBlock(dim, dim, norm=norm, activation=activ)]
        self.model += [LinearBlock(dim, output_dim, norm='none', activation='none')]  # no output activations
        self.model = nn.Sequential(*self.model)

    def forward(self, x):
        return self.model(x.view(x.size(0), -1))


class ExpNet(LoadableNet):
    def __init__(self, e_ch=32, ae_ch=256, d_ch=32, trans_ch=256, lambda_=1, unet_ch=32):
        super().__init__()

        self.face_encoder = ImageEncoder(3, e_ch, ae_ch * 2 + trans_ch * 2)
        self.mesh_decoder = nn.Sequential(
            PixelNorm(),
            FullyConnect(ae_ch, ae_ch * 4**2, shape=(-1, ae_ch, 4, 4)),
            UpscaleBlock(ae_ch, ae_ch),
            MeshDecoder(6, ae_ch, d_ch)
        )
        self.view2aug = nn.Sequential(
            PixelNorm(),
            nn.Linear(trans_ch, 256),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Linear(256, 256),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Linear(256, 256),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Linear(256, 64),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Linear(64, 10)
        )

        self.tex_decoder2 = nn.Sequential(
            PixelNorm(),
            FullyConnect(ae_ch, ae_ch * 4**2, shape=(-1, ae_ch, 4, 4)),
            UpscaleBlock(ae_ch, ae_ch),
            TexDecoder(7, ae_ch, d_ch)
        )

        self.ae_ch = ae_ch
        self.trans_ch = trans_ch

    def face_encode(self, x):
        para_encode = self.face_encoder(x)
        assert torch.isfinite(para_encode).all(), para_encode
        face_latent, kld = self.sample(para_encode, self.ae_ch + self.trans_ch)
        return face_latent, kld


criterion_content = ContentLoss()
criterion_content7 = ContentLoss(channel=7)
criterion_l2 = nn.MSELoss()
criterion_latent = SimilarityLoss()