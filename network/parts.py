import torch
import torch.nn as nn
from torch.autograd import Function
import torch.nn.functional as F
from torch.nn.utils.parametrizations import spectral_norm


class PixelNorm(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x / torch.sqrt(torch.mean(x ** 2, dim=1, keepdim=True) + 1e-8)


class PixelNormSplit(nn.Module):
    def __init__(self, splits):
        super().__init__()
        self.splits = splits
        self.pn = PixelNorm()

    def forward(self, x):
        xs = []
        last = 0
        for step in self.splits:
            xs.append(self.pn(x[:, last:last + step]))
            last += step

        x = torch.cat(xs, dim=1)
        return x


class ConvConvBlock_3_1(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.LeakyReLU(0.1, inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


class ConvConvBlock_3_1_sn(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            spectral_norm(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)),
            nn.LeakyReLU(0.1, inplace=True),
            spectral_norm(nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)),
            nn.LeakyReLU(0.1, inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


class ConvBlock_3_1(nn.Module):
    def __init__(self, in_channels, out_channels, gn):
        super().__init__()
        self.conv = nn.Sequential()
        self.conv.add_module("Conv2d", nn.Conv2d(in_channels, out_channels, kernel_size=3))
        self.conv.add_module("LeakyReLU", nn.LeakyReLU(0.1, inplace=True))
        if gn:
            self.conv.add_module("GroupNorm", nn.GroupNorm(4, out_channels))

    def forward(self, x):
        return self.conv(x)

class ConvBlock_4_0(nn.Module):
    def __init__(self, in_channels, out_channels, gn):
        super().__init__()
        self.conv = nn.Sequential()
        self.conv.add_module("Conv2d", nn.Conv2d(in_channels, out_channels, kernel_size=4))
        self.conv.add_module("LeakyReLU", nn.LeakyReLU(0.1, inplace=True))
        if gn:
            self.conv.add_module("GroupNorm", nn.GroupNorm(4, out_channels))

    def forward(self, x):
        return self.conv(x)

class ConvBlock_5_2(nn.Module):
    def __init__(self, in_channels, out_channels, gn):
        super().__init__()
        self.conv = nn.Sequential()
        self.conv.add_module("Conv2d", nn.Conv2d(in_channels, out_channels, kernel_size=5, stride=2, padding=2))
        self.conv.add_module("LeakyReLU", nn.LeakyReLU(0.1, inplace=True))
        if gn:
            self.conv.add_module("GroupNorm", nn.GroupNorm(4, out_channels))

    def forward(self, x):
        return self.conv(x)


class ConvBlock_7_2(nn.Module):
    def __init__(self, in_channels, out_channels, gn):
        super().__init__()
        self.conv = nn.Sequential()
        self.conv.add_module("Conv2d", nn.Conv2d(in_channels, out_channels, kernel_size=7, stride=2, padding=3))
        self.conv.add_module("LeakyReLU", nn.LeakyReLU(0.1, inplace=True))
        if gn:
            self.conv.add_module("GroupNorm", nn.GroupNorm(4, out_channels))

    def forward(self, x):
        return self.conv(x)


class FullyConnect(nn.Module):
    def __init__(self, in_size, out_size, shape=None, sn=False):
        super(FullyConnect, self).__init__()
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_size, out_size) if not sn else spectral_norm(nn.Linear(in_size, out_size))
        )
        self.shape = shape

    def forward(self, x):
        x = self.fc(x)
        if self.shape is not None:
            x = x.reshape(self.shape)
        return x


class UpscaleBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.UpConv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels * 4, 3, 1, padding=1, bias=False),
            nn.PixelShuffle(2),
            nn.LeakyReLU(0.1, inplace=True),
        )
        self.after = ResidualBlock(out_channels)

    def forward(self, x):
        x = self.UpConv(x)
        x = self.after(x)
        return x


class UpConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.UpConv = nn.Sequential(
            nn.ConvTranspose2d(in_channels, in_channels, kernel_size=2, stride=2),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(in_channels, out_channels, 3, 1, padding=1),
            nn.LeakyReLU(0.1, inplace=True),
        )
        self.after = ResidualBlock(out_channels)

    def forward(self, x):
        x = self.UpConv(x)
        x = self.after(x)
        return x


class UpscaleResizeBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.UpConv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels * 4, 3, 1, padding=1, bias=False),
            nn.PixelShuffle(2),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(in_channels, out_channels, 3, 1, padding=1),
            nn.LeakyReLU(0.1, inplace=True),
        )
        self.after = ResidualBlock(out_channels)

    def forward(self, x):
        x = self.UpConv(x)
        x = self.after(x)
        return x


class ResidualBlock(nn.Module):
    def __init__(self, in_ch, kernel_size=3):
        super().__init__()
        self.in_ch = in_ch
        self.kernel_size = kernel_size
        self._build_net(in_ch, kernel_size)

    def _build_net(self, in_ch, kernel_size):
        self.conv1 = nn.Conv2d(in_ch, in_ch, kernel_size=kernel_size, padding=1)
        self.conv1_relu = nn.LeakyReLU(0.2, inplace=True)
        self.conv2 = nn.Conv2d(in_ch, in_ch, kernel_size=kernel_size, padding=1)
        self.conv2_relu = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        x1 = self.conv1(x)
        x1_relu = self.conv1_relu(x1)
        x2 = self.conv2(x1_relu)
        x2_relu = self.conv2_relu(x2 + x)
        return x2_relu


class ParaBlock(nn.Module):
    def __init__(self, concat_channels, shape):
        super().__init__()
        self.para = nn.Parameter(torch.randn((1, concat_channels, shape[0], shape[1])))

    def forward(self, x):
        para = self.para.expand(x.shape[0], self.para.shape[1], self.para.shape[2], self.para.shape[3])
        x = torch.cat([x, para], dim=1)
        return x


class GradientReversalFunction(Function):
    """
    Gradient Reversal Layer from:
    Unsupervised Domain Adaptation by Backpropagation (Ganin & Lempitsky, 2015)

    Forward pass is the identity function. In the backward pass,
    the upstream gradients are multiplied by -lambda (i.e. gradient is reversed)
    """

    @staticmethod
    def forward(ctx, x, lambda_):
        ctx.lambda_ = lambda_
        return x.clone()

    @staticmethod
    def backward(ctx, grads):
        lambda_ = ctx.lambda_
        lambda_ = grads.new_tensor(lambda_)
        dx = -lambda_ * grads
        return dx, None


class GradientReversal(torch.nn.Module):
    def __init__(self, lambda_=1):
        super(GradientReversal, self).__init__()
        self.lambda_ = lambda_

    def forward(self, x):
        return GradientReversalFunction.apply(x, self.lambda_)


class Conv2dBlock(nn.Module):
    def __init__(self, input_dim, output_dim, kernel_size, stride,
                 padding=0, norm='none', activation='relu', pad_type='zero'):
        super(Conv2dBlock, self).__init__()
        self.use_bias = True
        # initialize padding
        if pad_type == 'reflect':
            self.pad = nn.ReflectionPad2d(padding)
        elif pad_type == 'replicate':
            self.pad = nn.ReplicationPad2d(padding)
        elif pad_type == 'zero':
            self.pad = nn.ZeroPad2d(padding)
        else:
            assert 0, "Unsupported padding type: {}".format(pad_type)

        # initialize normalization
        norm_dim = output_dim
        if norm == 'bn':
            self.norm = nn.BatchNorm2d(norm_dim)
        elif norm == 'in':
            #self.norm = nn.InstanceNorm2d(norm_dim, track_running_stats=True)
            self.norm = nn.InstanceNorm2d(norm_dim)
        elif norm == 'ln':
            self.norm = LayerNorm(norm_dim)
        elif norm == 'adain':
            self.norm = AdaptiveInstanceNorm2d(norm_dim)
        elif norm == 'none' or norm == 'sn':
            self.norm = None
        else:
            assert 0, "Unsupported normalization: {}".format(norm)

        # initialize activation
        if activation == 'relu':
            self.activation = nn.ReLU(inplace=True)
        elif activation == 'lrelu':
            self.activation = nn.LeakyReLU(0.2, inplace=True)
        elif activation == 'prelu':
            self.activation = nn.PReLU()
        elif activation == 'selu':
            self.activation = nn.SELU(inplace=True)
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'none':
            self.activation = None
        else:
            assert 0, "Unsupported activation: {}".format(activation)

        # initialize convolution
        if norm == 'sn':
            self.conv = nn.SpectralNorm(nn.Conv2d(input_dim, output_dim, kernel_size, stride, bias=self.use_bias))
        else:
            self.conv = nn.Conv2d(input_dim, output_dim, kernel_size, stride, bias=self.use_bias)

    def forward(self, x):
        x = self.conv(self.pad(x))
        if self.norm:
            x = self.norm(x)
        if self.activation:
            x = self.activation(x)
        return x


class LinearBlock(nn.Module):
    def __init__(self, input_dim, output_dim, norm='none', activation='relu'):
        super(LinearBlock, self).__init__()
        use_bias = True
        # initialize fully connected layer
        if norm == 'sn':
            self.fc = SpectralNorm(nn.Linear(input_dim, output_dim, bias=use_bias))
        else:
            self.fc = nn.Linear(input_dim, output_dim, bias=use_bias)

        # initialize normalization
        norm_dim = output_dim
        if norm == 'bn':
            self.norm = nn.BatchNorm1d(norm_dim)
        elif norm == 'in':
            self.norm = nn.InstanceNorm1d(norm_dim)
        elif norm == 'ln':
            self.norm = LayerNorm(norm_dim)
        elif norm == 'none' or norm == 'sn':
            self.norm = None
        else:
            assert 0, "Unsupported normalization: {}".format(norm)

        # initialize activation
        if activation == 'relu':
            self.activation = nn.ReLU(inplace=True)
        elif activation == 'lrelu':
            self.activation = nn.LeakyReLU(0.2, inplace=True)
        elif activation == 'prelu':
            self.activation = nn.PReLU()
        elif activation == 'selu':
            self.activation = nn.SELU(inplace=True)
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'none':
            self.activation = None
        else:
            assert 0, "Unsupported activation: {}".format(activation)

    def forward(self, x):
        out = self.fc(x)
        if self.norm:
            out = self.norm(out)
        if self.activation:
            out = self.activation(out)
        return out


class ResBlock(nn.Module):
    def __init__(self, dim, norm='in', activation='relu', pad_type='zero'):
        super(ResBlock, self).__init__()

        model = []
        model += [Conv2dBlock(dim, dim, 3, 1, 1, norm=norm, activation=activation, pad_type=pad_type)]
        model += [Conv2dBlock(dim, dim, 3, 1, 1, norm=norm, activation='none', pad_type=pad_type)]
        self.model = nn.Sequential(*model)

    def forward(self, x):
        residual = x
        out = self.model(x)
        out += residual
        return out


class ResBlocks(nn.Module):
    def __init__(self, num_blocks, dim, norm='in', activation='relu', pad_type='zero'):
        super(ResBlocks, self).__init__()
        self.model = []
        for i in range(num_blocks):
            self.model += [ResBlock(dim, norm=norm, activation=activation, pad_type=pad_type)]
        self.model = nn.Sequential(*self.model)

    def forward(self, x):
        return self.model(x)


class AdaptiveInstanceNorm2d(nn.Module):
    def __init__(self, num_features, eps=1e-5, momentum=0.1):
        super(AdaptiveInstanceNorm2d, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        # weight and bias are dynamically assigned
        self.weight = None
        self.bias = None
        # just dummy buffers, not used
        self.register_buffer('running_mean', torch.zeros(num_features))
        self.register_buffer('running_var', torch.ones(num_features))

    def forward(self, x):
        assert self.weight is not None and self.bias is not None, "Please assign weight and bias before calling AdaIN!"
        b, c = x.size(0), x.size(1)
        running_mean = self.running_mean.repeat(b)
        running_var = self.running_var.repeat(b)

        # Apply instance norm
        x_reshaped = x.contiguous().view(1, b * c, *x.size()[2:])

        out = F.batch_norm(
            x_reshaped, running_mean, running_var, self.weight, self.bias,
            True, self.momentum, self.eps)

        return out.view(b, c, *x.size()[2:])

    def __repr__(self):
        return self.__class__.__name__ + '(' + str(self.num_features) + ')'


class LayerNorm(nn.Module):
    def __init__(self, num_features, eps=1e-5, affine=True):
        super(LayerNorm, self).__init__()
        self.num_features = num_features
        self.affine = affine
        self.eps = eps

        if self.affine:
            self.gamma = nn.Parameter(torch.Tensor(num_features).uniform_())
            self.beta = nn.Parameter(torch.zeros(num_features))

    def forward(self, x):
        shape = [-1] + [1] * (x.dim() - 1)
        # print(x.size())
        if x.size(0) == 1:
            # These two lines run much faster in pytorch 0.4 than the two lines listed below.
            mean = x.view(-1).mean().view(*shape)
            std = x.view(-1).std().view(*shape)
        else:
            mean = x.view(x.size(0), -1).mean(1).view(*shape)
            std = x.view(x.size(0), -1).std(1).view(*shape)

        x = (x - mean) / (std + self.eps)

        if self.affine:
            shape = [1, -1] + [1] * (x.dim() - 2)
            x = x * self.gamma.view(*shape) + self.beta.view(*shape)
        return x
