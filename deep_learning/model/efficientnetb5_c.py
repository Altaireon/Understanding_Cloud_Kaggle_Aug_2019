from .common import *
from external_libs.synchronized_BatchNorm.sync_batchnorm import SynchronizedBatchNorm2d

# https://echo-ai.readthedocs.io/en/latest/_modules/echoAI/Activation/Torch/beta_mish.html#BetaMish


# class Swish(nn.Module):
#     def forward(self, x):
#         return x * torch.sigmoid(x)
#
class SwishFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        result = x * torch.sigmoid(x)
        ctx.save_for_backward(x)
        return result

    @staticmethod
    def backward(ctx, grad_output):
        x = ctx.saved_variables[0]
        sigmoid_x = torch.sigmoid(x)
        return grad_output * (sigmoid_x * (1 + x * (1 - sigmoid_x)))

class Swish(nn.Module):
    def forward(self, x):
        return SwishFunction.apply(x)

#-----------

# https://arxiv.org/pdf/1801.07145.pdf
BETA_SWISH = 1.125
class BetaSwishFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x):
        result = BETA_SWISH*x * torch.sigmoid(x)
        ctx.save_for_backward(x)
        return result

    @staticmethod
    def backward(ctx, grad_output):
        x = ctx.saved_variables[0]
        sigmoid_x = torch.sigmoid(x)
        return grad_output * (sigmoid_x * (BETA_SWISH + x * (1 - sigmoid_x)))

class BetaSwish(nn.Module):
    def forward(self, x):
        return BetaSwishFunction.apply(x)

#-----------

#https://arxiv.org/pdf/1908.08681.pdf
# mish(x) = x * tanh(softplus(x)) = x * tanh(ln(1 + exp(x)))
# class MishFunction(torch.autograd.Function):
#
#     @staticmethod
#     def forward(ctx, x):
#         result = x * torch.tanh(F.softplus(x))   # x * tanh(ln(1 + exp(x)))
#         ctx.save_for_backward(x)
#         return result
#
#     @staticmethod
#     def backward(ctx, grad_output):
#         x = ctx.saved_variables[0]
#         sigmoid_x = torch.sigmoid(x)
#         return grad_output * (sigmoid_x * (BETA_SWISH + x * (1 - sigmoid_x)))
#
# class Mish(nn.Module):
#     def forward(self, x):
#         return MishFunction.apply(x)

class Mish(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x *( torch.tanh(F.softplus(x)))


# https://github.com/digantamisra98/Beta-Mish
class BetaMish(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        beta=1.5
        return x * torch.tanh(torch.log(torch.pow((1+torch.exp(x)),beta)))





###############################################################################

BatchNorm2d = SynchronizedBatchNorm2d
Act = Mish


IMAGE_RGB_MEAN = [0.485, 0.456, 0.406]
IMAGE_RGB_STD  = [0.229, 0.224, 0.225]

IS_PYTORCH_PAD = True  # True  # False
IS_GATHER_EXCITE = False



def drop_connect(x, probability, training):
    if not training: return x

    batch_size = len(x)
    keep_probability = 1 - probability
    noise = keep_probability
    noise += torch.rand([batch_size, 1, 1, 1], dtype=x.dtype, device=x.device)
    mask = torch.floor(noise)
    x = x / keep_probability * mask

    return x


class Identity(nn.Module):
    def forward(self, x):
        return x

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(len(x),-1)

class Conv2dBn(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size=1, stride=1, zero_pad=[0,0,0,0], group=1):
        super(Conv2dBn, self).__init__()
        if IS_PYTORCH_PAD: zero_pad = [kernel_size//2]*4
        self.pad  = nn.ZeroPad2d(zero_pad)
        self.conv = nn.Conv2d(in_channel, out_channel, kernel_size=kernel_size, padding=0, stride=stride, groups=group, bias=False)
        self.bn   = BatchNorm2d(out_channel, eps=1e-03, momentum=0.01)
        #print(zero_pad)

    def forward(self, x):
        x = self.pad (x)
        x = self.conv(x)
        x = self.bn  (x)
        return x


class SqueezeExcite(nn.Module):
    def __init__(self, in_channel, reduction_channel, excite_size):
        super(SqueezeExcite, self).__init__()
        self.excite_size=excite_size

        self.squeeze = nn.Conv2d(in_channel, reduction_channel, kernel_size=1, padding=0)
        self.excite  = nn.Conv2d(reduction_channel, in_channel, kernel_size=1, padding=0)
        self.act = Act()

    # def forward(self, x):
    #     s = F.adaptive_avg_pool2d(x,1)
    #     s = self.act(self.squeeze(s))
    #     s = torch.sigmoid(self.excite(s))
    #
    #     x = s*x
    #     return x


    def forward(self, x):

        if IS_GATHER_EXCITE:
            s = F.avg_pool2d(x, kernel_size=self.excite_size)
        else:
            s = F.adaptive_avg_pool2d(x,1)

        s = self.act(self.squeeze(s))
        s = torch.sigmoid(self.excite(s))

        if IS_GATHER_EXCITE:
            s = F.interpolate(s, size=(x.shape[2],x.shape[3]), mode='nearest')

        x = s*x
        return x

#---

class EfficientBlock(nn.Module):

    def __init__(self, in_channel, channel, out_channel, kernel_size, stride, zero_pad, excite_size, drop_connect_rate):
        super().__init__()
        self.is_shortcut = stride == 1 and in_channel == out_channel
        self.drop_connect_rate = drop_connect_rate

        if in_channel == channel:
            self.bottleneck = nn.Sequential(
                Conv2dBn(   channel, channel, kernel_size=kernel_size, stride=stride, zero_pad=zero_pad, group=channel),
                Act(),
                SqueezeExcite(channel, in_channel//4, excite_size) if excite_size>0
                else Identity(),
                Conv2dBn(channel, out_channel, kernel_size=1, stride=1),
            )
        else:
            self.bottleneck = nn.Sequential(
                Conv2dBn(in_channel, channel, kernel_size=1, stride=1),
                Act(),
                Conv2dBn(   channel, channel, kernel_size=kernel_size, stride=stride, zero_pad=zero_pad, group=channel),
                Act(),
                SqueezeExcite(channel, in_channel//4, excite_size) if excite_size>0
                else Identity(),
                Conv2dBn(channel, out_channel, kernel_size=1, stride=1)
            )

    def forward(self, x):
        b = self.bottleneck(x)

        if self.is_shortcut:
            if self.training: b = drop_connect(b, self.drop_connect_rate, True)
            x = b + x
        else:
            x = b
        return x



# https://arogozhnikov.github.io/einops/pytorch-examples.html

# actual padding used in tensorflow
class EfficientNetB5(nn.Module):

    def __init__(self, drop_connect_rate=0.4):
        super(EfficientNetB5, self).__init__()
        d = drop_connect_rate

        # bottom-top
        self.stem  = nn.Sequential(
            Conv2dBn(3,48, kernel_size=3,stride=2,zero_pad=[0,1,0,1]),
            Act()
        )

        self.block1 = nn.Sequential(
               EfficientBlock( 48,  48,  24, kernel_size=3, stride=1, zero_pad=[1,1,1,1], excite_size=128, drop_connect_rate=d*1/7),
            * [EfficientBlock( 24,  24,  24, kernel_size=3, stride=1, zero_pad=[1,1,1,1], excite_size=128, drop_connect_rate=d*1/7) for i in range(1,3)],
        )
        self.block2 = nn.Sequential(
               EfficientBlock( 24, 144,  40, kernel_size=3, stride=2, zero_pad=[0,1,0,1], excite_size= 64, drop_connect_rate=d*2/7),
            * [EfficientBlock( 40, 240,  40, kernel_size=3, stride=1, zero_pad=[1,1,1,1], excite_size= 64, drop_connect_rate=d*2/7) for i in range(1,5)],
        )
        self.block3 = nn.Sequential(
               EfficientBlock( 40, 240,  64, kernel_size=5, stride=2, zero_pad=[1,2,1,2], excite_size= 32, drop_connect_rate=d*3/7),
            * [EfficientBlock( 64, 384,  64, kernel_size=5, stride=1, zero_pad=[2,2,2,2], excite_size= 32, drop_connect_rate=d*3/7) for i in range(1,5)],
        )
        self.block4 = nn.Sequential(
               EfficientBlock( 64, 384, 128, kernel_size=3, stride=2, zero_pad=[0,1,0,1], excite_size= 16, drop_connect_rate=d*4/7),
            * [EfficientBlock(128, 768, 128, kernel_size=3, stride=1, zero_pad=[1,1,1,1], excite_size= 16, drop_connect_rate=d*4/7) for i in range(1,7)],
        )
        self.block5 = nn.Sequential(
               EfficientBlock(128, 768, 176, kernel_size=5, stride=1, zero_pad=[2,2,2,2], excite_size= 16, drop_connect_rate=d*5/7),
            * [EfficientBlock(176,1056, 176, kernel_size=5, stride=1, zero_pad=[2,2,2,2], excite_size= 16, drop_connect_rate=d*5/7) for i in range(1,7)],
        )
        self.block6 = nn.Sequential(
               EfficientBlock(176,1056, 304, kernel_size=5, stride=2, zero_pad=[1,2,1,2], excite_size=  8, drop_connect_rate=d*6/7),
            * [EfficientBlock(304,1824, 304, kernel_size=5, stride=1, zero_pad=[2,2,2,2], excite_size=  8, drop_connect_rate=d*6/7) for i in range(1,9)],
        )
        self.block7 = nn.Sequential(
               EfficientBlock(304,1824, 512, kernel_size=3, stride=1, zero_pad=[1,1,1,1], excite_size=  8, drop_connect_rate=d*7/7),
            * [EfficientBlock(512,3072, 512, kernel_size=3, stride=1, zero_pad=[1,1,1,1], excite_size=  8, drop_connect_rate=d*7/7) for i in range(1,3)],
        )

        self.last = nn.Sequential(
            Conv2dBn(512, 2048,kernel_size=1,stride=1),
            Act()
        )

        self.logit = nn.Linear(2048,1000)

    def forward(self, x):
        batch_size = len(x)

        x = self.stem(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        x = self.block6(x)
        x = self.block7(x)
        x = self.last(x)

        x = F.adaptive_avg_pool2d(x,1).reshape(batch_size,-1)
        logit = self.logit(x)

        return logit