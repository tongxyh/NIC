import torch
import torch.nn as nn
from torch.autograd import Function
from torch import Tensor
import torch.nn.functional as F


class Round(Function):

    @staticmethod
    def forward(self, input):
        output = torch.round(input)
        return output

    @staticmethod
    def backward(self, grad_output):
        grad_input = grad_output.clone()
        return grad_input


class WeightQuantizer(nn.Module):

    def __init__(self, bits, nl):
        super(WeightQuantizer, self).__init__()
        self.bits = bits
        self.nl = nl
        self.nr = self.bits - self.nl

    def forward(self, tensor_input: torch.Tensor):
        tensor_input = tensor_input.clamp(-2 ** self.nl + 1, 2 ** self.nl - 1)
        tensor_max = tensor_input.abs().max().item()
        r_fix = (Round.apply(torch.Tensor([tensor_max]) * (2 ** self.nr)) / (2 ** self.nr)).cuda()
        if r_fix == 0:
            r_fix = 2 ** (-self.nr)
        tensor_input = tensor_input / r_fix
        tensor_input = tensor_input.clamp(-1, 1)
        tensor_output = Round.apply(tensor_input * (2 ** (self.bits - 1))) / (2 ** (self.bits - 1))
        tensor_output = (tensor_output * r_fix).cuda()
        return tensor_output, r_fix


class ActiveQuantizer(WeightQuantizer):
    def __init__(self, bits, nl):
        super(ActiveQuantizer, self).__init__(bits=bits, nl=nl)

    def forward(self, tensor_input: torch.Tensor):
        # tensor_input = tensor_input.clamp(-2 ** self.nl + 1, 2 ** self.nl - 1)
        tensor_output = torch.round(tensor_input * (2 ** (self.nr - 1)))
        return tensor_output


class Quantizer(nn.Module):
    def __init__(self, precise=10):
        super(Quantizer, self).__init__()
        self.precise = precise

    def forward(self, input):
        return Round.apply(input << self.precise) >> self.precise


class Conv3d_Q(nn.Conv3d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1,
                 bias=True, padding_mode='zeros'):
        super(Conv3d_Q, self).__init__(in_channels, out_channels, kernel_size, stride,
                                       padding, dilation, groups,
                                       bias, padding_mode)
        self.quantizer = Quantizer(16)

    def forward(self, input: Tensor) -> Tensor:
        output = F.conv3d(self.quantizer(input), self.quantizer(self.weight), self.quantizer(self.bias),
                          self.stride,
                          self.padding, self.dilation, self.groups)
        output = self.quantizer(output)
        return output


class Conv2d_Q(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1,
                 bias=True, padding_mode='zeros'):
        super(Conv2d_Q, self).__init__(in_channels, out_channels, kernel_size, stride,
                                       padding, dilation, groups,
                                       bias, padding_mode)
        self.quantizer = Quantizer(16)

    def forward(self, input: Tensor) -> Tensor:
        output = F.conv2d(self.quantizer(input), self.quantizer(self.weight), self.quantizer(self.bias),
                          self.stride,
                          self.padding, self.dilation, self.groups)
        output = self.quantizer(output)
        return output


class ConvTrans2d_Q(nn.ConvTranspose2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, output_padding=0, groups=1, bias=True,
                 dilation=1, padding_mode='zeros'):
        super(ConvTrans2d_Q, self).__init__(in_channels, out_channels, kernel_size, stride,
                                            padding, output_padding, groups, bias,
                                            dilation, padding_mode)

        self.quantizer = Quantizer(16)

    def forward(self, input, output_size=None):
        if self.padding_mode != 'zeros':
            raise ValueError('Only `zeros` padding mode is supported for ConvTranspose2d')
        output_padding = self._output_padding(input, output_size, self.stride, self.padding, self.kernel_size)

        output = F.conv_transpose2d(
            self.quantizer(input), self.quantizer(self.weight), self.quantizer(self.bias), self.stride, self.padding,
            output_padding, self.groups, self.dilation)
        output = self.quantizer(output)
        return output


class Hardmax(Function):

    @staticmethod
    def forward(ctx, input):
        f1_max = input.max(-1).values.unsqueeze(-1)
        output = input.ge(f1_max).float()  # hard max
        return output

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = grad_output.clone()
        return grad_input


class Sigmoid_Q(Function):

    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        output = torch.round(input << 10) >> 14 + 1 >> 1
        output = output.clamp(0, 1)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = grad_output.clone() * 0.25
        input, = ctx.saved_tensors
        grad_input[input < -2] = 0
        grad_input[input > 2] = 0
        return grad_input
