# Hardware-aware neural-network forward pass and analog activation.

import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from config import A1_CONV1, B1_CONV1, A1_CONV2, B1_CONV2, A_QUAD, B_QUAD, C_QUAD
from quantization import device_positions_for_count


def make_feedback_matrix(rows, cols, generator, mode="orthogonal", scale=1.0):
    if mode == "orthogonal":
        if rows > cols:
            raise ValueError("Row-orthogonal feedback requires rows <= cols")
        random_matrix = torch.randn(cols, rows, generator=generator)
        q_matrix, _ = torch.linalg.qr(random_matrix, mode="reduced")
        return scale * q_matrix.transpose(0, 1)
    if mode == "random":
        return scale * torch.randn(rows, cols, generator=generator) / np.sqrt(2.0)
    raise ValueError(f"Unsupported feedback mode: {mode}")


class AnalogReLU(nn.Module):
    def __init__(self, vth=0.0, gain=1.0):
        super().__init__()
        self.register_buffer("vth", torch.tensor(float(vth)))
        self.register_buffer("gain", torch.tensor(float(gain)))

    def forward(self, z):
        return torch.clamp(F.relu(z - self.vth) * self.gain, 0.0, 1.0)


class Net(nn.Module):
    def __init__(self, max_read_conv1, max_read_conv2, max_read_fc1, conv_output_divisor, random_seed=0,
                 feedback_mode="orthogonal", feedback_scale=1.0):
        super().__init__()
        self.conv_output_divisor = conv_output_divisor
        self.conv1 = nn.Conv2d(1, 1, kernel_size=(2, 2), bias=False)
        self.conv2 = nn.Conv2d(1, 2, kernel_size=(2, 2), bias=False)
        self.register_buffer("A1_conv1", A1_CONV1.view(1, 1, 2, 2, 1))
        self.register_buffer("B1_conv1", B1_CONV1.view(1, 1, 2, 2, 1))
        self.register_buffer("A1_conv2", A1_CONV2.view(1, 1, 2, 2, 1))
        self.register_buffer("B1_conv2", B1_CONV2.view(1, 1, 2, 2, 1))
        self.register_buffer("A_quad", A_QUAD.view(1, 1, 2, 2, 1))
        self.register_buffer("B_quad", B_QUAD.view(1, 1, 2, 2, 1))
        self.register_buffer("C_quad", C_QUAD.view(1, 1, 2, 2, 1))
        self.fc1 = nn.Linear(2, 4, bias=False)
        self.fc2 = nn.Linear(4, 2, bias=False)
        generator = torch.Generator().manual_seed(int(random_seed))
        self.register_buffer("B_conv1", make_feedback_matrix(2, 4, generator, feedback_mode, feedback_scale))
        self.register_buffer("B_conv2", make_feedback_matrix(2, 2, generator, feedback_mode, feedback_scale))
        self.register_buffer("B_fc1", make_feedback_matrix(2, 4, generator, feedback_mode, feedback_scale))
        self.relu1 = AnalogReLU(gain=1.0 / max_read_conv1)
        self.relu2 = AnalogReLU(gain=1.0 / max_read_conv2)
        self.max_read_fc1 = max_read_fc1
        with torch.no_grad():
            self.conv1.weight.uniform_(-0.2, 0.8, generator=generator)
            self.conv2.weight.uniform_(-0.2, 0.8, generator=generator)

    def custom_convolution(self, x, conv_layer, daq_limit=None):
        batch, channels, height, width = x.shape
        out_channels, _, kernel_h, kernel_w = conv_layer.weight.shape
        patches = F.unfold(x, kernel_size=(kernel_h, kernel_w))
        locations = patches.shape[-1]
        patches = patches.view(batch, channels, kernel_h, kernel_w, locations)
        A_lin, B_lin = (self.A1_conv1, self.B1_conv1) if conv_layer is self.conv1 else (self.A1_conv2, self.B1_conv2)
        transformed = A_lin * patches + B_lin
        transformed = self.A_quad * transformed**2 + self.B_quad * transformed + self.C_quad
        weights = conv_layer.weight.view(1, out_channels, channels, kernel_h, kernel_w, 1)
        transformed = transformed.view(batch, 1, channels, kernel_h, kernel_w, locations)
        sum9 = (transformed * weights).sum(dim=(2, 3, 4))
        if self.training:
            setattr(self, "_last_sum9_conv1" if conv_layer is self.conv1 else "_last_sum9_conv2", sum9)
        if daq_limit is not None:
            sum9 = torch.clamp(sum9, max=daq_limit)
        result = sum9 / self.conv_output_divisor
        return result.view(batch, out_channels, height - kernel_h + 1, width - kernel_w + 1)

    def custom_fc(self, x, fc_layer, positions):
        rows = torch.tensor([pos[0] for pos in positions], device=x.device)
        cols = torch.tensor([pos[1] for pos in positions], device=x.device)
        A = self.A1_conv1.view(2, 2)[rows, cols].view(1, -1)
        B = self.B1_conv1.view(2, 2)[rows, cols].view(1, -1)
        quad_a = self.A_quad.view(2, 2)[rows, cols].view(1, -1)
        quad_b = self.B_quad.view(2, 2)[rows, cols].view(1, -1)
        quad_c = self.C_quad.view(2, 2)[rows, cols].view(1, -1)
        modified = A * x + B
        transformed = quad_a * modified**2 + quad_b * modified + quad_c
        return transformed.matmul(fc_layer.weight.t()) / x.shape[1]

    def forward(self, x):
        start = time.time()
        self.conv1_pre = self.custom_convolution(x, self.conv1, daq_limit=200)
        self.conv1_act = self.relu1(self.conv1_pre)
        self.conv2_pre = self.custom_convolution(self.conv1_act, self.conv2, daq_limit=200)
        self.conv2_act = self.relu2(self.conv2_pre)
        self.conv_time = getattr(self, "conv_time", 0.0) + time.time() - start
        start = time.time()
        x = torch.flatten(self.conv2_act, 1)
        self.fc1_input = x
        self.fc1_pre = self.custom_fc(x, self.fc1, device_positions_for_count(2))
        self.fc1_relu = F.relu(self.fc1_pre)
        self.fc1_act = torch.clamp(self.fc1_relu / self.max_read_fc1, 0.0, 1.0)
        output = self.custom_fc(self.fc1_act, self.fc2, device_positions_for_count(4))
        self.fc_time = getattr(self, "fc_time", 0.0) + time.time() - start
        return output