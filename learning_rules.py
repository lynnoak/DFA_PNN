# Backpropagation and Direct Feedback Alignment gradient rules.

import torch
import torch.nn.functional as F


class BackpropRule:
    def backward(self, model, loss, outputs=None, labels=None, extra_conv1=None):
        loss.backward()


class DFARule:
    def __init__(self, output_error_mode="logits"):
        self.output_error_mode = output_error_mode

    @staticmethod
    def _add_grad(parameter, gradient):
        if gradient is None:
            return
        if parameter.grad is None:
            parameter.grad = gradient.detach().clone()
        else:
            parameter.grad.add_(gradient.detach())

    @staticmethod
    def _analog_relu_grad(pre_activation, relu):
        active = (pre_activation > relu.vth) & ((pre_activation - relu.vth) * relu.gain < 1.0)
        return active.to(pre_activation.dtype) * relu.gain

    def backward(self, model, loss, outputs, labels, extra_conv1=None):
        batch_size = labels.shape[0]
        targets = F.one_hot(labels, num_classes=outputs.shape[1]).to(outputs.dtype)
        if self.output_error_mode == "softmax":
            error = (F.softmax(outputs, dim=1) - targets) / batch_size
        elif self.output_error_mode == "logits":
            error = (outputs - targets) / batch_size
        else:
            raise ValueError(f"Unsupported output error mode: {self.output_error_mode}")

        local_loss = (outputs * error.detach()).sum()
        self._add_grad(model.fc1.weight, torch.autograd.grad(local_loss, model.fc1.weight, retain_graph=True)[0])
        delta = error.matmul(model.B_conv1).view_as(model.conv1_act)
        delta = delta * self._analog_relu_grad(model.conv1_pre.detach(), model.relu1)
        local_loss = (model.conv1_pre * delta.detach()).sum()
        self._add_grad(model.conv1.weight, torch.autograd.grad(local_loss, model.conv1.weight, retain_graph=True)[0])

        if extra_conv1 is not None:
            self._add_grad(model.conv1.weight, torch.autograd.grad(extra_conv1, model.conv1.weight, retain_graph=True)[0])


def make_gradient_rule(config):
    if config.gradient_mode == "bp":
        return BackpropRule()
    if config.gradient_mode == "dfa":
        return DFARule(config.output_error_mode)
    raise ValueError(f"Unsupported gradient mode: {config.gradient_mode}")