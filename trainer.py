# Training, evaluation, optimizer selection, and update-history handling.

import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from config import device
from learning_rules import make_gradient_rule
from quantization import quantize_model_weights, quantize_parameter_values, quantization_residual_metrics


def daq_overshoot_loss(z_sum, limit, alpha=1e-3):
    # DAQ overshoot penalties stay local to the measured layer.
    return alpha * torch.relu(z_sum - limit).mean()


def make_trainloader(dataset, seed, config):
    generator = torch.Generator().manual_seed(seed)
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=config.shuffle_train_data,
        generator=generator,
    )


def evaluate(model, testloader, criterion, discrete_sets, config):
    # Legacy stochastic quantization intentionally quantizes before evaluation.
    if config.quantize_weights and config.quantization_mode == "stochastic":
        quantize_model_weights(model, discrete_sets, stochastic=True)
    model.eval()
    correct, total, test_loss = 0, 0, 0.0
    with torch.no_grad():
        for images, labels in testloader:
            outputs = model(images.to(device))
            labels = labels.to(device)
            test_loss += criterion(outputs, labels).item()
            correct += (outputs.argmax(dim=1) == labels).sum().item()
            total += labels.size(0)
    return test_loss / len(testloader), 100.0 * correct / total


def _make_optimizer(parameters, learning_rate, config):
    if config.optimizer_name == "sgd":
        return torch.optim.SGD(parameters, lr=learning_rate, weight_decay=config.weight_decay)
    if config.optimizer_name == "sgd_momentum":
        return torch.optim.SGD(
            parameters,
            lr=learning_rate,
            momentum=config.momentum,
            weight_decay=config.weight_decay,
        )
    if config.optimizer_name == "adam":
        return torch.optim.Adam(parameters, lr=learning_rate, weight_decay=config.weight_decay)
    if config.optimizer_name == "adamw":
        return torch.optim.AdamW(parameters, lr=learning_rate, weight_decay=config.weight_decay)
    raise ValueError(f"Unsupported optimizer: {config.optimizer_name}")


def _validate_quantization_strategy(strategy):
    valid = {"shadow_weight_residual", "only_residual", "only_residual_with_reset"}
    if strategy not in valid:
        raise ValueError(f"Unsupported quantization strategy: {strategy}. Expected one of {sorted(valid)}")


def _copy_gradients(source_model, target_parameters):
    for name, parameter in source_model.named_parameters():
        target_parameters[name].grad = None if parameter.grad is None else parameter.grad.detach().clone()


def _save_update_history(history, seed, learning_rate, config):
    if not config.record_update_history or not history:
        return
    os.makedirs(config.update_history_dir, exist_ok=True)
    tag = str(learning_rate).replace(".", "p")
    path = os.path.join(config.update_history_dir, f"update_history_lr_{tag}_seed_{seed}.csv")
    pd.DataFrame(history).to_csv(path, index=False)


def train_one_seed(model_factory, seed, learning_rate, trainset, testloader, discrete_sets, config, max_read_fc1):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    model = model_factory(seed).to(device)
    if config.quantize_weights and config.quantize_after_update:
        _validate_quantization_strategy(config.quantization_strategy)
    if config.quantize_weights and config.quantize_after_update:
        quantize_model_weights(model, discrete_sets, stochastic=config.quantization_mode == "stochastic")
    shadow_parameters = None
    residuals = None
    if config.quantize_weights and config.quantize_after_update and config.quantization_strategy == "shadow_weight_residual":
        shadow_parameters = {
            name: torch.nn.Parameter(parameter.detach().clone())
            for name, parameter in model.named_parameters()
        }
        optimizer = _make_optimizer(shadow_parameters.values(), learning_rate, config)
    else:
        optimizer = _make_optimizer(model.parameters(), learning_rate, config)
        if config.quantize_weights and config.quantize_after_update:
            residuals = {
                name: torch.zeros_like(parameter)
                for name, parameter in model.named_parameters()
            }
    gradient_rule = make_gradient_rule(config, max_read_fc1)
    criterion = nn.CrossEntropyLoss()
    trainloader = make_trainloader(trainset, seed, config)
    best_accuracy, best_epoch, best_loss, best_state = -1.0, None, None, None
    update_history = []
    update_index = 0

    for epoch in range(config.epoch_limit):
        model.train()
        for images, labels in trainloader:
            update_index += 1
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            if shadow_parameters is not None:
                model.zero_grad(set_to_none=True)
            outputs = model(images)
            loss = criterion(outputs, labels)
            extra_conv1 = extra_conv2 = None
            if config.use_daq_overshoot_loss:
                extra_conv1 = daq_overshoot_loss(model._last_sum9_conv1, 200)
                extra_conv2 = daq_overshoot_loss(model._last_sum9_conv2, 200)
                loss = loss + extra_conv1 + extra_conv2
            gradient_rule.backward(model, loss, outputs, labels, extra_conv1, extra_conv2)
            weights_before = {
                name: parameter.detach().clone()
                for name, parameter in model.named_parameters()
            }
            if shadow_parameters is not None:
                shadow_before = {
                    name: parameter.detach().clone()
                    for name, parameter in shadow_parameters.items()
                }
                _copy_gradients(model, shadow_parameters)
            optimizer.step()
            if shadow_parameters is not None:
                weights_after_optimizer = {
                    name: parameter.detach().clone()
                    for name, parameter in shadow_parameters.items()
                }
                quantized_values = quantize_parameter_values(
                    weights_after_optimizer,
                    discrete_sets,
                    stochastic=config.quantization_mode == "stochastic",
                )
                with torch.no_grad():
                    for name, parameter in model.named_parameters():
                        parameter.copy_(quantized_values[name])
                residual_metrics = quantization_residual_metrics(
                    weights_after_optimizer,
                    quantized_values,
                    discrete_sets,
                )
            else:
                weights_after_optimizer = {
                    name: parameter.detach().clone()
                    for name, parameter in model.named_parameters()
                }
                quantized_values = None
                if config.quantize_weights and config.quantize_after_update:
                    candidate_values = {
                        name: weights_after_optimizer[name] + residuals[name]
                        for name in weights_after_optimizer
                    }
                    quantized_values = quantize_parameter_values(
                        candidate_values,
                        discrete_sets,
                        stochastic=config.quantization_mode == "stochastic",
                    )
                    with torch.no_grad():
                        for name, parameter in model.named_parameters():
                            if config.quantization_strategy == "only_residual_with_reset":
                                crossed_threshold = quantized_values[name] != weights_before[name]
                                residuals[name] = candidate_values[name] - quantized_values[name]
                                residuals[name].masked_fill_(crossed_threshold, 0.0)
                            else:
                                residuals[name] = candidate_values[name] - quantized_values[name]
                            parameter.copy_(quantized_values[name])
                    residual_metrics = quantization_residual_metrics(
                        candidate_values,
                        quantized_values,
                        discrete_sets,
                    )
                else:
                    residual_metrics = {
                        "max_abs_residual": 0.0,
                        "threshold_at_max_residual": float("inf"),
                        "max_residual_threshold_ratio": 0.0,
                    }
            if config.record_update_history:
                optimizer_deltas = torch.cat([(weights_after_optimizer[name] - (
                    shadow_before[name] if shadow_parameters is not None else weights_before[name]
                )).abs().reshape(-1) for name in weights_before])
                quantized_deltas = torch.cat([
                    (model.state_dict()[name] - weights_before[name]).abs().reshape(-1)
                    for name in weights_before
                ])
                update_history.append({
                    "seed": seed,
                    "learning_rate": learning_rate,
                    "epoch": epoch + 1,
                    "update": update_index,
                    "loss": float(loss.item()),
                    "gradient_mode": config.gradient_mode,
                    "optimizer_delta_mean": float(optimizer_deltas.mean().item()),
                    "optimizer_delta_max": float(optimizer_deltas.max().item()),
                    "quantized_delta_mean": float(quantized_deltas.mean().item()),
                    "quantized_delta_max": float(quantized_deltas.max().item()),
                    "quantization_zero_fraction": float((quantized_deltas == 0).float().mean().item()),
                    "quantization_strategy": config.quantization_strategy if config.quantize_weights else "disabled",
                    "max_abs_residual": residual_metrics["max_abs_residual"],
                    "threshold_at_max_residual": residual_metrics["threshold_at_max_residual"],
                    "max_residual_threshold_ratio": residual_metrics["max_residual_threshold_ratio"],
                })

        test_loss, accuracy = evaluate(model, testloader, criterion, discrete_sets, config)
        if accuracy > best_accuracy:
            best_accuracy, best_epoch, best_loss = accuracy, epoch + 1, test_loss
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        if accuracy >= config.target_accuracy:
            break

    _save_update_history(update_history, seed, learning_rate, config)
    if update_history and config.quantize_weights and config.quantize_after_update:
        zero_fractions = [row["quantization_zero_fraction"] for row in update_history]
        residual_ratios = [row["max_residual_threshold_ratio"] for row in update_history]
        max_residual = max(row["max_abs_residual"] for row in update_history)
        max_threshold_ratio = max(residual_ratios)
        print(
            f"Quantization diagnostic: {np.mean(zero_fractions) * 100.0:.2f}% "
            "of parameter values kept their pre-update value after quantization; "
            f"max residual={max_residual:.6g}, max residual/threshold={max_threshold_ratio:.2%}"
        )
    return {
        "seed": seed,
        "learning_rate": learning_rate,
        "reached_target": best_accuracy >= config.target_accuracy,
        "epochs_to_target": best_epoch if best_accuracy >= config.target_accuracy else None,
        "best_epoch": best_epoch,
        "best_accuracy": best_accuracy,
        "best_loss": best_loss,
        "state_dict": best_state,
    }