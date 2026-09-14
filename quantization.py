# Discrete device-weight loading, quantization, and hardware read helpers.

import glob
import os
import numpy as np
import pandas as pd
import torch

from config import A1_CONV1, B1_CONV1, A_QUAD, B_QUAD, C_QUAD, device


def device_positions_for_count(count, repeat=False):
    positions = [(0, 0), (0, 1), (1, 0), (1, 1)]
    if repeat:
        return [positions[idx % len(positions)] for idx in range(count)]
    if count > len(positions):
        raise ValueError(f"Only up to 4 logical device positions are available, got {count}")
    return positions[:count]


def load_discrete_weight_sets_from_folders(base_dir="weights"):
    weight_sets = {}
    for i in range(2):
        for j in range(2):
            dirpath = os.path.join(base_dir, f"weights_{i}{j}")
            csv_paths = sorted(glob.glob(os.path.join(dirpath, "*.csv")))
            if not csv_paths and i == 1:
                dirpath = os.path.join(base_dir, f"weights_0{j}")
                csv_paths = sorted(glob.glob(os.path.join(dirpath, "*.csv")))
            if not csv_paths:
                raise FileNotFoundError(f"No CSV files found for logical position ({i},{j})")
            values = []
            for path in csv_paths:
                current = pd.read_csv(path, header=None).values.flatten()
                if current.size:
                    values.append(current.astype(np.float32))
            if not values:
                raise ValueError(f"No values loaded for logical position ({i},{j})")
            weight_sets[(i, j)] = torch.tensor(np.concatenate(values), dtype=torch.float32, device=device)
    return weight_sets


def _quantize(weight_tensor, discrete_sets, positions, stochastic):
    result = weight_tensor.clone()
    if weight_tensor.dim() == 4:
        _, _, kernel_h, kernel_w = weight_tensor.shape
        for tap_idx, position in enumerate(positions):
            row, col = divmod(tap_idx, kernel_w)
            result[:, :, row, col] = _quantize_column(
                weight_tensor[:, :, row, col], discrete_sets[position], stochastic
            )
        return result
    if weight_tensor.dim() != 2:
        raise ValueError(f"Expected a 2-D or 4-D weight tensor, got {weight_tensor.dim()} dimensions")
    for col, position in enumerate(positions):
        result[:, col] = _quantize_column(weight_tensor[:, col], discrete_sets[position], stochastic)
    return result


def _quantize_column(values, codes, stochastic):
    flat = values.reshape(-1)
    if not stochastic:
        distances = torch.abs(flat[:, None] - codes[None, :])
        return codes[torch.argmin(distances, dim=1)].reshape(values.shape)
    sorted_codes, _ = torch.sort(codes)
    high_index = torch.searchsorted(sorted_codes, flat, right=True)
    low_index = torch.clamp(high_index - 1, 0, sorted_codes.numel() - 1)
    high_index = torch.clamp(high_index, 0, sorted_codes.numel() - 1)
    low = sorted_codes[low_index]
    high = sorted_codes[high_index]
    exact = (high - low).abs() < 1e-8
    probability_high = (flat - low).abs() / ((flat - low).abs() + (high - flat).abs() + 1e-8)
    result = torch.where(torch.rand_like(probability_high) < probability_high, high, low)
    return torch.where(exact, low, result).reshape(values.shape)


def quantize_weight_tensor(weight_tensor, discrete_sets, positions=None, stochastic=False):
    if positions is None:
        positions = device_positions_for_count(weight_tensor.shape[-1])
    return _quantize(weight_tensor, discrete_sets, positions, stochastic)


def quantize_model_weights(net, discrete_sets, stochastic=False):
    with torch.no_grad():
        quantized = quantize_parameter_values(
            {name: parameter for name, parameter in net.named_parameters()},
            discrete_sets,
            stochastic,
        )
        for name, parameter in net.named_parameters():
            parameter.copy_(quantized[name])


def quantize_parameter_values(parameter_values, discrete_sets, stochastic=False):
    positions_by_name = {
        "conv1.weight": device_positions_for_count(4),
        "conv2.weight": device_positions_for_count(4),
        "fc1.weight": device_positions_for_count(2),
        "fc2.weight": device_positions_for_count(4),
    }
    quantized = {}
    for name, values in parameter_values.items():
        if name not in positions_by_name:
            quantized[name] = values.detach().clone()
            continue
        quantized[name] = quantize_weight_tensor(
            values,
            discrete_sets,
            positions_by_name[name],
            stochastic,
        )
    return quantized


def quantization_residual_metrics(continuous_values, quantized_values, discrete_sets):
    positions_by_name = {
        "conv1.weight": device_positions_for_count(4),
        "conv2.weight": device_positions_for_count(4),
        "fc1.weight": device_positions_for_count(2),
        "fc2.weight": device_positions_for_count(4),
    }
    residuals = []
    thresholds = []
    for name, continuous in continuous_values.items():
        if name not in positions_by_name:
            continue
        quantized = quantized_values[name]
        positions = positions_by_name[name]
        if continuous.dim() == 4:
            _, _, kernel_h, kernel_w = continuous.shape
            columns = [
                (continuous[:, :, row, col].reshape(-1), quantized[:, :, row, col].reshape(-1), position)
                for tap_idx, position in enumerate(positions)
                for row, col in [divmod(tap_idx, kernel_w)]
            ]
        else:
            columns = [
                (continuous[:, col].reshape(-1), quantized[:, col].reshape(-1), position)
                for col, position in enumerate(positions)
            ]
        for current, hardware, position in columns:
            codes = torch.sort(discrete_sets[position]).values
            code_indices = torch.searchsorted(codes, hardware.contiguous())
            code_indices = torch.clamp(code_indices, 0, codes.numel() - 1)
            direction = torch.sign(current - hardware)
            next_indices = torch.where(direction >= 0, code_indices + 1, code_indices - 1)
            valid = (next_indices >= 0) & (next_indices < codes.numel()) & (direction != 0)
            next_indices = torch.clamp(next_indices, 0, codes.numel() - 1)
            distance = torch.where(valid, (codes[next_indices] - hardware).abs(), torch.full_like(hardware, float("inf")))
            residuals.append((current - hardware).abs())
            thresholds.append(distance * 0.5)
    if not residuals:
        return {"max_abs_residual": 0.0, "threshold_at_max_residual": float("inf"), "max_residual_threshold_ratio": 0.0}
    residual = torch.cat(residuals)
    threshold = torch.cat(thresholds)
    max_index = torch.argmax(residual)
    threshold_at_max = threshold[max_index]
    finite_ratio = residual / threshold
    finite_ratio = finite_ratio[torch.isfinite(finite_ratio)]
    return {
        "max_abs_residual": float(residual.max().item()),
        "threshold_at_max_residual": float(threshold_at_max.item()),
        "max_residual_threshold_ratio": float(finite_ratio.max().item()) if finite_ratio.numel() else 0.0,
    }


def compute_max_read_from_individual_devices(base_dir="weights_line"):
    modulation = (A1_CONV1 + B1_CONV1).numpy()
    transformed = A_QUAD.numpy() * modulation**2 + B_QUAD.numpy() * modulation + C_QUAD.numpy()
    max_read = -1.0
    for i in range(2):
        for j in range(2):
            paths = sorted(glob.glob(os.path.join(base_dir, f"weights_{i}{j}", "*.csv")))
            if not paths and i == 1:
                paths = sorted(glob.glob(os.path.join(base_dir, f"weights_0{j}", "*.csv")))
            if not paths:
                continue
            values = [np.loadtxt(path, delimiter=",", dtype=np.float32).reshape(-1) for path in paths]
            all_values = np.concatenate(values)
            device_reads = all_values * float(transformed[i, j])
            max_read = max(max_read, float(np.max(device_reads)))
    return float(max_read)