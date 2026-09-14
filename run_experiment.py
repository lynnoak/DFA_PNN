# Modular experiment entry point for comparing DFA and conventional BP.

import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from config import DEFAULT_CONFIG, device
from data import LineDataset
from hardware_model import Net
from quantization import compute_max_read_from_individual_devices, load_discrete_weight_sets_from_folders, quantize_model_weights
from trainer import train_one_seed


TRAINING_PRESET = "sgd_general_error"
GRADIENT_MODE = "dfa"  # Change to "bp" for conventional backpropagation.
QUANTIZE_WEIGHTS = True  # Change to False to disable hardware weight quantization.
QUANTIZATION_STRATEGIES = (
    "shadow_weight_residual",
    "only_residual",
    "only_residual_with_reset",
)

OPTIMIZER_NAME = "sgd_momentum"  # "sgd", "sgd_momentum", "adam", or "adamw".
MOMENTUM = 0.9  # Used by "sgd_momentum".
LEARNING_RATES = (0.2, 0.5, 1.0)
SEED_START = 0
SEED_END = 19
EPOCH_LIMIT = 100
DATASET_MODE = "generalization"  # Change to "clean" for the six clean prototypes.


def set_random_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def save_weights(model, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    np.savetxt(os.path.join(output_dir, "conv1_weights.csv"), model.conv1.weight.detach().cpu().numpy().reshape(1, -1), delimiter=",")
    np.savetxt(os.path.join(output_dir, "fc1_weights_line.csv"), model.fc1.weight.detach().cpu().numpy(), delimiter=",")
    np.savetxt(os.path.join(output_dir, "B_conv1_DFA.csv"), model.B_conv1.detach().cpu().numpy(), delimiter=",")


def main():
    # default configuration is loaded and modified based on the selected preset and gradient mode.
    config = DEFAULT_CONFIG.with_preset(TRAINING_PRESET)
    # override the default configuration need to update before the final version of the code.
    config = config.__class__(
        **{
            **config.__dict__,
            "gradient_mode": GRADIENT_MODE,
            "quantize_weights": QUANTIZE_WEIGHTS,
            "optimizer_name": OPTIMIZER_NAME,
            "momentum": MOMENTUM,
            "learning_rates": LEARNING_RATES,
            "seed_start": SEED_START,
            "seed_end": SEED_END,
            "epoch_limit": EPOCH_LIMIT,
            "dataset_mode": DATASET_MODE,
        }
    )
    print("Training configuration:")
    print(f"  gradient_rule: {config.gradient_mode}")
    print(f"  quantize_weights: {config.quantize_weights}")
    print(f"  quantization_strategies: {QUANTIZATION_STRATEGIES}")
    print(f"  optimizer: {config.optimizer_name}")
    print(f"  momentum: {config.momentum}")
    print(f"  learning_rates: {config.learning_rates}")
    print(f"  target_error: {config.output_error_mode} - one_hot")
    print(f"  feedback_matrix: {config.feedback_mode}")
    print(f"  dataset_mode: {config.dataset_mode}")
    print(f"  seeds: {config.seed_start}..{config.seed_end}")
    set_random_seed(0)
    max_read = compute_max_read_from_individual_devices(config.weights_dir)
    discrete_sets = load_discrete_weight_sets_from_folders(config.weights_dir)
    trainset = LineDataset(
        split="train",
        dataset_mode=config.dataset_mode,
        seed=config.dataset_seed,
        train_corruptions_per_prototype=config.train_corruptions_per_prototype,
        test_corruptions_per_class=config.test_corruptions_per_class,
    )
    testset = LineDataset(
        split="test",
        dataset_mode=config.dataset_mode,
        seed=config.dataset_seed,
        train_corruptions_per_prototype=config.train_corruptions_per_prototype,
        test_corruptions_per_class=config.test_corruptions_per_class,
    )
    testloader = torch.utils.data.DataLoader(testset, batch_size=len(testset), shuffle=False)

    def model_factory(seed):
        return Net(
            max_read,
            config.conv_output_divisor,
            random_seed=seed,
            feedback_mode=config.feedback_mode,
            feedback_scale=config.feedback_scale,
        )

    results = []
    summary_rows = []
    for quantization_strategy in QUANTIZATION_STRATEGIES:
        config = config.__class__(
            **{
                **config.__dict__,
                "quantization_strategy": quantization_strategy,
            }
        )
        print(f"\nStarting quantization strategy: {quantization_strategy}")
        for learning_rate in config.learning_rates:
            successful_for_rate = []
            for seed in range(config.seed_start, config.seed_end + 1):
                print(
                    f"\nStarting strategy {quantization_strategy}, learning rate {learning_rate}, seed {seed} "
                    f"({seed - config.seed_start + 1}/{config.seed_end - config.seed_start + 1})",
                    flush=True,
                )
                result = train_one_seed(model_factory, seed, learning_rate, trainset, testloader, discrete_sets, config, max_read)
                result["quantization_strategy"] = quantization_strategy
                results.append(result)
                if result["reached_target"]:
                    successful_for_rate.append(result)
                    result_row = {key: value for key, value in result.items() if key != "state_dict"}
                    summary_rows.append(result_row)
                    print(f"\nCompleted successful seed {seed} at learning rate {learning_rate}:")
                    print(pd.DataFrame([result_row]).to_string(index=False))

            if not successful_for_rate:
                summary_rows.append({
                    "seed": None,
                    "learning_rate": learning_rate,
                    "quantization_strategy": quantization_strategy,
                    "reached_target": False,
                    "epochs_to_target": None,
                    "best_epoch": None,
                    "best_accuracy": None,
                    "best_loss": None,
                })
                print(
                    f"\nStrategy {quantization_strategy}, learning rate {learning_rate}: "
                    "reached_target = false"
                )

    summary = pd.DataFrame(summary_rows)
    print("\nAll training results:")
    print(summary.to_string(index=False))
    successful = [result for result in results if result["reached_target"]]
    for quantization_strategy in QUANTIZATION_STRATEGIES:
        strategy_successful = [
            result for result in successful
            if result["quantization_strategy"] == quantization_strategy
        ]
        if strategy_successful:
            selected = min(strategy_successful, key=lambda result: result["epochs_to_target"])
            model = model_factory(selected["seed"]).to(device)
            model.load_state_dict(selected["state_dict"])
            if config.quantize_weights:
                quantize_model_weights(model, discrete_sets, stochastic=False)
            output_dir = os.path.join(
                "DFA_weights_modular",
                quantization_strategy,
                f"{config.gradient_mode}_seed_{selected['seed']}",
            )
            save_weights(model, output_dir)
            summary.to_csv(os.path.join(output_dir, "summary.csv"), index=False)


if __name__ == "__main__":
    main()