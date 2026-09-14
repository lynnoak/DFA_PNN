# Shared experiment configuration and hardware calibration constants.

from dataclasses import dataclass, replace
import torch


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

A_QUAD = torch.tensor([[-0.044165, 0.17314], [-0.044165, 0.17314]], dtype=torch.float32)
B_QUAD = torch.tensor([[0.335825, 0.157285], [0.335825, 0.157285]], dtype=torch.float32)
C_QUAD = torch.tensor([[-0.495205, -0.793455], [-0.495205, -0.793455]], dtype=torch.float32)
A1_CONV1 = torch.full((2, 2), 0.4, dtype=torch.float32)
B1_CONV1 = torch.tensor([[3.0, 1.9], [3.0, 1.9]], dtype=torch.float32)


@dataclass(frozen=True)
class ExperimentConfig:
    batch_size: int = 1
    # Learning-rate grid used by the DFA comparison.
    learning_rates: tuple = (0.01, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0)
    epoch_limit: int = 100
    target_accuracy: float = 100.0
    seed_start: int = 0
    seed_end: int = 99
    optimizer_name: str = "sgd"
    momentum: float = 0.9
    weight_decay: float = 0.0
    gradient_mode: str = "dfa"
    feedback_mode: str = "orthogonal"
    feedback_scale: float = 1.0
    output_error_mode: str = "softmax"
    conv_output_divisor: float = 4.0
    use_daq_overshoot_loss: bool = False
    shuffle_train_data: bool = False
    quantize_weights: bool = True
    quantize_after_update: bool = True
    quantization_strategy: str = "shadow_weight_residual"
    quantization_mode: str = "nearest"
    record_update_history: bool = True
    update_history_dir: str = "DFA_update_history_general_error"
    train_samples: int = 2
    test_samples: int = 20
    weights_dir: str = "weights_line1_DFA"

    def with_preset(self, preset):
        if preset == "sgd_general_error":
            return self
        if preset == "legacy_adamw":
            return replace(
                self,
                optimizer_name="adamw",
                weight_decay=1e-3,
                output_error_mode="softmax",
                conv_output_divisor=2.0,
                use_daq_overshoot_loss=True,
                shuffle_train_data=True,
                quantize_after_update=False,
                quantization_mode="stochastic",
                record_update_history=False,
            )
        raise ValueError(f"Unsupported training preset: {preset}")


DEFAULT_CONFIG = ExperimentConfig()