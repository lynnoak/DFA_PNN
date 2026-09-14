Report document (read-only on Overleaf):
https://www.overleaf.com/read/mtmjjzzdzwhr#e90288

# Modular Hardware-Aware DFA Training Framework

This project reorganizes the existing PNN training code into a modular pipeline so that the hardware-aware forward model, learning rule, parameter update, and hardware-weight mapping can be developed and evaluated independently. The current entry point is `run_experiment.py`. The default configuration is DFA, quantization after every update, and SGD with momentum. Constants in the experiment entry point can be used to switch the gradient rule, optimizer, and quantization strategy.

A single training update can be summarized as:

```text
Input image
  -> hardware-aware forward pass (convolution / fully connected / analog activation)
  -> output error
  -> DFA direct error feedback, or retained BP backpropagation
  -> local gradients
  -> optimizer updates continuous weights
  -> continuous weights are mapped to programmable discrete device states
  -> next batch uses the quantized hardware weights
```

The corresponding modular training cycle is:

```text
Input and label
  -> hardware_model.py: hardware-aware forward pass
  -> learning_rules.py: output error and DFA / BP
  -> trainer.py: optimizer updates continuous weights
  -> quantization.py: projection to device states
  -> next forward pass uses quantized hardware weights
```

The modularization separates the training cycle into four principal functional blocks: the physical forward path, the DFA learning rule, the parameter-update mechanism, and the device-weight mapping. Two additional modules provide data generation and experiment-level control. This separation is important because these blocks correspond to different requirements in a future hardware implementation.

---

## 1. Code Structure

| Module | Main Responsibility | Position in the Training Model |
| --- | --- | --- |
| `data.py` | Generates vertical-line / horizontal-line binary classification data | Input and labels |
| `config.py` | Training hyperparameters, hardware calibration coefficients, device selection | Global configuration and hardware parameters |
| `hardware_model.py` | Hardware-aware convolution, fully connected layers, analog ReLU, DFA feedback matrices | Forward model |
| `learning_rules.py` | BP and DFA error / gradient computation | Backward error and gradients |
| `trainer.py` | Batch handling, loss, optimizer, update, evaluation, update history | Training controller |
| `quantization.py` | Reads device code values, continuous-value quantization, quantization residual diagnostics | Weight-to-hardware-state mapping |
| `run_experiment.py` | Combines configurations, scans strategies / learning rates / random seeds, saves results | Experiment entry point |

This modular design allows the physical forward model, learning rule, parameter update, and device-weight mapping to be replaced or evaluated independently.

---

## 2. Hardware-Aware Forward Path

The forward path is implemented in `hardware_model.py`. It is not an ideal floating-point network; instead, it uses calibrated device-response equations to approximate actual device readout.

Current network structure:

```text
Input 1 x 3 x 3
  -> conv1: 1 -> 1, kernel = 2 x 2, output 1 x 2 x 2
  -> AnalogReLU
  -> flatten, producing 4 features
  -> fc1: 4 -> 2
  -> two-class output logits
```

The data module currently uses artificially generated 3 x 3 images: even samples are center vertical lines, and odd samples are center horizontal lines. This dataset is mainly used to verify the hardware-aware training pipeline rather than a general image recognition task.

Note: in subsequent experiments, this neural network architecture configuration was not suitable for the corresponding test dataset. It is planned to be modified in the future based on the characteristics of the dataset.

### 2.1 Device Readout Model

For each input patch in a convolution, the code first applies a per-position linear modulation:

$$
m_{ij} = A1_{ij}x_{ij} + B1_{ij}
$$

Then it uses a quadratic calibration curve to obtain the device output:

$$
r_{ij} = A_{ij}^{quad}m_{ij}^{2} + B_{ij}^{quad}m_{ij} + C_{ij}^{quad}
$$

Finally, it multiplies by the corresponding weight and accumulates:

$$
s = \sum_{ij} r_{ij}w_{ij}
$$

`custom_convolution` uses `F.unfold` to expand sliding windows, then performs the per-tap transformation, weighted sum, and `conv_output_divisor` scaling. `custom_fc` uses the same calibration idea: each input column is bound to a logical device position, and then matrix multiplication is performed.

The calibration coefficients `A1_*`, `B1_*`, `A_quad`, `B_quad`, and `C_quad` come from `config.py`. The forward pass and `compute_max_read_from_individual_devices` use the same set of coefficients.

### 2.2 Analog Activation and Readout Limits

`AnalogReLU` performs thresholding, gain, and upper-limit clipping:

$$
y = \mathrm{clip}\left(\max(0, z - v_{th}) \times gain,\ 0,\ 1\right)
$$

The convolutional layer applies a DAQ upper limit to the accumulated readout, currently 200. The analog ReLU output is then flattened and sent directly to the single output fully connected layer. This corresponds to the ADC / DAQ dynamic range and analog activation saturation in hardware, rather than post-processing after training.

---

## 3. DFA Error Feedback and BP Baseline

The learning rule is isolated in `learning_rules.py`. Its input is the difference between the forward propagation output and the target labels. Its output consists of local gradients at each location.

### 3.1 Output Error

`DFARule.backward` first converts labels to one-hot vectors, then supports two output-error forms:

- `logits`: `(outputs - targets) / batch_size`. This is the current default for `sgd_general_error`.
- `softmax`: `(softmax(outputs) - targets) / batch_size`. This is retained for compatibility with older experiment settings.

Therefore, the current implementation does not require a full softmax in hardware. The `logits` mode only requires output differences, label encoding, and batch normalization.

### 3.2 DFA Feedback

DFA does not propagate the error layer by layer through transposed weights of the next layer. Instead, it multiplies the output error directly by a fixed feedback matrix for each layer:

$$
\boldsymbol{\delta}^{(l)}
=
\left(\mathbf{e}\mathbf{B}^{(l)}\right)
\odot
f'_{l}\left(\mathbf{a}^{(l)}\right)
$$

Here `B_conv1` is generated when `Net` is initialized according to a random seed. It directly maps the two output errors to the four post-ReLU convolution positions. It can be chosen as:

- `orthogonal`: generates row-orthogonal feedback matrices;
- `random`: generates scaled random matrices.

These matrices remain fixed during training. The current implementation still uses `torch.autograd.grad` to convert these explicitly constructed local DFA signals into weight gradients. Consequently, this module currently represents the algorithmic target for a future DFA feedback / update circuit rather than a fully hardware-native implementation.

### 3.3 Retained BP Option

`BackpropRule` is also retained. When `config.gradient_mode == "bp"`, the code directly executes `loss.backward()`. When it is set to `"dfa"`, `DFARule` is used. Therefore, the same forward model and the same optimization / quantization pipeline can directly compare BP and DFA. BP can also serve as a software baseline or debugging mode when DFA hardware is not available.

---

## 4. Optimizer: From Gradients to Continuous Candidate Weights

The optimizer logic is located in `trainer.py`. DFA / BP only produces gradients; the optimizer is responsible for converting the error into parameter changes.

The general update form is:

$$
W^{*}_{t+1}
=
\mathcal{U}
\left(
W_t,\,
g_t;\,
\eta,\,
S_t
\right)
$$

where `\mathcal{U}` is the selected optimization rule, `S_t` is its internal state, and `W^{*}_{t+1}` is the resulting continuous candidate weight.

Four optimization methods are currently supported:

| Method | Update Characteristic | Hardware Implication |
| --- | --- | --- |
| `sgd` | Directly applies the current gradient, approximately `w <- w - lr * g` | Requires only gradients, learning rate, and an accumulator; easiest to implement as a local update |
| `sgd_momentum` | Stores momentum `v` and smooths updates using historical gradients | Requires additional momentum storage and multiply-accumulate per weight; current default `momentum=0.9` |
| `adam` | Stores first- and second-moment estimates and applies bias correction | Requires two state values per weight, square / square-root or approximation circuits, and higher control complexity |
| `adamw` | Adam adaptive update with decoupled weight decay | Requires Adam state plus an independent weight-decay path |

The current code uses PyTorch optimizers, so optimizer states mainly reside in the software controller / training host. For hardware migration, SGD is the most suitable first implementation. Momentum, Adam, and AdamW require state to be placed in on-chip memory, near-memory controllers, or an external digital controller.

In general, DFA determines the spatial credit-assignment mechanism, that is, how the output error is transformed into a local gradient for each layer. The optimizer determines how these gradients are integrated over training iterations. This separation is important because DFA updates can be noisy, and useful directional information may only become apparent through repeated updates.

---

## 5. Projection to Discrete Device States

The quantization module is located in `quantization.py`. Its input is the continuous candidate weight produced by the optimizer, and its output is a discrete weight value compatible with the programmable states of the physical device.

Let the optimizer produce a continuous candidate weight `W^{*}_{t+1}`. The hardware-compatible weight is obtained by projecting this candidate onto the set of available device states:

$$
W^{\mathrm{HW}}_{t+1}
=
Q_{\mathcal{W}_{\mathrm{device}}}
\left(W^{*}_{t+1}\right)
$$

where `\mathcal{W}_{\mathrm{device}}` denotes the measured set of programmable device weights and `Q` denotes the quantization operator.

### 5.1 Discrete Device Code Values

`quantization.py` reads the available device weight code values for each logical position `(i,j)` from `weights_ij/*.csv`. The four logical positions are:

```text
(0,0)  (0,1)
(1,0)  (1,1)
```

If the second-row directory is missing, the current code reuses the corresponding first-row directory. `compute_max_read_from_individual_devices` uses the discrete values and the same set of calibration curves to estimate the maximum readout, which is used for analog activation gain and readout range.

### 5.2 Numerical Quantization Methods

`quantize_weight_tensor` supports two methods for mapping continuous values to device code values:

- `nearest`: selects the available code value with the smallest absolute distance. It is deterministic and simple to implement.
- `stochastic`: finds the two adjacent code values around the continuous value and selects one randomly according to distance-based probabilities. It is closer to the continuous value on long-term average but requires a random number source.

Convolution weights are bound to logical positions according to each spatial tap of the kernel. Fully connected weights are bound to logical positions according to input columns. `quantize_parameter_values` processes `conv1.weight` and `fc1.weight`, while other parameters remain unchanged.

### 5.3 Quantization Update Strategies

When `quantize_after_update=True`, the trainer supports three residual-aware update strategies. These are quantization update strategies, not the four optimizers above.

| Strategy | Description | Characteristics |
| --- | --- | --- |
| `shadow_weight_residual` | The optimizer updates a continuous shadow weight, which is then quantized and written into the model | The model uses discrete values, while the continuous shadow retains small updates and avoids long-term loss of small gradients |
| `only_residual` | The previous residual is added to the current actual weight before quantization; the new residual is the candidate continuous value minus the quantized value | Preserves unapplied updates and quantization remainder; can be implemented with a digital accumulator in hardware |
| `only_residual_with_reset` | Maintains the residual in the same way, but resets the residual at a position once the discrete code value is crossed and an actual transition occurs | Simplifies hardware implementation and is naturally compatible with capacitor / charge accumulation and threshold-triggered structures, but discards part of the continuous update information |

The trainer also records the continuous optimizer update, the actual quantized update, the zero-update ratio, and the residual ratio relative to the next code-value threshold. These diagnostics can be used to judge whether the learning rate is too small or whether quantization is discarding most updates.

---

## 6. Current Order of One Update

The actual order in `trainer.train_one_seed` is:

1. Create the model and, if needed, quantize the initial weights first.
2. Run the forward pass to obtain hardware-aware outputs, and compute cross-entropy and optional DAQ overshoot penalty.
3. Select DFA or BP according to `gradient_mode` to obtain gradients for each trainable weight.
4. Use one of the four optimizers to update continuous parameters or shadow parameters.
5. Generate discrete candidate values according to the quantization strategy and `nearest` / `stochastic` mode.
6. Copy the discrete values back to the actual model; update the residual if necessary.
7. Evaluate using the current actual discrete weights and record the update history.

The training objective is therefore simultaneously constrained by three factors:

- the forward readout curve limits the signal;
- DFA / BP determines how the error is generated;
- quantization determines whether the parameters can finally be realized on the device.

During hardware validation, these three factors should be checked carefully to ensure that they use the same set of calibration coefficients, readout limits, and discrete code values.

---

## 7. Running and Results

Current entry point:

```bash
python run_experiment.py
```

`run_experiment.py` scans quantization strategies, learning rates, and random seeds, and saves successful models and update summaries to:

```text
DFA_weights_modular/
```

These results can serve as algorithmic baselines before hardware mapping.