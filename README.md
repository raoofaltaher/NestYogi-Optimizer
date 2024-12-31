# NestYogi Optimizer

![Python 3.12](https://www.python.org/static/community_logos/python-logo-master-v3-TM.png)

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python Versions](https://img.shields.io/pypi/pyversions/nestyogi.svg)](https://python.org/)
[![Build Status](https://github.com/yourusername/NestYogi-optimizer/workflows/CI/badge.svg)](https://github.com/yourusername/NestYogi-optimizer/actions)


**NestYogi** is a novel hybrid optimization algorithm designed to enhance the generalization and convergence rates of deep learning models, particularly in facial biometric applications. Proposed in the paper ["Novel Hybrid Optimization Techniques for Enhanced Generalization and Faster Convergence in Deep Learning Models: The NestYogi Approach to Facial Biometrics"](https://www.mdpi.com/2227-7390/12/18/2919), NestYogi integrates the adaptive learning capabilities of the Yogi optimizer, anticipatory updates of Nesterov momentum, and the generalization power of Stochastic Weight Averaging (SWA).

doi: (https://doi.org/10.3390/math12182919)

This repository provides implementations of the NestYogi optimizer in both **TensorFlow** and **PyTorch**, facilitating its integration into various deep learning frameworks for tasks such as face detection and recognition.

- **Hybrid Optimization Strategy**: Combines Yogi's adaptive learning rates, Nesterov momentum, and SWA for improved performance.
- **Framework Compatibility**: Implemented for both TensorFlow and PyTorch.
- **Advanced Regularization**: Supports L1 and L2 regularization, gradient clipping, and weight decay.
- **Lookahead Mechanism**: Optional Lookahead for stabilizing training.
- **AMSGrad Variant**: Option to use the AMSGrad variant for enhanced stability.
- **Extensive Documentation**: Detailed analysis and usage instructions based on the original research paper.
- © 2024 by the authors. **Licensee MDPI, Basel, Switzerland**. This code implementation for The article distributed under the terms and conditions of the Creative Commons Attribution (CC BY) license (https://creativecommons.org/licenses/by/4.0/).


---


## Mechanism

1. **Initialize State**: Set up step counter, moment vectors, and any other required state variables.
2. **Compute Gradients**: Optionally clip and decay gradients.
3. **Update Moments**: Adjust first and second moments using gradients and specified hyperparameters.
4. **Bias Correction**: Correct bias in moment estimates due to initialization.
5. **Parameter Update**: Update parameters using the corrected moments and specified momentum type.
6. **Lookahead**: Periodically sync parameters with slow-moving versions for stability.


---


## Initialization and Parameters

The **__init__** method initializes the optimizer with several parameters, ensuring they are within valid ranges:

* **params**: Parameters to optimize.
* **lr**: Learning rate, must be non-negative.
* **betas**: Coefficients for computing running averages of gradient and its square, each must be in [0, 1).
* **eps**: A small term to improve numerical stability, must be non-negative.
* **l1_regularization_strength**: Strength for L1 regularization, non-negative.
* **l2_regularization_strength**: Strength for L2 regularization, non-negative.
* **initial_accumulator_value**: Initial value for accumulators, used for maintaining running averages.
* **activation**: Can be 'sign' or 'tanh', used in computing v_delta.
* **momentum_type**: 'classical' or 'nesterov', determines the type of momentum used.
* **weight_decay**: Coefficient for weight decay, non-negative.
* **amsgrad**: Boolean, whether to use AMSGrad variant.
* **clip_grad_norm**: Value for gradient clipping, optional.
* **lookahead**: Boolean, whether to use the Lookahead mechanism.
* **k** and **alpha**: Parameters for Lookahead, k is the interval and alpha is the interpolation factor.


---


## State Initialization

In the `step` method, the state is initialized if it is empty for a parameter. This includes step counter, first moment vector (`m`), second moment vector (`v`), optional max second moment vector (`max_v`), and optionally a copy of the parameter for Lookahead (`slow_param`).

## Gradient Clipping and Weight Decay

If `clip_grad_norm` is set, gradients are clipped to this norm. If `weight_decay` is non-zero, it is applied to the gradients.

## Bias Correction

The bias correction terms are computed to adjust for initialization bias in the running averages:

```python
bias_correction1 = 1 - beta1 ** state['step']
bias_correction2 = 1 - beta2 ** state['step']
```

## Regularization

If L1 or L2 regularization strengths are set, they are applied to the gradients:

* **L1**: Adds the L1 regularization term.
* **L2**: Adds the L2 regularization term.

## First and Second Moment Updates

The first moment vector `m` is updated as:

```python
m.mul_(beta1).add_(grad, alpha=1 - beta1)
```

The second moment vector `v` is updated differently depending on the activation function:

* If `activation` is 'sign':
```python
v_delta = (grad.pow(2) - v).sign_()
```
* If `activation` is 'tanh':
```python
v_delta = torch.tanh(10 * (grad.pow(2) - v))
```

## AMSGrad

If `amsgrad` is enabled, the maximum of `v` and the existing `max_v` is taken to ensure the second moment does not decrease:

```python
torch.max(state['max_v'], v, out=state['max_v'])
v_hat = state['max_v']
```

Otherwise, `v_hat` is just `v`.

## Parameter Updates

The parameters are updated differently depending on the momentum type:

* **Classical momentum**:
```python
p.addcdiv_(m, denom, value=-step_size)
```
* **Nesterov momentum**:
```python
m_nesterov = m.mul(beta1).add(grad, alpha=1 - beta1)
p.addcdiv_(m_nesterov, denom, value=-step_size)
```

## Lookahead Mechanism

If `lookahead` is enabled, parameters are periodically interpolated with their slow-moving versions:

```python
if group['lookahead'] and state['step'] % group['k'] == 0:
    slow_p = state['slow_param']
    slow_p.add_(p - slow_p, alpha=group['alpha'])
    p.copy_(slow_p)
```

---


## Features and Variables Table

| Feature | Variable | Description |
|---------|----------|-------------|
| Learning Rate | `lr` | Controls the step size for updates. |
| Betas | `betas` | Coefficients for computing running averages of gradient and its square. |
| Epsilon | `eps` | Small term to improve numerical stability. |
| L1 Regularization Strength | `l1_regularization_strength` | Strength for L1 regularization. |
| L2 Regularization Strength | `l2_regularization_strength` | Strength for L2 regularization. |
| Initial Accumulator Value | `initial_accumulator_value` | Initial value for accumulators, used for maintaining running averages. |
| Activation Function | `activation` | Function used for computing `v_delta` (`sign` or `tanh`). |
| Momentum Type | `momentum_type` | Type of momentum used (`classical` or `nesterov`). |
| Weight Decay | `weight_decay` | Coefficient for weight decay. |
| AMSGrad | `amsgrad` | Whether to use AMSGrad variant (`True` or `False`). |
| Gradient Clipping Norm | `clip_grad_norm` | Value for gradient clipping. |
| Lookahead Mechanism | `lookahead` | Whether to use the Lookahead mechanism (`True` or `False`). |
| Lookahead Sync Interval | `k` | Interval for Lookahead synchronization. |
| Lookahead Interpolation Factor | `alpha` | Interpolation factor for Lookahead. |

## Additional State Variables

These variables are initialized and used internally within the `step` method:

| State Variable | Description |
|----------------|-------------|
| `state['step']` | Step counter for the parameter. |
| `state['m']` | First moment vector. |
| `state['v']` | Second moment vector. |
| `state['max_v']` | Maximum second moment vector (used if `amsgrad` is `True`). |
| `state['slow_param']` | Copy of the parameter for Lookahead (used if `lookahead` is `True`). |
