# tfNestYogi.py

# to import use "from tfNestYogi import NestYogi",,   check the end of the code.

import tensorflow as tf
from tensorflow_addons.utils.types import FloatTensorLike
from tensorflow_addons.optimizers import Yogi
from typeguard import typechecked
from typing import Union, Callable

class NestYogi(Yogi):
    @typechecked
    def __init__(
        self,
        learning_rate: Union[FloatTensorLike, Callable] = 1e-3,
        beta1: FloatTensorLike = 0.9,
        beta2: FloatTensorLike = 0.999,
        epsilon: FloatTensorLike = 1e-3,
        l1_regularization_strength: FloatTensorLike = 0.0,
        l2_regularization_strength: FloatTensorLike = 0.0,
        initial_accumulator_value: FloatTensorLike = 1e-6,
        activation: str = "sign",
        momentum_type: str = "nesterov",
        weight_decay: FloatTensorLike = 0.0,
        amsgrad: bool = False,
        clip_grad_norm: Union[FloatTensorLike, None] = None,
        lookahead: bool = False,
        lookahead_k: int = 5,
        lookahead_alpha: FloatTensorLike = 0.5,
        name: str = "NestYogi",
        **kwargs
    ):
        super().__init__(
            learning_rate=learning_rate,
            beta1=beta1,
            beta2=beta2,
            epsilon=epsilon,
            l1_regularization_strength=l1_regularization_strength,
            l2_regularization_strength=l2_regularization_strength,
            initial_accumulator_value=initial_accumulator_value,
            activation=activation,
            name=name,
            **kwargs
        )

        if activation not in ['sign', 'tanh']:
            raise ValueError(f"Invalid activation function: {activation}")
        if momentum_type not in ['classical', 'nesterov']:
            raise ValueError(f"Invalid momentum type: {momentum_type}")

        self._set_hyper("beta1", beta1)
        self._set_hyper("beta2", beta2)
        self._set_hyper("epsilon", epsilon)
        self._set_hyper("l1_regularization_strength", l1_regularization_strength)
        self._set_hyper("l2_regularization_strength", l2_regularization_strength)
        self._set_hyper("weight_decay", weight_decay)
        self.activation = activation
        self.momentum_type = momentum_type
        self.amsgrad = amsgrad
        self.clip_grad_norm = clip_grad_norm
        self.lookahead = lookahead
        self.lookahead_k = lookahead_k
        self.lookahead_alpha = lookahead_alpha

    def _create_slots(self, var_list):
        for var in var_list:
            self.add_slot(var, "m")
            self.add_slot(var, "v")
            if self.amsgrad:
                self.add_slot(var, "vhat")
            if self.lookahead:
                self.add_slot(var, "slow_param")

    @tf.function
    def _resource_apply_dense(self, grad, var):
        var_dtype = var.dtype.base_dtype
        lr_t = self._decayed_lr(var_dtype)
        beta1_t = self._get_hyper("beta1", var_dtype)
        beta2_t = self._get_hyper("beta2", var_dtype)
        epsilon_t = self._get_hyper("epsilon", var_dtype)
        l1_t = self._get_hyper("l1_regularization_strength", var_dtype)
        l2_t = self._get_hyper("l2_regularization_strength", var_dtype)
        weight_decay = self._get_hyper("weight_decay", var_dtype)

        m = self.get_slot(var, "m")
        v = self.get_slot(var, "v")

        if self.clip_grad_norm:
            grad = tf.clip_by_norm(grad, self.clip_grad_norm)

        if weight_decay != 0:
            grad = grad + weight_decay * var

        local_step = tf.cast(self.iterations + 1, var_dtype)
        beta1_power = tf.pow(beta1_t, local_step)
        beta2_power = tf.pow(beta2_t, local_step)

        if l1_t > 0:
            grad = grad + l1_t * tf.sign(var) * tf.abs(grad)
        if l2_t > 0:
            grad = grad + l2_t * var

        # Compute m_t
        m_t = m.assign(beta1_t * m + (1 - beta1_t) * grad, use_locking=self._use_locking)

        # Compute v_t
        grad_squared = tf.square(grad)
        if self.activation == 'sign':
            v_delta = tf.sign(grad_squared - v)
        elif self.activation == 'tanh':
            v_delta = tf.tanh(10 * (grad_squared - v))
        v_t = v.assign_add(v_delta * grad_squared * (1 - beta2_t), use_locking=self._use_locking)

        if self.amsgrad:
            vhat = self.get_slot(var, "vhat")
            vhat_t = vhat.assign(tf.maximum(vhat, v_t), use_locking=self._use_locking)
            v_sqrt = tf.sqrt(vhat_t)
        else:
            v_sqrt = tf.sqrt(v_t)

        denom = v_sqrt / tf.sqrt(1 - beta2_power) + epsilon_t
        step_size = lr_t / (1 - beta1_power)

        if self.momentum_type == 'classical':
            var_update = var.assign_sub(step_size * m_t / denom, use_locking=self._use_locking)
        else:  # nesterov
            m_nesterov = beta1_t * m_t + (1 - beta1_t) * grad
            var_update = var.assign_sub(step_size * m_nesterov / denom, use_locking=self._use_locking)

        if self.lookahead:
            slow_param = self.get_slot(var, "slow_param")
            if tf.equal(tf.math.floormod(local_step, self.lookahead_k), 0):
                slow_update = slow_param + self.lookahead_alpha * (var - slow_param)
                var_update = var.assign(slow_update, use_locking=self._use_locking)
                slow_param.assign(slow_update, use_locking=self._use_locking)

        return var_update

    def get_config(self):
        config = super().get_config()
        config.update({
            "beta1": self._serialize_hyperparameter("beta1"),
            "beta2": self._serialize_hyperparameter("beta2"),
            "epsilon": self._serialize_hyperparameter("epsilon"),
            "l1_regularization_strength": self._serialize_hyperparameter("l1_regularization_strength"),
            "l2_regularization_strength": self._serialize_hyperparameter("l2_regularization_strength"),
            "weight_decay": self._serialize_hyperparameter("weight_decay"),
            "activation": self.activation,
            "momentum_type": self.momentum_type,
            "amsgrad": self.amsgrad,
            "clip_grad_norm": self.clip_grad_norm,
            "lookahead": self.lookahead,
            "lookahead_k": self.lookahead_k,
            "lookahead_alpha": self.lookahead_alpha,
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


# optimizer = NestYogi(
#     learning_rate=1e-3,
#     beta1=0.9,
#     beta2=0.999,
#     epsilon=1e-3,
#     l1_regularization_strength=0.0,
#     l2_regularization_strength=0.0,
#     activation='sign',
#     momentum_type='nesterov',
#     weight_decay=0.0,
#     amsgrad=False,
#     clip_grad_norm=None,
#     lookahead=False,
#     lookahead_k=5,
#     lookahead_alpha=0.5
# )