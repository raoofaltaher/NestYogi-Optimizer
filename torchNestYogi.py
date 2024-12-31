import math
import torch
from torch.optim.optimizer import Optimizer

class NestYogi(Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-3,
                 l1_regularization_strength=0.0, l2_regularization_strength=0.0,
                 initial_accumulator_value=1e-6, activation='sign', momentum_type='nesterov',
                 weight_decay=0, amsgrad=False, clip_grad_norm=None, lookahead=False, k=5, alpha=0.5):
        if isinstance(params, (list, tuple)) and len(params) > 0 and isinstance(params[0], dict):
            param_groups = params
        else:
            param_groups = [{'params': params}]

        for group in param_groups:
            group.setdefault('lr', lr)

        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= eps:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")
        if activation not in ['sign', 'tanh']:
            raise ValueError(f"Invalid activation function: {activation}")
        if momentum_type not in ['classical', 'nesterov']:
            raise ValueError(f"Invalid momentum type: {momentum_type}")

        defaults = dict(betas=betas, eps=eps,
                        l1_regularization_strength=l1_regularization_strength,
                        l2_regularization_strength=l2_regularization_strength,
                        initial_accumulator_value=initial_accumulator_value,
                        activation=activation, momentum_type=momentum_type,
                        weight_decay=weight_decay, amsgrad=amsgrad,
                        clip_grad_norm=clip_grad_norm, lookahead=lookahead, k=k, alpha=alpha)
        super(NestYogi, self).__init__(param_groups, defaults)

    def __setstate__(self, state):
        super(NestYogi, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('momentum_type', 'nesterov')
            group.setdefault('amsgrad', False)
            group.setdefault('lookahead', False)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad

                if group['clip_grad_norm']:
                    torch.nn.utils.clip_grad_norm_(p, group['clip_grad_norm'])

                if group['weight_decay'] != 0:
                    grad = grad.add(p, alpha=group['weight_decay'])

                state = self.state[p]
                if len(state) == 0:
                    state['step'] = 0
                    state['m'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    state['v'] = torch.full_like(p, group['initial_accumulator_value'], memory_format=torch.preserve_format)
                    if group['amsgrad']:
                        state['max_v'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    if group['lookahead']:
                        state['slow_param'] = torch.clone(p).detach()

                m, v = state['m'], state['v']
                beta1, beta2 = group['betas']
                momentum_type = group['momentum_type']
                activation = group['activation']

                state['step'] += 1
                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']

                if group['l1_regularization_strength'] > 0:
                    grad = grad.add(p, alpha=group['l1_regularization_strength']).sign_().mul_(grad.abs())
                if group['l2_regularization_strength'] > 0:
                    grad = grad.add(p, alpha=group['l2_regularization_strength'])

                # Compute m_t
                m.mul_(beta1).add_(grad, alpha=1 - beta1)

                # Compute v_t
                grad_squared = grad.pow(2)
                if activation == 'sign':
                    v_delta = (grad_squared - v).sign_()
                elif activation == 'tanh':
                    v_delta = torch.tanh(10 * (grad_squared - v))
                v.add_(v_delta.mul_(grad_squared), alpha=1 - beta2)

                if group['amsgrad']:
                    torch.max(state['max_v'], v, out=state['max_v'])
                    v_hat = state['max_v']
                else:
                    v_hat = v

                denom = (v_hat.sqrt() / math.sqrt(bias_correction2)).add_(group['eps'])

                step_size = group['lr'] / bias_correction1

                if momentum_type == 'classical':
                    p.addcdiv_(m, denom, value=-step_size)
                elif momentum_type == 'nesterov':
                    m_nesterov = m.mul(beta1).add(grad, alpha=1 - beta1)
                    p.addcdiv_(m_nesterov, denom, value=-step_size)

                if group['lookahead'] and state['step'] % group['k'] == 0:
                    slow_p = state['slow_param']
                    slow_p.add_(p - slow_p, alpha=group['alpha'])
                    p.copy_(slow_p)

        return loss

    def get_lr(self):
        return [group['lr'] for group in self.param_groups]
        
# Example usage:
# optimizer = NestYogi([
#     {'params': model.backbone.parameters(), 'lr': 0.01},
#     {'params': model.head.parameters(), 'lr': 0.001},
#     {'params': model.other_parts.parameters(), 'lr': 0.005}
# ],
#     betas=(0.9, 0.999),
#     eps=1e-4,
#     l1_regularization_strength=0.01,
#     l2_regularization_strength=0.005,
#     activation='tanh',
#     momentum_type='nesterov',
#     weight_decay=0.001,
#     amsgrad=True,
#     clip_grad_norm=1.0,
#     lookahead=True,
#     k=5,
#     alpha=0.5
# )