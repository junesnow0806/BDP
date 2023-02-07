import torch
from typing import Optional, Callable
from torch.optim import Optimizer
from opt_einsum.contract import contract
from opacus.optimizers.optimizer import DPOptimizer, _check_processed_flag, _mark_as_processed

class NSGDOptimizer(DPOptimizer):
    def __init__(
        self, 
        optimizer: Optimizer, 
        *, 
        regularizer: float,
        noise_multiplier: float, 
        max_grad_norm: float, 
        expected_batch_size: Optional[int], 
        loss_reduction: str = "mean", 
        generator=None, 
        secure_mode: bool = False
    ):
        super().__init__(
            optimizer=optimizer, 
            noise_multiplier=noise_multiplier, 
            max_grad_norm=max_grad_norm, 
            expected_batch_size=expected_batch_size, 
            loss_reduction=loss_reduction, 
            generator=generator, 
            secure_mode=secure_mode
        )
        self.regularizer = regularizer

    def regularize_and_accumulate(self):
        """
        Performs gradient regularizing.
        Stores regularized and aggregated gradients into `p.summed_grad```
        """

        if len(self.grad_samples[0]) == 0:
            # Empty batch
            # per_sample_clip_factor = torch.zeros((0,))
            per_sample_regularize_factor = torch.zeros((0,))
        else:
            per_param_norms = [
                g.reshape(len(g), -1).norm(2, dim=-1) for g in self.grad_samples
            ]
            per_sample_norms = torch.stack(per_param_norms, dim=1).norm(2, dim=1)
            # per_sample_clip_factor = (
            #     self.max_grad_norm / (per_sample_norms + 1e-6)
            # ).clamp(max=1.0)
            per_sample_regularize_factor = 1 / (per_sample_norms + self.regularizer)
        
        for p in self.params:
            _check_processed_flag(p.grad_sample)
            grad_sample = self._get_flat_grad_sample(p)
            # grad = contract("i,i...", per_sample_clip_factor, grad_sample)
            grad = contract("i,i...", per_sample_regularize_factor, grad_sample)

            if p.summed_grad is not None:
                p.summed_grad += grad
            else:
                p.summed_grad = grad

            _mark_as_processed(p.grad_sample)

    def pre_step(self, closure: Optional[Callable[[], float]] = None) -> Optional[float]:
        """
        Perform actions specific to ``DPOptimizer`` before calling
        underlying  ``optimizer.step()``

        Args:
            closure: A closure that reevaluates the model and
                returns the loss. Optional for most optimizers.
        """
        self.regularize_and_accumulate()
        if self._check_skip_next_step():
            self._is_last_step_skipped = True
            return False

        self.add_noise()
        self.scale_grad()

        if self.step_hook:
            self.step_hook(self)

        self._is_last_step_skipped = False
        return True
