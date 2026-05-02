from typing import NamedTuple, Callable, Optional


class DiffusionModel(NamedTuple):
    num_steps: int
    forward_model: Callable
    backward_model: Callable
    drift_fn: Callable
    delta_t_fn: Callable
    friction_fn: Callable
    mass_fn: Callable
    prior_sampler: Callable
    prior_log_prob: Callable
    input_encoder: Optional[Callable] = None 