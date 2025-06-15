from stable_baselines3.common.callbacks import BaseCallback
from typing import Callable
import numpy as np


def linear_schedule(initial_value: float, min_value: float = 0.0) -> Callable[[float], float]:
    if not (0 <= min_value <= initial_value):
        raise ValueError(f"min_value {min_value} must be between 0 and initial_value {initial_value}")
    def func(progress_remaining: float) -> float:
        return min_value + (initial_value - min_value) * progress_remaining
    return func


def delayed_linear_schedule_with_min(
        initial_value: float,
        min_value: float = 0.0,
        decay_start_fraction: float = 0.0
) -> Callable[[float], float]:

    if not (0 <= min_value <= initial_value):
        raise ValueError(f"min_value ({min_value}) must be between 0 and initial_value ({initial_value})")

    if not (0.0 <= decay_start_fraction <= 1.0):
        raise ValueError(f"decay_start_fraction ({decay_start_fraction}) must be between 0.0 and 1.0")

    def func(progress_remaining: float):

        progress_so_far = 1.0 - progress_remaining

        if progress_so_far < decay_start_fraction:
            return initial_value
        else:
            if (1.0 - decay_start_fraction) <= 1e-8:  # Avoid division by zero if decay_start_fraction is 1.0
                return min_value

            progress_in_decay_phase = (progress_so_far - decay_start_fraction) / (1.0 - decay_start_fraction)
            progress_in_decay_phase = np.clip(progress_in_decay_phase, 0.0, 1.0)  # Ensure it's within [0,1]

            # Linearly interpolate from initial_value down to min_value over the decay phase
            return initial_value - (initial_value - min_value) * progress_in_decay_phase

    return func

class EntCoefSchedulerCallback(BaseCallback):

    def __init__(self, initial_ent_coef: float, final_ent_coef: float,
                 total_timesteps: int, decay_start_fraction: float = 0.5,
                 verbose: int = 0):
        super(EntCoefSchedulerCallback, self).__init__(verbose)
        self.initial_ent_coef = initial_ent_coef
        self.final_ent_coef = final_ent_coef
        self.total_timesteps = total_timesteps
        self.decay_start_fraction = decay_start_fraction
        self.schedule = delayed_linear_schedule_with_min(
            initial_value=initial_ent_coef,
            min_value=final_ent_coef,
            decay_start_fraction=decay_start_fraction
        )

    def _on_step(self) -> bool:
        progress_remaining = 1.0 - (self.num_timesteps / self.total_timesteps)
        progress_remaining = np.clip(progress_remaining, 0.0, 1.0)

        new_ent_coef = self.schedule(progress_remaining)

        if hasattr(self.model, 'ent_coef'):
            self.model.ent_coef = new_ent_coef
        elif hasattr(self.model, 'policy') and hasattr(self.model.policy, 'ent_coef'):
            self.model.policy.ent_coef = new_ent_coef
        elif self.verbose > 0:
            print(f"Warning: Could not set ent_coef. Model type: {type(self.model)}")

        self.logger.record("train/ent_coef_scheduled", new_ent_coef)
        return True
