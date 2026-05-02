"""Base class for off-policy algorithms."""

import jax
import optax
from copy import deepcopy
import jax.numpy as jnp
import gymnasium as gym
from src.utils.envs_tools import check
from src.utils.models_tools import update_linear_schedule


class OffPolicyBase:
    # def __init__(self, args, obs_space, act_space, device=torch.device("cpu")):
    #     pass
    args: dict
    obs_space: gym.Space
    act_space: gym.Space
    batch_size: int

    def lr_decay(self, step, steps):
        """Decay the actor and critic learning rates.
        Args:
            step: (int) current training step.
            steps: (int) total number of training steps.
        """
        new_lr = float(self.lr) - float(self.lr) * ((int(step) - 1) / float(steps))
        new_tx = optax.adam(learning_rate=new_lr)
        self.actor_state = self.actor_state.replace(tx=new_tx)
        # update_linear_schedule(self.actor_optimizer, step, steps, self.lr)

    # def get_actions(self, obs, randomness):
    #     pass
    #
    # def get_target_actions(self, obs):
    #     pass

    @staticmethod
    @jax.jit
    def soft_update(actor_state):
        """Soft update target actor."""
        actor_state = actor_state.replace(
            target_params=optax.incremental_update(actor_state.params, actor_state.target_params, self.polyak))
        return actor_state

    def save(self, save_dir, id):
        """Save the actor and target actor."""
        torch.save(
            self.actor.state_dict(), str(save_dir) + "/actor_agent" + str(id) + ".pt"
        )
        torch.save(
            self.target_actor.state_dict(),
            str(save_dir) + "/target_actor_agent" + str(id) + ".pt",
        )

    def restore(self, model_dir, id):
        """Restore the actor and target actor."""
        actor_state_dict = torch.load(str(model_dir) + "/actor_agent" + str(id) + ".pt")
        self.actor.load_state_dict(actor_state_dict)
        target_actor_state_dict = torch.load(
            str(model_dir) + "/target_actor_agent" + str(id) + ".pt"
        )
        self.target_actor.load_state_dict(target_actor_state_dict)

    def turn_on_grad(self):
        """Turn on grad for actor parameters."""
        for p in self.actor.parameters():
            p.requires_grad = True

    def turn_off_grad(self):
        """Turn off grad for actor parameters."""
        for p in self.actor.parameters():
            p.requires_grad = False
