import jax
import jax.numpy as jnp

from src.algorithms.actors.diffusion.common.utils import init_dt
from src.algorithms.actors.diffusion.common.utils import inverse_softplus
from src.algorithms.actors.diffusion.common.diffusion_models import DiffusionModel
from src.algorithms.actors.diffusion.common.init_diffusion_model import init_od, init_langevin, init_model


def init_dis(key, cfg, dim, obs_dim, target=None, use_ort=False, latent_dim=1):

    params = {'params': {'betas': jnp.ones((cfg["diff_steps"],)),
                         'prior_mean': jnp.zeros((dim,)),
                         'prior_std': jnp.ones((dim,)) * inverse_softplus(cfg["init_std"]),
                         'mass_std': jnp.ones(1) * inverse_softplus(1.),
                         'dt': init_dt(cfg),
                         'friction': jnp.ones(dim) * inverse_softplus(cfg["friction"]) if cfg["per_dim_friction"] else jnp.ones(1) * inverse_softplus(cfg["friction"]),
                         }}

    prior_log_prob, prior_sampler, delta_t_fn, friction_fn, mass_fn = init_od(cfg, dim)
    if target is not None:
        langevin_fn = init_langevin(cfg, prior_log_prob, target.log_prob)

    def forward_model(step, x, obs, model_state, params, aux, P=None):
        langevin_vals = aux
        if use_ort:
            return model_state.apply_fn[0](params['params']['fwd_params'], x, obs, P, step,
                                       jax.lax.stop_gradient(langevin_vals))
        else:
            return model_state.apply_fn[0](params['params']['fwd_params'], x, obs, step,
                                       jax.lax.stop_gradient(langevin_vals))

    def backward_model(step, x, obs, model_state, params, aux, P=None):
        return jnp.zeros_like(x)

    def drift_fn(step, x, params):
        # 其实这里可能也要改，不过 dis 里用不到，所以就先这样吧
        if target is not None:
            if cfg["use_target_score"]:
                _, aux = langevin_fn(step, x, params)
            else:
                aux = None
        else:
            aux = None

        return jax.grad(prior_log_prob)(x, params), aux

    key, key_gen = jax.random.split(key)
    if use_ort:
        model_state, encoder_state = init_model(key, params, cfg, dim, obs_dim, learn_forward=True, learn_backward=False, use_ort=use_ort, latent_dim=latent_dim)
        
        def encoder_model(latent, encoder_state, params):
            return encoder_state.apply_fn[0](params, latent)

        return DiffusionModel(num_steps=cfg["diff_steps"],
                          forward_model=forward_model,
                          backward_model=backward_model,
                          drift_fn=drift_fn,
                          delta_t_fn=delta_t_fn,
                          friction_fn=friction_fn,
                          mass_fn=mass_fn,
                          prior_sampler=prior_sampler,
                          prior_log_prob=prior_log_prob,
                          input_encoder=encoder_model, 
                          ), model_state, encoder_state
    
    else:
        model_state = init_model(key, params, cfg, dim, obs_dim, learn_forward=True, learn_backward=False, use_ort=use_ort)

        return DiffusionModel(num_steps=cfg["diff_steps"],
                            forward_model=forward_model,
                            backward_model=backward_model,
                            drift_fn=drift_fn,
                            delta_t_fn=delta_t_fn,
                            friction_fn=friction_fn,
                            mass_fn=mass_fn,
                            prior_sampler=prior_sampler,
                            prior_log_prob=prior_log_prob,
                            ), model_state
