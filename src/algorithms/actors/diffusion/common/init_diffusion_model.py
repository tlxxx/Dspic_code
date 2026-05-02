import distrax
import jax.numpy as jnp
import jax
import optax
from flax.training import train_state
from jax._src.nn.functions import softplus

from src.algorithms.actors.diffusion.common.learning_rate_scheduler import get_learning_rate_scheduler
from src.algorithms.actors.diffusion.common.models.pisgrad_net import PISGRADNet
from src.algorithms.actors.diffusion.common.models.ort_pisgard_net import OrthogonalPISGRADNet
from src.algorithms.actors.diffusion.common.models.input_encoder import Encoder
from src.algorithms.actors.diffusion.common.utils import flattened_traversal
from src.algorithms.actors.diffusion.common.scheduler import get_cosine_schedule, get_smac_schedule


def init_od(cfg, dim):

    def prior_sampler(params, key, n_samples):
        samples = distrax.MultivariateNormalDiag(params['params']['prior_mean'],
                                                 jnp.ones(dim) * jax.nn.softplus(params['params']['prior_std'])).sample(
            seed=key,
            sample_shape=(
                n_samples,))
        return samples if cfg["learn_prior"] else jax.lax.stop_gradient(samples)

    if cfg["learn_prior"]:
        def prior_log_prob(x, params):
            log_probs = distrax.MultivariateNormalDiag(params['params']['prior_mean'],
                                                       jnp.ones(dim) * jax.nn.softplus(
                                                           params['params']['prior_std'])).log_prob(x)
            return log_probs
    else:
        def prior_log_prob(x, params):
            log_probs = distrax.MultivariateNormalDiag(jnp.zeros(dim), jnp.ones(dim) * cfg["init_std"]).log_prob(x)
            return log_probs

    # dt_schedule = cfg["dt_schedule"]
    dt_schedule = get_cosine_schedule(total_steps=cfg["diff_steps"], min=0.001, s=0.008, pow=2)
    # dt_schedule = get_smac_schedule(total_steps=cfg["diff_steps"])

    def delta_t_fn(step, params):
        if cfg["per_step_dt"]:
            dt = params['params']['dt'][step.astype(int)] if cfg["learn_dt"] else jax.lax.stop_gradient(params['params']['dt'][step.astype(int)])
            return softplus(dt)
        else:
            dt = params['params']['dt'] if cfg["learn_dt"] else jax.lax.stop_gradient(params['params']['dt'])
            return softplus(dt) * dt_schedule(step)

    def friction_fn(step, params):
        friction = jax.nn.softplus(params['params']['friction'])
        return friction if cfg["learn_friction"] else jax.lax.stop_gradient(friction)

    def mass_fn(params):
        mass_std = jax.nn.softplus(params['params']['mass_std'])
        return mass_std if cfg["learn_mass_matrix"] else jax.lax.stop_gradient(mass_std)

    return prior_log_prob, prior_sampler, delta_t_fn, friction_fn, mass_fn


def init_langevin(cfg, prior_log_prob, target_log_prob):
    # Unimplemented!
    exit(114514)
    alg_cfg = cfg.algorithm
    dim = cfg.target.dim
    target_score_max_norm = alg_cfg.target_score_max_norm

    def get_betas(params):
        b = jax.nn.softplus(params['params']['betas'])
        b = jnp.cumsum(b) / jnp.sum(b)
        b = b if alg_cfg.learn_betas else jax.lax.stop_gradient(b)

        # Freeze first and last beta
        b = b.at[0].set(jax.lax.stop_gradient(b[0]))
        b = b.at[-1].set(jax.lax.stop_gradient(b[-1]))

        def get_beta(step):
            return b[jnp.array(step, int)]

        return get_beta

    def clip_target_score(target_score):
        target_score_norm = jnp.linalg.norm(target_score)
        target_score_clipped = jnp.where(target_score_norm > target_score_max_norm * jnp.sqrt(dim),
                                         (target_score_max_norm * jnp.sqrt(dim) * target_score) / target_score_norm,
                                         target_score)
        return target_score_clipped

    def langevin_fn(step, x, params):
        beta = get_betas(params)(step)
        target_score = jax.grad(lambda x: jnp.squeeze(target_log_prob(x)))(x)
        prior_score = jax.grad(lambda x: jnp.squeeze(prior_log_prob(x, params)))(x)
        if target_score_max_norm is None:
            return beta * target_score + (1 - beta) * prior_score, target_score

        else:
            target_score_clipped = clip_target_score(target_score)
            return beta * target_score_clipped + (1 - beta) * prior_score, target_score_clipped

    return langevin_fn


def init_model(key, params, cfg, dim, obs_dim, learn_forward=True, learn_backward=True, use_ort=False, latent_dim=1):
    # Define the model

    in_dim = 2 * dim if cfg["underdamped"] else dim

    key, key_gen = jax.random.split(key)
    if learn_forward:
        if use_ort:
            fwd_model = OrthogonalPISGRADNet(dim=dim, **cfg["score_model"])
            fwd_params = fwd_model.init(key, jnp.ones([cfg["batch_size"], in_dim]),
                                        jnp.ones(([cfg["batch_size"], obs_dim])),
                                        jnp.ones((cfg["batch_size"], cfg["score_model"]["latent_dim"], cfg["score_model"]["latent_dim"])), 
                                        jnp.ones([cfg["batch_size"], 1]),
                                        jnp.ones([cfg["batch_size"], dim]))
        else:
            fwd_model = PISGRADNet(dim=dim, **cfg["score_model"])
            fwd_params = fwd_model.init(key, jnp.ones([cfg["batch_size"], in_dim]),
                                        jnp.ones(([cfg["batch_size"], obs_dim])),
                                        jnp.ones([cfg["batch_size"], 1]),
                                        jnp.ones([cfg["batch_size"], dim]))
        params['params']['fwd_params'] = fwd_params
        fwd_apply_fn = fwd_model.apply
    else:
        fwd_apply_fn = None

    key, key_gen = jax.random.split(key_gen)
    if learn_backward:
        if use_ort:
            bwd_model = OrthogonalPISGRADNet(dim=dim, **cfg["score_model"])
            bwd_params = bwd_model.init(key, jnp.ones([cfg["batch_size"], in_dim]),
                                    jnp.ones(([cfg["batch_size"], obs_dim])),
                                    jnp.ones((cfg["batch_size"], cfg["score_model"]["latent_dim"], cfg["score_model"]["latent_dim"])), 
                                    jnp.ones([cfg["batch_size"], 1]),
                                    jnp.ones([cfg["batch_size"], dim]))
        else:
            bwd_model = PISGRADNet(dim=dim, **cfg["score_model"])
            bwd_params = bwd_model.init(key, jnp.ones([cfg["batch_size"], in_dim]),
                                        jnp.ones(([cfg["batch_size"], obs_dim])),
                                        jnp.ones([cfg["batch_size"], 1]),
                                        jnp.ones([cfg["batch_size"], dim]))
        params['params']['bwd_params'] = bwd_params
        bwd_apply_fn = bwd_model.apply
    else:
        bwd_apply_fn = None

    if cfg["use_step_size_scheduler"]:
        model_opt = optax.masked(optax.adam(get_learning_rate_scheduler(cfg, cfg["lr"]),
                                            b1=cfg["b1"]),
                                 mask=flattened_traversal(
                                     lambda path, _: ('fwd_params' in path) or ('bwd_params' in path)))
        betas_opt = optax.masked(optax.adam(get_learning_rate_scheduler(cfg, cfg["lr"]),
                                            b1=cfg["b1"]),
                                 mask=flattened_traversal(
                                     lambda path, _: ('fwd_params' not in path) and ('bwd_params' not in path)))
    else:
        model_opt = optax.masked(optax.adam(cfg["lr"], b1=cfg["b1"]),
                                 mask=flattened_traversal(
                                     lambda path, _: ('fwd_params' in path) or ('bwd_params' in path)))
        betas_opt = optax.masked(optax.adam(cfg["lr"], b1=cfg["b1"]),
                                 mask=flattened_traversal(
                                     lambda path, _: ('fwd_params' not in path) and ('bwd_params' not in path)))

    if cfg["do_actor_grad_clip"]:
        optimizer = optax.chain(optax.zero_nans(),
                                optax.clip(cfg["actor_grad_clip"]),
                                model_opt, betas_opt)
    else:
        optimizer = optax.chain(optax.zero_nans(),
                                model_opt, betas_opt)

    model_state = train_state.TrainState.create(apply_fn=(fwd_apply_fn, bwd_apply_fn), params=params, tx=optimizer)
    if use_ort:
        encoder = Encoder(cfg["score_model"]["latent_dim"])
        dummy_latent = jnp.zeros((cfg["batch_size"], latent_dim), jnp.float32)
        actor_key, key_gen = jax.random.split(key_gen, 2)
        encoder_params = encoder.init(actor_key, dummy_latent)['params']
        encoder_optx = optax.adam(cfg["lr"])
        encoder_state = train_state.TrainState.create(apply_fn=encoder.apply, params=encoder_params, tx=encoder_optx)
        return model_state, encoder_state

    else:
        return model_state
