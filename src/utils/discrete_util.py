import jax
import jax.numpy as jnp
from functools import partial

  
@partial(jax.jit, static_argnames=['eps'])
def onehot_from_logits(logits: jnp.ndarray, eps: float = 0.0) -> jnp.ndarray:
    argmax_indices = jnp.argmax(logits, axis=-1)
    argmax_actions_onehot = jax.nn.one_hot(argmax_indices, num_classes=logits.shape[-1])
    if eps == 0.0:
        return argmax_actions_onehot

    else:
        assert 0, "bad condition!"

def sample_gumbel(key, shape, eps=1e-20) -> jnp.ndarray:
    U = jax.random.uniform(key, shape, minval=0, maxval=1)
    return -jnp.log(-jnp.log(U + eps) + eps)

@partial(jax.jit, static_argnames=['temperature'])
def gumbel_softmax_sample(key: jax.random.PRNGKey, 
                          logits: jnp.ndarray, 
                          temperature: float) -> jnp.ndarray:
    gumbel_noise = sample_gumbel(key, logits.shape)
    y = logits + gumbel_noise
    return jax.nn.softmax(y / temperature, axis=1)

@partial(jax.jit, static_argnames=['hard', 'temperature'])
def gumbel_softmax(key: jax.random.PRNGKey,
                   logits: jnp.ndarray,
                   temperature: float = 1.0,
                   hard: bool = False,) -> jnp.ndarray:
    y_soft = gumbel_softmax_sample(key, logits, temperature)
    
    if hard:
        y_hard = onehot_from_logits(y_soft)
        y_hard_stop_gradient = jax.lax.stop_gradient(y_hard)
        y_soft_stop_gradient = jax.lax.stop_gradient(y_soft)
        y = y_hard_stop_gradient + y_soft - y_soft_stop_gradient
        return y
    else:
        return y_soft
