import jax
import jax.numpy as jnp
from flax import linen as nn


class Encoder(nn.Module):
    """MLP encoder network."""
    latent_dim: int 

    def setup(self):
        self.hidden_size = 64

    @nn.compact
    def __call__(self, role):
        # 这里的输入是 n_agent * dim 大小的，直接投影成矩阵

        # attention_output = self.attention_layer(role)
        P = nn.Dense(self.hidden_size)(role)
        P = nn.relu(P)
        P = nn.Dense(self.latent_dim * self.latent_dim)(P)
        P = P.reshape(-1, self.latent_dim, self.latent_dim)

        norms = jnp.linalg.norm(P, axis=(1, 2), keepdims=True)
        P_normalized = P / norms
        
        return P_normalized
    
    def attention_layer(self, role):
        # Use MultiHeadDotProductAttention to capture the relationship between agents
        attention = nn.MultiHeadDotProductAttention(
            num_heads=4,  # Number of attention heads
            dtype=jnp.float32,
            qkv_features=self.hidden_size,  # Hidden dimension size for query, key, and value
            out_features=self.hidden_size  # Output dimension of attention layer
        )
        
        # Apply attention on the input role, assuming role shape is (n_agents, role_dim)
        return attention(role, role)

    @staticmethod
    @jax.jit
    def cal_encoder(latent, params, encoder_state):
        P = encoder_state.apply_fn({"params": params}, latent)
        return P