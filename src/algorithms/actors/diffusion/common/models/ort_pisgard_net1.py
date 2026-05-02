import jax.numpy as jnp
from flax import linen as nn


class OrthogonalPISGRADNet(nn.Module):
    dim: int
    use_target_score: True
    layer_norm: bool = False
    time_coder_out: int = 64
    latent_dim: int = 32

    num_layers: int = 2
    num_hid: int = 64
    outer_clip: float = 1e4
    inner_clip: float = 1e2

    weight_init: float = 1e-8
    bias_init: float = 0.

    def setup(self):
        self.timestep_phase = self.param('timestep_phase', nn.initializers.zeros_init(), (1, self.num_hid))
        self.timestep_coeff = jnp.linspace(start=0.1, stop=100, num=self.num_hid)[None]

        self.time_coder_state = nn.Sequential([
            nn.Dense(self.num_hid),
            nn.gelu,
            nn.Dense(self.time_coder_out),
        ])

        self.time_coder_grad = nn.Sequential([nn.Dense(self.num_hid)] + [nn.Sequential(
            [nn.gelu, nn.Dense(self.num_hid)]) for _ in range(self.num_layers)] + [
                                                 nn.Dense(self.dim, kernel_init=nn.initializers.constant(self.weight_init),
                                                          bias_init=nn.initializers.constant(self.bias_init))])

        self.latent_net = nn.Dense(self.latent_dim)

        if self.layer_norm:
            self.state_time_net = nn.Sequential([nn.Sequential(
                [nn.Dense(self.num_hid), nn.LayerNorm(), nn.gelu]) for _ in range(self.num_layers)] + [
                                                    nn.Dense(self.dim, kernel_init=nn.initializers.constant(1e-8),
                                                             bias_init=nn.initializers.zeros_init())])
        else:
            self.state_time_net = nn.Sequential([nn.Sequential(
                [nn.Dense(self.num_hid), nn.gelu]) for _ in range(self.num_layers)] + [
                                                    nn.Dense(self.dim, kernel_init=nn.initializers.constant(1e-8),
                                                             bias_init=nn.initializers.zeros_init())])

    def get_fourier_features(self, timesteps):
        sin_embed_cond = jnp.sin(
            (self.timestep_coeff * timesteps) + self.timestep_phase
        )
        cos_embed_cond = jnp.cos(
            (self.timestep_coeff * timesteps) + self.timestep_phase
        )
        return jnp.concatenate([sin_embed_cond, cos_embed_cond], axis=-1)

    def __call__(self, input_array, obs_array, P, time_array, target_score=None):
        # CHECK：加不加特征提取
        latent_array = self.latent_net(obs_array)
        latent_array = jnp.expand_dims(latent_array, axis=-1)
        latent_array = jnp.matmul(P, latent_array)
        latent_array = jnp.squeeze(latent_array, -1)

        time_array_emb = self.get_fourier_features(time_array)
        if len(input_array.shape) == 1:
            time_array_emb = time_array_emb[0]

        t_net1 = self.time_coder_state(time_array_emb)

        extended_input = jnp.concatenate((input_array, obs_array, latent_array, t_net1), axis=-1)
        out_state = self.state_time_net(extended_input)
        out_state = jnp.clip(out_state, -self.outer_clip, self.outer_clip)
        if self.use_target_score:
            t_net2 = self.time_coder_grad(time_array_emb)
            target_score = jnp.clip(target_score, -self.inner_clip, self.inner_clip)
            return out_state + t_net2 * target_score
        else:
            return out_state


# import jax.numpy as jnp
# from flax import linen as nn


# class VectorAttentionGate(nn.Module):
#     """
#     一个接收两个向量并使用注意力门控机制进行融合的模块。
#     """
#     embed_dim: int

#     @nn.compact
#     def __call__(self, x: jnp.ndarray, context: jnp.ndarray) -> jnp.ndarray:
#         """
#         Args:
#             x: 第一个输入向量, shape (batch, x_dim)
#             context: 第二个输入向量, shape (batch, context_dim)
#         Returns:
#             融合后的向量, shape (batch, embed_dim)
#         """
#         # 1. 将 x 和 context 投影到相同的维度
#         x_proj = nn.Dense(self.embed_dim, name="x_projection")(x)
#         context_proj = nn.Dense(self.embed_dim, name="context_projection")(context)

#         # 2. 计算注意力门控分数 (Attention Gate Score)
#         #    将两者拼接，然后通过一个MLP来预测一个门控值 alpha
#         gate_input = jnp.concatenate([x_proj, context_proj], axis=-1)
#         gate_mlp = nn.Sequential([
#             nn.Dense(self.embed_dim),
#             nn.relu,
#             nn.Dense(self.embed_dim, kernel_init=nn.initializers.zeros)  # 初始化为0，让初始时更稳定
#         ], name="gate_mlp")

#         # 使用 sigmoid 将门控值缩放到 (0, 1) 区间
#         attention_gate = nn.sigmoid(gate_mlp(gate_input))

#         # 3. 应用门控
#         #    用 alpha 融合 x 和 context
#         #    这是一种常见的融合方式，类似于 GRU 或 LSTM 中的门
#         fused_output = (1 - attention_gate) * x_proj + attention_gate * context_proj

#         # 4. (可选) 添加残差连接和层归一化
#         output = nn.LayerNorm()(x_proj + fused_output)  # 残差连接到原始的 x_proj 上

#         return output

# class OrthogonalPISGRADNet(nn.Module):
#     dim: int
#     use_target_score: True
#     layer_norm: bool = False
#     time_coder_out: int = 64
#     latent_dim: int = 32

#     num_layers: int = 2
#     num_hid: int = 64
#     outer_clip: float = 1e4
#     inner_clip: float = 1e2

#     weight_init: float = 1e-8
#     bias_init: float = 0.

#     def setup(self):
#         self.timestep_phase = self.param('timestep_phase', nn.initializers.zeros_init(), (1, self.num_hid))
#         self.timestep_coeff = jnp.linspace(start=0.1, stop=100, num=self.num_hid)[None]

#         self.time_coder_state = nn.Sequential([
#             nn.Dense(self.num_hid),
#             nn.gelu,
#             nn.Dense(self.time_coder_out),
#         ])

#         self.time_coder_grad = nn.Sequential([nn.Dense(self.num_hid)] + [nn.Sequential(
#             [nn.gelu, nn.Dense(self.num_hid)]) for _ in range(self.num_layers)] + [
#                                                  nn.Dense(self.dim, kernel_init=nn.initializers.constant(self.weight_init),
#                                                           bias_init=nn.initializers.constant(self.bias_init))])

#         self.latent_net = nn.Dense(self.latent_dim)

#         self.input_proj = nn.Dense(self.num_hid)
#         self.attention_block = VectorAttentionGate(embed_dim=self.num_hid)


#         if self.layer_norm:
#             self.state_time_net = nn.Sequential([nn.Sequential(
#                 [nn.Dense(self.num_hid), nn.LayerNorm(), nn.gelu]) for _ in range(self.num_layers)] + [
#                                                     nn.Dense(self.dim, kernel_init=nn.initializers.constant(1e-8),
#                                                              bias_init=nn.initializers.zeros_init())])
#         else:
#             self.state_time_net = nn.Sequential([nn.Sequential(
#                 [nn.Dense(self.num_hid), nn.gelu]) for _ in range(self.num_layers)] + [
#                                                     nn.Dense(self.dim, kernel_init=nn.initializers.constant(1e-8),
#                                                              bias_init=nn.initializers.zeros_init())])

#     def get_fourier_features(self, timesteps):
#         sin_embed_cond = jnp.sin(
#             (self.timestep_coeff * timesteps) + self.timestep_phase
#         )
#         cos_embed_cond = jnp.cos(
#             (self.timestep_coeff * timesteps) + self.timestep_phase
#         )
#         return jnp.concatenate([sin_embed_cond, cos_embed_cond], axis=-1)

#     def __call__(self, input_array, obs_array, P, time_array, target_score=None):
#         # CHECK：加不加特征提取
#         latent_array = self.latent_net(obs_array)
#         latent_array = jnp.expand_dims(latent_array, axis=-1)
#         latent_array = jnp.matmul(P, latent_array)
#         latent_array = jnp.squeeze(latent_array, -1)

#         time_array_emb = self.get_fourier_features(time_array)
#         if len(input_array.shape) == 1:
#             time_array_emb = time_array_emb[0]

#         t_net1 = self.time_coder_state(time_array_emb)
#         query = self.input_proj(input_array)
#         context = jnp.concatenate((obs_array, latent_array, t_net1), axis=-1)
#         fused_representation = self.attention_block(query, context)
#         out_state = self.state_time_net(fused_representation)

#         # extended_input = jnp.concatenate((input_array, obs_array, latent_array, t_net1), axis=-1)
#         # out_state = self.state_time_net(extended_input)
#         out_state = jnp.clip(out_state, -self.outer_clip, self.outer_clip)
#         if self.use_target_score:
#             t_net2 = self.time_coder_grad(time_array_emb)
#             target_score = jnp.clip(target_score, -self.inner_clip, self.inner_clip)
#             return out_state + t_net2 * target_score
#         else:
#             return out_state
