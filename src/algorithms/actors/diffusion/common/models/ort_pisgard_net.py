# import jax
# import jax.numpy as jnp
# from flax import linen as nn
# from typing import Optional

# class MultiHeadVectorAttentionGate(nn.Module):
#     """
#     一个接收两个向量并使用【多头】注意力门控机制进行融合的模块。
#     修复了 vmap 嵌套导致的形状推断错误。
#     """
#     embed_dim: int
#     num_heads: int = 4

#     @nn.compact
#     def __call__(self, x: jnp.ndarray, context: jnp.ndarray) -> jnp.ndarray:
#         # --- 1. 投影 x 和 context (不变) ---
#         x_proj = nn.Dense(self.embed_dim, name="x_projection")(x)
#         context_proj = nn.Dense(self.embed_dim, name="context_projection")(context)

#         # --- 2. 分割成多头 (不变) ---
#         head_dim = self.embed_dim // self.num_heads
#         assert self.embed_dim % self.num_heads == 0, "embed_dim must be divisible by num_heads"
        
#         x_heads = x_proj.reshape(x_proj.shape[:-1] + (self.num_heads, head_dim))
#         context_heads = context_proj.reshape(context_proj.shape[:-1] + (self.num_heads, head_dim))
        
#         # --- 3. 核心修改：手动实现门控计算，放弃 vmap ---
        
#         # a. 准备门控输入
#         #    gate_input_heads shape: (B, H, 2 * D_h)
#         gate_input_heads = jnp.concatenate([x_heads, context_heads], axis=-1)

#         # b. 定义门控 MLP 层。我们直接在这里定义它们，而不是在Sequential中
#         #    第一层
#         gate_dense1 = nn.Dense(features=head_dim, name="gate_dense_1")
#         #    第二层
#         gate_dense2 = nn.Dense(
#             features=head_dim, 
#             kernel_init=nn.initializers.zeros, 
#             bias_init=nn.initializers.constant(-2.0),
#             name="gate_dense_2"
#         )
        
#         # c. 手动应用 MLP
#         #    gate_input_heads shape: (B, H, 2 * D_h)
#         #    经过 gate_dense1 后, shape: (B, H, D_h)
#         hidden = gate_dense1(gate_input_heads)
#         hidden = nn.relu(hidden)
#         #    经过 gate_dense2 后, shape: (B, H, D_h)
#         attention_gate_heads = gate_dense2(hidden)
#         attention_gate_heads = nn.sigmoid(attention_gate_heads)

#         # --- 4. 应用门控 (不变) ---
#         fused_heads = (1 - attention_gate_heads) * x_heads + attention_gate_heads * context_heads
        
#         # --- 5. 合并多头 (不变) ---
#         fused_output = fused_heads.reshape(fused_heads.shape[:-2] + (self.embed_dim,))
        
#         # --- 6. 输出投影 (不变) ---
#         output_proj = nn.Dense(self.embed_dim, name="output_proj")
#         fused_output = output_proj(fused_output)

#         # --- 7. 残差和归一化 (不变) ---
#         output = nn.LayerNorm()(x_proj + fused_output)
        
#         return output

# # class OrthogonalPISGRADNet(nn.Module):
# #     dim: int
# #     use_target_score: True
# #     layer_norm: bool = False
# #     time_coder_out: int = 64
# #     latent_dim: int = 32

# #     num_layers: int = 2
# #     num_hid: int = 64
# #     outer_clip: float = 1e4
# #     inner_clip: float = 1e2

# #     weight_init: float = 1e-8
# #     bias_init: float = 0.

# #     def setup(self):
# #         self.timestep_phase = self.param('timestep_phase', nn.initializers.zeros_init(), (1, self.num_hid))
# #         self.timestep_coeff = jnp.linspace(start=0.1, stop=100, num=self.num_hid)[None]

# #         self.time_coder_state = nn.Sequential([
# #             nn.Dense(self.num_hid),
# #             nn.gelu,
# #             nn.Dense(self.time_coder_out),
# #         ])

# #         self.time_coder_grad = nn.Sequential([nn.Dense(self.num_hid)] + [nn.Sequential(
# #             [nn.gelu, nn.Dense(self.num_hid)]) for _ in range(self.num_layers)] + [
# #                                                  nn.Dense(self.dim, kernel_init=nn.initializers.constant(self.weight_init),
# #                                                           bias_init=nn.initializers.constant(self.bias_init))])

# #         self.latent_net = nn.Dense(self.latent_dim)

# #         self.input_proj = nn.Dense(self.num_hid)
# #         self.attention_block = MultiHeadVectorAttentionGate(embed_dim=self.num_hid)


# #         if self.layer_norm:
# #             self.state_time_net = nn.Sequential([nn.Sequential(
# #                 [nn.Dense(self.num_hid), nn.LayerNorm(), nn.gelu]) for _ in range(self.num_layers)] + [
# #                                                     nn.Dense(self.dim, kernel_init=nn.initializers.constant(1e-8),
# #                                                              bias_init=nn.initializers.zeros_init())])
# #         else:
# #             self.state_time_net = nn.Sequential([nn.Sequential(
# #                 [nn.Dense(self.num_hid), nn.gelu]) for _ in range(self.num_layers)] + [
# #                                                     nn.Dense(self.dim, kernel_init=nn.initializers.constant(1e-8),
# #                                                              bias_init=nn.initializers.zeros_init())])
# #         # self.state_time_net = nn.Sequential([nn.Dense(256), nn.relu, nn.Dense(256), nn.relu, nn.Dense(self.dim, kernel_init=nn.initializers.constant(1e-8),
# #         #                                                      bias_init=nn.initializers.zeros_init())])
        

# #     def get_fourier_features(self, timesteps):
# #         sin_embed_cond = jnp.sin(
# #             (self.timestep_coeff * timesteps) + self.timestep_phase
# #         )
# #         cos_embed_cond = jnp.cos(
# #             (self.timestep_coeff * timesteps) + self.timestep_phase
# #         )
# #         return jnp.concatenate([sin_embed_cond, cos_embed_cond], axis=-1)

# #     def __call__(self, input_array, obs_array, P, time_array, target_score=None):
# #         # print(1, input_array.shape, obs_array.shape, time_array.shape, P.shape)
# #         # CHECK：加不加特征提取
# #         latent_array = self.latent_net(obs_array)
# #         # latent_array = obs_array
# #         latent_array_pr = jnp.expand_dims(latent_array, axis=-1)
# #         latent_array_pr = jnp.matmul(P, latent_array_pr)
# #         latent_array_pr = jnp.squeeze(latent_array_pr, -1)

# #         time_array_emb = self.get_fourier_features(time_array)
# #         if len(input_array.shape) == 1:
# #             time_array_emb = time_array_emb[0]

# #         t_net1 = self.time_coder_state(time_array_emb)
# #         query = self.input_proj(input_array)
        
# #         context = jnp.concatenate((latent_array, latent_array_pr, t_net1), axis=-1)
# #         fused_representation = self.attention_block(query, context)
# #         out_state = self.state_time_net(fused_representation)

# #         # extended_input = jnp.concatenate((input_array, obs_array, latent_array, t_net1), axis=-1)
# #         # out_state = self.state_time_net(extended_input)
# #         out_state = jnp.clip(out_state, -self.outer_clip, self.outer_clip)
# #         if self.use_target_score:
# #             t_net2 = self.time_coder_grad(time_array_emb)
# #             target_score = jnp.clip(target_score, -self.inner_clip, self.inner_clip)
# #             return out_state + t_net2 * target_score
# #         else:
# #             return out_state

# class OrthogonalPISGRADNet(nn.Module):
#     dim: int
#     use_target_score: bool = True
#     layer_norm: bool = False
#     time_coder_out: int = 64
#     latent_dim: int = 32

#     # === 修改点 1: 增加深度参数 ===
#     num_blocks: int = 4  # 比如设为 4 层
#     num_layers: int = 2
#     num_hid: int = 128
#     outer_clip: float = 1e4
#     inner_clip: float = 1e2

#     weight_init: float = 1e-8
#     bias_init: float = 0.

#     def setup(self):
#         # ... (时间编码部分保持不变) ...
#         self.timestep_phase = self.param('timestep_phase', nn.initializers.zeros_init(), (1, self.num_hid))
#         self.timestep_coeff = jnp.linspace(start=0.1, stop=100, num=self.num_hid)[None]

#         self.time_coder_state = nn.Sequential([
#             nn.Dense(self.num_hid),
#             nn.gelu,
#             nn.Dense(self.time_coder_out),
#         ])
        
#         # ... (梯度引导部分保持不变) ...
#         self.time_coder_grad = nn.Sequential([nn.Dense(self.num_hid)] + [nn.Sequential(
#             [nn.gelu, nn.Dense(self.num_hid)]) for _ in range(self.num_layers)] + [
#                                                  nn.Dense(self.dim, kernel_init=nn.initializers.constant(self.weight_init),
#                                                           bias_init=nn.initializers.constant(self.bias_init))])

#         self.latent_net = nn.Dense(self.latent_dim)
#         self.input_proj = nn.Dense(self.num_hid)

#         # === 修改点 2: 定义多层 Attention 和 MLP ===
#         # 不再是一个 self.attention_block，而是一个列表
#         num_hids = [128, 64, 64, 128]
#         self.attention_stack = [
#             MultiHeadVectorAttentionGate(embed_dim=num_hids[_]) 
#             for _ in range(self.num_blocks)
#         ]
        
#         # 为了配合 Attention，每层后面通常跟一个简单的 MLP (FFN) 做特征混合
#         # 结构: LayerNorm -> Dense -> Gelu -> Dense
#         self.ffn_stack = [
#             nn.Sequential([
#                 nn.Dense(num_hids[_]),
#                 nn.gelu,
#                 nn.Dense(num_hids[_])
#             ])
#             for _ in range(self.num_blocks)
#         ]

#         # ... (输出层保持不变) ...
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
#         # ... (保持不变) ...
#         sin_embed_cond = jnp.sin((self.timestep_coeff * timesteps) + self.timestep_phase)
#         cos_embed_cond = jnp.cos((self.timestep_coeff * timesteps) + self.timestep_phase)
#         return jnp.concatenate([sin_embed_cond, cos_embed_cond], axis=-1)

#     def __call__(self, input_array, obs_array, P, time_array, target_score=None):
#         # ... (前处理代码保持不变) ...
#         latent_array = self.latent_net(obs_array)
#         latent_array_pr = jnp.expand_dims(latent_array, axis=-1)
#         latent_array_pr = jnp.matmul(P, latent_array_pr)
#         latent_array_pr = jnp.squeeze(latent_array_pr, -1)

#         time_array_emb = self.get_fourier_features(time_array)
#         if len(input_array.shape) == 1:
#             time_array_emb = time_array_emb[0]

#         t_net1 = self.time_coder_state(time_array_emb)
        
#         # === 修改点 3: 循环执行多层 ===
#         # 1. 初始 Query 投影
#         h = self.input_proj(input_array)
        
#         # 2. Context 拼接
#         # 建议加上 LayerNorm 保证输入 context 的稳定性
#         context = jnp.concatenate((obs_array, latent_array_pr, t_net1), axis=-1)
        
#         # 3. 核心循环：Attention -> FFN -> Attention ...
#         for attn_layer, ffn_layer in zip(self.attention_stack, self.ffn_stack):
#             # Part A: Cross Attention (你的 Gate 模块内部已经有 Residual + LayerNorm)
#             h = attn_layer(h, context)
            
#             # Part B: FeedForward Network (模拟 Transformer Block)
#             # 加上 Residual Connection: x = x + FFN(x)
#             h_res = ffn_layer(h)
#             h = h + h_res
            
#         fused_representation = h
        
#         # ... (输出部分保持不变) ...
#         out_state = self.state_time_net(fused_representation)
#         out_state = jnp.clip(out_state, -self.outer_clip, self.outer_clip)
        
#         if self.use_target_score:
#             t_net2 = self.time_coder_grad(time_array_emb)
#             target_score = jnp.clip(target_score, -self.inner_clip, self.inner_clip)
#             return out_state + t_net2 * target_score
#         else:
#             return out_state


import jax
import jax.numpy as jnp
from flax import linen as nn
from typing import Optional

class MultiHeadVectorAttentionGate(nn.Module):
    """
    一个接收两个向量并使用【多头】注意力门控机制进行融合的模块。
    修复了 vmap 嵌套导致的形状推断错误。
    """
    embed_dim: int
    num_heads: int = 4

    @nn.compact
    def __call__(self, x: jnp.ndarray, context: jnp.ndarray) -> jnp.ndarray:
        # --- 1. 投影 x 和 context (不变) ---
        x_proj = nn.Dense(self.embed_dim, name="x_projection")(x)
        context_proj = nn.Dense(self.embed_dim, name="context_projection")(context)

        # --- 2. 分割成多头 (不变) ---
        head_dim = self.embed_dim // self.num_heads
        assert self.embed_dim % self.num_heads == 0, "embed_dim must be divisible by num_heads"
        
        x_heads = x_proj.reshape(x_proj.shape[:-1] + (self.num_heads, head_dim))
        context_heads = context_proj.reshape(context_proj.shape[:-1] + (self.num_heads, head_dim))
        
        # --- 3. 核心修改：手动实现门控计算，放弃 vmap ---
        
        # a. 准备门控输入
        #    gate_input_heads shape: (B, H, 2 * D_h)
        gate_input_heads = jnp.concatenate([x_heads, context_heads], axis=-1)

        # b. 定义门控 MLP 层。我们直接在这里定义它们，而不是在Sequential中
        #    第一层
        gate_dense1 = nn.Dense(features=head_dim, name="gate_dense_1")
        #    第二层
        gate_dense2 = nn.Dense(
            features=head_dim, 
            kernel_init=nn.initializers.zeros, 
            name="gate_dense_2"
        )
        
        # c. 手动应用 MLP
        #    gate_input_heads shape: (B, H, 2 * D_h)
        #    经过 gate_dense1 后, shape: (B, H, D_h)
        hidden = gate_dense1(gate_input_heads)
        hidden = nn.relu(hidden)
        #    经过 gate_dense2 后, shape: (B, H, D_h)
        attention_gate_heads = gate_dense2(hidden)
        attention_gate_heads = nn.sigmoid(attention_gate_heads)

        # --- 4. 应用门控 (不变) ---
        fused_heads = (1 - attention_gate_heads) * x_heads + attention_gate_heads * context_heads
        
        # --- 5. 合并多头 (不变) ---
        fused_output = fused_heads.reshape(fused_heads.shape[:-2] + (self.embed_dim,))
        
        # --- 6. 输出投影 (不变) ---
        output_proj = nn.Dense(self.embed_dim, name="output_proj")
        fused_output = output_proj(fused_output)

        # --- 7. 残差和归一化 (不变) ---
        output = nn.LayerNorm()(x_proj + fused_output)
        
        return output

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

        self.input_proj = nn.Dense(self.num_hid)
        self.attention_block = MultiHeadVectorAttentionGate(embed_dim=self.num_hid)


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
        # if self.layer_norm:
        #     self.state_time_net = nn.Sequential([nn.Sequential(
        #         [nn.Dense(self.num_hid), nn.LayerNorm(), nn.gelu]) for _ in range(self.num_layers)] + [
        #                                             nn.Dense(self.dim)])
        # else:
        #     self.state_time_net = nn.Sequential([nn.Sequential(
        #         [nn.Dense(self.num_hid), nn.gelu]) for _ in range(self.num_layers)] + [
        #                                             nn.Dense(self.dim)])

    def get_fourier_features(self, timesteps):
        sin_embed_cond = jnp.sin(
            (self.timestep_coeff * timesteps) + self.timestep_phase
        )
        cos_embed_cond = jnp.cos(
            (self.timestep_coeff * timesteps) + self.timestep_phase
        )
        return jnp.concatenate([sin_embed_cond, cos_embed_cond], axis=-1)

    def __call__(self, input_array, obs_array, P, time_array, target_score=None):
        # print(1, input_array.shape, obs_array.shape, time_array.shape, P.shape)
        # CHECK：加不加特征提取
        latent_array = self.latent_net(obs_array)
        # latent_array = obs_array
        latent_array = jnp.expand_dims(latent_array, axis=-1)
        latent_array = jnp.matmul(P, latent_array)
        latent_array = jnp.squeeze(latent_array, -1)

        time_array_emb = self.get_fourier_features(time_array)
        if len(input_array.shape) == 1:
            time_array_emb = time_array_emb[0]

        t_net1 = self.time_coder_state(time_array_emb)
        query = self.input_proj(input_array)
        # print(2, obs_array.shape, latent_array.shape, t_net1.shape)
        # assert 0
        context = jnp.concatenate((obs_array, latent_array, t_net1), axis=-1)
        fused_representation = self.attention_block(query, context)
        # fused_representation = jnp.concatenate((query, context), axis=-1)
        out_state = self.state_time_net(fused_representation)

        # extended_input = jnp.concatenate((input_array, obs_array, latent_array, t_net1), axis=-1)
        # out_state = self.state_time_net(extended_input)
        out_state = jnp.clip(out_state, -self.outer_clip, self.outer_clip)
        if self.use_target_score:
            t_net2 = self.time_coder_grad(time_array_emb)
            target_score = jnp.clip(target_score, -self.inner_clip, self.inner_clip)
            return out_state + t_net2 * target_score
        else:
            return out_state


# import jax
# import jax.numpy as jnp
# from flax import linen as nn
# from typing import Optional

# class MultiHeadVectorAttentionGate(nn.Module):
#     """
#     一个接收两个向量并使用【多头】注意力门控机制进行融合的模块。
#     修复了 vmap 嵌套导致的形状推断错误。
#     """
#     embed_dim: int
#     num_heads: int = 4

#     @nn.compact
#     def __call__(self, x: jnp.ndarray, context: jnp.ndarray) -> jnp.ndarray:
#         # --- 1. 投影 x 和 context (不变) ---
#         x_proj = nn.Dense(self.embed_dim, name="x_projection")(x)
#         context_proj = nn.Dense(self.embed_dim, name="context_projection")(context)

#         # --- 2. 分割成多头 (不变) ---
#         head_dim = self.embed_dim // self.num_heads
#         assert self.embed_dim % self.num_heads == 0, "embed_dim must be divisible by num_heads"
        
#         x_heads = x_proj.reshape(x_proj.shape[:-1] + (self.num_heads, head_dim))
#         context_heads = context_proj.reshape(context_proj.shape[:-1] + (self.num_heads, head_dim))
        
#         # --- 3. 核心修改：手动实现门控计算，放弃 vmap ---
        
#         # a. 准备门控输入
#         #    gate_input_heads shape: (B, H, 2 * D_h)
#         gate_input_heads = jnp.concatenate([x_heads, context_heads], axis=-1)

#         # b. 定义门控 MLP 层。我们直接在这里定义它们，而不是在Sequential中
#         #    第一层
#         gate_dense1 = nn.Dense(features=head_dim, name="gate_dense_1")
#         #    第二层
#         gate_dense2 = nn.Dense(
#             features=head_dim, 
#             kernel_init=nn.initializers.zeros, 
#             name="gate_dense_2"
#         )
        
#         # c. 手动应用 MLP
#         #    gate_input_heads shape: (B, H, 2 * D_h)
#         #    经过 gate_dense1 后, shape: (B, H, D_h)
#         hidden = gate_dense1(gate_input_heads)
#         hidden = nn.relu(hidden)
#         #    经过 gate_dense2 后, shape: (B, H, D_h)
#         attention_gate_heads = gate_dense2(hidden)
#         attention_gate_heads = nn.sigmoid(attention_gate_heads)

#         # --- 4. 应用门控 (不变) ---
#         fused_heads = (1 - attention_gate_heads) * x_heads + attention_gate_heads * context_heads
        
#         # --- 5. 合并多头 (不变) ---
#         fused_output = fused_heads.reshape(fused_heads.shape[:-2] + (self.embed_dim,))
        
#         # --- 6. 输出投影 (不变) ---
#         output_proj = nn.Dense(self.embed_dim, name="output_proj")
#         fused_output = output_proj(fused_output)

#         # --- 7. 残差和归一化 (不变) ---
#         output = nn.LayerNorm()(x_proj + fused_output)
        
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
#         self.attention_block = MultiHeadVectorAttentionGate(embed_dim=self.num_hid)


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
