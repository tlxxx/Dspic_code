import jax.numpy as jnp


def get_linear_schedule(total_steps, min=0.01):
    def linear_noise_schedule(step):
        t = (total_steps - step) / total_steps
        return (1. - t) * min + t

    return linear_noise_schedule


def get_cosine_schedule(total_steps, min=0.01, s=0.008, pow=2):
    def cosine_schedule(step):
        t = (total_steps - step) / total_steps
        offset = 1 + s
        return (1. - min) * jnp.cos(0.5 * jnp.pi * (offset - t) / offset) ** pow + min

    return cosine_schedule


def get_constant_schedule():
    def constant_schedule(step):
        return jnp.array(1.)

    return constant_schedule

def get_smac_schedule(total_steps):
    # === 核心修改：针对 T=4 的手动指定 Schedule ===
    # 这里的数值代表每一步 dt (时间步长) 的相对大小
    # 顺序对应 step 0 -> step 3
    # 我们希望：初期噪声大(大dt)，后期极其稳定(极小dt)
    
    # 如果你设置 diff_steps=4，这个数组长度必须也是4
    # 最后一步给 0.05 是为了防止 SMAC 的动作在最后关头跳变
    schedule_values = jnp.array([1.0, 0.7, 0.3, 0.05]) 
    
    def stepped_schedule(step):
        # JAX 中 step 是一个 tracer，需要转换成整数索引
        # 使用 clip 防止越界
        idx = jnp.clip(step, 0, total_steps - 1).astype(int)
        return schedule_values[idx]

    return stepped_schedule
