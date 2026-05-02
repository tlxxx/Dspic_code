"""Tools for HARL."""
import copy
import math
import jax


def get_active_func(name: str):
    name = (name or "").lower()
    if name == "sigmoid":
        return jax.nn.sigmoid
    elif name == "tanh":
        return jax.nn.tanh
    elif name == "relu":
        return jax.nn.relu
    elif name == "leaky_relu":
        return jax.nn.leaky_relu
    elif name == "selu":
        return jax.nn.selu
    elif name == "hardswish":
        return jax.nn.hard_swish
    elif name == "identity":
        return lambda x: x
    else:
        raise AssertionError("activation function not supported!")



def get_init_method(initialization_method: str):
    if initialization_method.endswith('_'):
        initialization_method = initialization_method[:-1]

    if hasattr(jax.nn.initializers, initialization_method):
        return getattr(jax.nn.initializers, initialization_method)
    else:
        raise ValueError(f"JAX 中未知的初始化方法: {initialization_method}")


# pylint: disable-next=invalid-name
def huber_loss(e, d):
    """Huber loss."""
    a = (abs(e) <= d).float()
    b = (abs(e) > d).float()
    return a * e**2 / 2 + b * d * (abs(e) - d / 2)


# pylint: disable-next=invalid-name
def mse_loss(e):
    """MSE loss."""
    return e**2 / 2


def update_linear_schedule(optimizer, epoch, total_num_epochs, initial_lr):
    """Decreases the learning rate linearly
    Args:
        optimizer: (torch.optim) optimizer
        epoch: (int) current epoch
        total_num_epochs: (int) total number of epochs
        initial_lr: (float) initial learning rate
    """
    learning_rate = initial_lr - (initial_lr * ((epoch - 1) / float(total_num_epochs)))
    for param_group in optimizer.param_groups:
        param_group["lr"] = learning_rate


def init(module, weight_init, bias_init, gain=1):
    """Init module.
    Args:
        module: (torch.nn) module
        weight_init: (torch.nn) weight init
        bias_init: (torch.nn) bias init
        gain: (float) gain
    Returns:
        module: (torch.nn) module
    """
    weight_init(module.weight.data, gain=gain)
    bias_init(module.bias.data)
    return module


def get_clones(module, N):
    """Clone module for N times."""
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


def get_grad_norm(parameters):
    """Get gradient norm."""
    sum_grad = 0
    for parameter in parameters:
        if parameter.grad is None:
            continue
        sum_grad += parameter.grad.norm() ** 2
    return math.sqrt(sum_grad)
