import flax.linen as nn


class Flatten(nn.Module):
    @nn.compact
    def __call__(self, x):
        return x.reshape((x.shape[0], -1))
