"""Critic registry."""
from src.algorithms.critics.soft_twin_continuous_q_critic import (
    SoftTwinContinuousQCritic,
)
from src.algorithms.critics.VectorCritic import SoftVectorCritic

CRITIC_REGISTRY = {
    "dspic": SoftTwinContinuousQCritic,
    "dspic_crossq": SoftVectorCritic,
}
