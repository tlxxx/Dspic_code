"""Runner registry."""
from src.runners.dspic_runner import DspicRunner

RUNNER_REGISTRY = {
    "dspic": DspicRunner,
}
