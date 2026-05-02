from absl import flags
from src.envs.smac.smac_logger import SMACLogger
from src.envs.smacv2.smacv2_logger import SMACv2Logger
from src.envs.mamujoco.mamujoco_logger import MAMuJoCoLogger
from src.envs.lbf.lab_logger import LBFLogger

FLAGS = flags.FLAGS
FLAGS(["train_sc.py"])

LOGGER_REGISTRY = {
    "smac": SMACLogger,
    "mamujoco": MAMuJoCoLogger,
    "smacv2": SMACv2Logger,
    "lbf": LBFLogger,
}
