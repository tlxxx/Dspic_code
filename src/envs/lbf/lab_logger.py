from src.common.base_logger import BaseLogger

class LBFLogger(BaseLogger):
    def get_task_name(self):
        return self.env_args["scenario"]
