from pathlib import Path
from typing import Any, Dict

import gin

from .logger_base import LoggerBase


@gin.configurable()
class DummyLogger(LoggerBase):
    """
    A logger that does nothing. Can be used if one does not want to log anything.
    """

    def __init__(self):
        super().__init__("")

    def log_metrics(self, metrics: Dict[str, Any], prefix: str):
        pass

    def log_code(self, source_path: str | Path):
        pass

    def log_to_file(self, content: Any, name: str, type: str = "txt"):
        pass

    def log_config(self, config: Dict[str, Any]):
        pass

    def close(self):
        pass

    def restart(self):
        pass
