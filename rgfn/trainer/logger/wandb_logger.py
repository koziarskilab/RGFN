import pickle
from pathlib import Path
from typing import Any, Dict, List

import gin
import wandb

from .logger_base import LoggerBase


@gin.configurable()
class WandbLogger(LoggerBase):
    """
    A logger that logs to wandb.

    Args:
        logdir: the directory to save the logs.
        project_name: the name of the project in wandb.
        experiment_name: the name of the experiment in wandb.
        kwargs: additional arguments to pass to wandb.init.
    """

    def __init__(
        self, logdir: str | Path, project_name: str, experiment_name: str, **kwargs: Dict[str, Any]
    ):
        super().__init__(logdir)
        self.project_name = project_name
        self.experiment_name = experiment_name
        self.kwargs = kwargs
        self.run = self._init_run()

    def _init_run(self):
        group_list = self.experiment_name.split("/")
        group = "/".join(group_list[:-1])
        return wandb.init(
            project=self.project_name,
            name=self.experiment_name,
            group=group,
            **self.kwargs,
        )

    def log_metrics(self, metrics: Dict[str, Any], prefix: str):
        metrics = {f"{prefix}/{k}": v for k, v in metrics.items()}
        self.run.log(metrics)

    def log_code(self, source_path: str | Path):
        self.run.log_code(root=str(source_path))

    def log_to_file(self, content: Any, name: str, type: str = "txt"):
        if type == "json":
            config_path = self.logdir / f"{name}.json"
        elif type == "txt":
            config_path = self.logdir / f"{name}.txt"
        elif type == "to_pickle":
            config_path = self.logdir / f"{name}.pkl"
        else:
            raise ValueError(f"Unknown type {type}")

        if type in ["json", "txt"]:
            with open(config_path, "w") as f:
                f.write(content)
        elif type == "to_pickle":
            with open(config_path, "wb") as f:
                pickle.dump(content, f)

        self.run.save(str(config_path))

    def log_config(self, config: Dict[str, Any]):
        self.run.config.update(config)

    def log_files(self, file_paths: List[Path | str]):
        for file_path in file_paths:
            self.run.save(str(file_path))

    def close(self):
        self.run.finish()

    def restart(self):
        self.close()
        self.run = self._init_run()
