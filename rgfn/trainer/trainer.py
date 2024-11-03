from pathlib import Path
from typing import Any, Dict, Generic, List, Literal, Sequence

import gin
import torch
from tqdm import tqdm

from rgfn.api.trajectories import Trajectories
from rgfn.trainer.artifacts.artifacts_base import ArtifactsBase, ArtifactsList
from rgfn.trainer.logger.logger_base import LoggerBase
from rgfn.trainer.metrics.metric_base import MetricsBase, MetricsList
from rgfn.utils.helpers import dict_mean, infer_metric_direction

from ..api.objective_base import ObjectiveBase
from ..api.replay_buffer_base import ReplayBufferBase
from ..api.sampler_base import SamplerBase
from ..api.training_hooks_mixin import TrainingHooksMixin
from ..api.type_variables import TAction, TActionSpace, TState
from .logger.dummy_logger import DummyLogger
from .optimizers.lr_scheduler import LRScheduler
from .optimizers.optimizer_base import OptimizerBase
from .trajectory_filters.trajectory_filter_base import (
    IdentityTrajectoryFilter,
    TrajectoryFilterBase,
)


@gin.configurable()
class Trainer(Generic[TState, TActionSpace, TAction], TrainingHooksMixin):
    """
    The class to train a model using the training loop.
    """

    def __init__(
        self,
        *,
        run_dir: str | Path,
        logger: LoggerBase | None,
        train_forward_sampler: SamplerBase[TState, TActionSpace, TAction] | None,
        train_backward_sampler: SamplerBase[TState, TActionSpace, TAction] | None = None,
        train_replay_buffer: ReplayBufferBase[TState, TActionSpace, TAction] | None,
        train_forward_n_trajectories: int,
        train_backward_n_trajectories: int = 0,
        train_replay_n_trajectories: int,
        train_batch_size: int = -1,
        train_metrics: Sequence[MetricsBase] = (),
        train_artifacts: Sequence[ArtifactsBase] = (),
        valid_sampler: SamplerBase[TState, TActionSpace, TAction] | None = None,
        valid_n_trajectories: int = 0,
        valid_batch_size: int = -1,
        valid_every_n_iterations: int = 10,
        valid_metrics: Sequence[MetricsBase] = (),
        valid_artifacts: Sequence[ArtifactsBase] = (),
        objective: ObjectiveBase[TState, TActionSpace, TAction],
        optimizer: OptimizerBase,
        gradient_clipping_norm: float = 10.0,
        lr_scheduler: LRScheduler | None = None,
        n_iterations: int,
        checkpoint_mode: Literal["none", "last", "best"] = "best",
        best_metric: str = "loss",
        metric_direction: Literal["auto", "min", "max"] = "auto",
        resume_path: str | Path | None = None,
        sanity_check_evaluation: bool = False,
        device: str = "auto",
        trajectory_filter: TrajectoryFilterBase[TState, TActionSpace, TAction] | None = None,
    ):
        """
        Args:
            run_dir: base directory to save the logs and checkpoints.
            logger: a logger object to log the metrics and artifacts.
            train_forward_sampler: a forward sampler to sample the forward trajectories.
            train_backward_sampler: a backward sampler to sample the backward trajectories.
            train_replay_buffer: a replay buffer to store and re-sample the trajectories.
            train_forward_n_trajectories: number of forward trajectories to sample in each iteration.
            train_backward_n_trajectories: number of backward trajectories to sample in each iteration.
            train_replay_n_trajectories: number of trajectories to sample from the replay buffer in each iteration.
            train_batch_size: batch size to use in training.
            train_metrics: a list of metrics to compute on training trajectories.
            train_artifacts: a list of artifacts to compute on training trajectories.
            valid_sampler: a separate sampler to sample the validation trajectories.
            valid_n_trajectories: number of validation trajectories to sample in each validation step.
            valid_batch_size: batch size to use in validation.
            valid_every_n_iterations: number of iterations after which to perform validation.
            valid_metrics: a list of metrics to compute on validation trajectories.
            valid_artifacts: a list of artifacts to compute on validation trajectories.
            objective: the objective to optimize.
            optimizer: the optimizer to use for optimization.
            gradient_clipping_norm: the norm to which to clip the gradients.
            lr_scheduler: the learning rate scheduler to use.
            n_iterations: number of iterations to train.
            checkpoint_mode: whether to save the last checkpoint, the best checkpoint, or none.
            best_metric: the metric to use for determining the best checkpoint.
            metric_direction: the direction of the metric to use for determining the best checkpoint.
            resume_path: path to the checkpoint to resume training from.
            sanity_check_evaluation: whether to perform a sanity check after one iteration of training.
            device: the device to use for training.
        """
        assert metric_direction in ("auto", "min", "max")
        self.device = "cpu"
        self.run_dir = Path(run_dir)
        self.train_forward_sampler = train_forward_sampler
        self.train_backward_sampler = train_backward_sampler
        self.train_replay_buffer = train_replay_buffer
        self.train_metrics = MetricsList(train_metrics)
        self.train_artifacts = ArtifactsList(train_artifacts)
        self.valid_sampler = valid_sampler
        self.valid_metrics = MetricsList(valid_metrics)
        self.valid_artifacts = ArtifactsList(valid_artifacts)
        self.objective = objective
        self.logger = logger if logger is not None else DummyLogger()
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.n_iterations = n_iterations
        self.train_forward_n_trajectories = train_forward_n_trajectories
        self.train_backward_n_trajectories = train_backward_n_trajectories
        self.train_replay_n_trajectories = train_replay_n_trajectories
        self.train_batch_size = train_batch_size
        self.valid_n_trajectories = valid_n_trajectories
        self.valid_every_n_iterations = valid_every_n_iterations
        self.valid_batch_size = valid_batch_size
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.checkpoint_mode = checkpoint_mode
        self.best_metric = best_metric
        self.gradient_clipping_norm = gradient_clipping_norm
        self.metric_direction = (
            infer_metric_direction(self.best_metric)
            if metric_direction == "auto"
            else metric_direction
        )
        self.best_valid_metrics: Dict[str, float] = {}

        self.optimizer.initialize(model=self.objective)
        if self.lr_scheduler:
            self.lr_scheduler.initialize(optimizer=self.optimizer.optimizer)

        self.set_device(device)
        self.start_iteration = 0
        if resume_path is not None:
            resume_path = Path(resume_path)
            checkpoint_dict = torch.load(resume_path, map_location=device)
            self.objective.load_state_dict(checkpoint_dict["model"])
            self.optimizer.optimizer.load_state_dict(checkpoint_dict["optimizer"])
            for state in self.optimizer.optimizer.state.values():
                for k, v in state.items():
                    if isinstance(v, torch.Tensor):
                        state[k] = v.to(device)
            if self.lr_scheduler is not None:
                self.lr_scheduler.lr_scheduler.load_state_dict(checkpoint_dict["lr_scheduler"])
            self.best_valid_metrics = checkpoint_dict["metrics"]
            self.start_iteration = int(self.best_valid_metrics["epoch"]) + 1
            if self.train_replay_buffer is not None:
                self.train_replay_buffer.load_state_dict(checkpoint_dict["replay_buffer"])
            print(f"Loaded checkpoint from {self.start_iteration} iteration")

        self.initial_n_backward_trajectories = self.train_backward_n_trajectories
        self.sanity_check_evaluation = sanity_check_evaluation
        self.trajectory_filter = trajectory_filter or IdentityTrajectoryFilter()

    @property
    def hook_objects(self) -> List["TrainingHooksMixin"]:
        hooks = [self.objective]
        if self.train_replay_buffer:
            hooks.append(self.train_replay_buffer)
        if self.train_forward_sampler:
            hooks.append(self.train_forward_sampler)
        if self.train_backward_sampler:
            hooks.append(self.train_backward_sampler)
        if self.valid_sampler:
            hooks.append(self.valid_sampler)
        return hooks

    def sample_training_trajectories(self) -> Trajectories[TState, TActionSpace, TAction]:
        """
        Sample the training trajectories from the samplers and replay buffer.

        Returns:
            a `Trajectories` object containing the training trajectories.
        """
        trajectories_list = []
        if self.train_replay_buffer and self.train_replay_n_trajectories > 0:
            for trajectories in self.train_replay_buffer.get_trajectories_iterator(
                n_total_trajectories=self.train_replay_n_trajectories,
                batch_size=self.train_batch_size,
            ):
                trajectories_list.append(trajectories)
        if self.train_forward_sampler and self.train_forward_n_trajectories > 0:
            forward_trajectories_list = []
            for trajectories in self.train_forward_sampler.get_trajectories_iterator(
                self.train_forward_n_trajectories, self.train_batch_size
            ):
                forward_trajectories_list.append(trajectories)
            forward_trajectories = Trajectories.from_trajectories(forward_trajectories_list)
            trajectories_list.append(forward_trajectories)
            if self.train_replay_buffer:
                self.train_replay_buffer.add_trajectories(forward_trajectories)
        if self.train_backward_sampler and self.train_backward_n_trajectories > 0:
            for trajectories in self.train_backward_sampler.get_trajectories_iterator(
                self.train_backward_n_trajectories, self.train_batch_size
            ):
                trajectories_list.append(trajectories)

        trajectories = Trajectories.from_trajectories(trajectories_list)
        return trajectories

    @torch.no_grad()
    def valid_step(self) -> Dict[str, float]:
        """
        Perform one validation step. It samples the validation trajectories, computes the objective output, and
        computes the metrics and artifacts.

        Returns:
            a dictionary containing the metrics.
        """
        if self.valid_sampler is None:
            return {}
        metrics_list = []
        trajectories_list = []
        for trajectories in self.valid_sampler.get_trajectories_iterator(
            self.valid_n_trajectories, self.valid_batch_size
        ):
            objective = self.objective.compute_objective_output(trajectories=trajectories)
            metrics = (
                self.valid_metrics.compute_metrics(trajectories=trajectories)
                | {"loss": objective.loss.item()}
                | objective.metrics
            )
            metrics_list.append(metrics)
            trajectories.set_device("cpu")
            trajectories_list.append(trajectories)

        metrics = dict_mean(metrics_list)
        self.logger.log_metrics(metrics=metrics, prefix="valid")
        trajectories = Trajectories.from_trajectories(trajectories_list)
        artifacts = self.valid_artifacts.compute_artifacts(trajectories=trajectories)
        for artifact in artifacts:
            self.logger.log_to_file(
                content=artifact.content, name=artifact.name, type=artifact.type
            )
        return metrics

    def make_checkpoint(self, checkpoint_name: str, metrics: Dict[str, Any]):
        """
        Make a checkpoint of the model, optimizer, lr_scheduler, and metrics.

        Args:
            checkpoint_name: the name of the checkpoint.
            metrics: a dictionary containing the metrics to save in the checkpoint.

        Returns:
            None
        """
        checkpoint_dir = self.run_dir / "train" / "checkpoints"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        metrics = {k: v for k, v in metrics.items() if isinstance(v, (float, int))}
        checkpoint_dict = {
            "model": self.objective.state_dict(),
            "optimizer": self.optimizer.optimizer.state_dict(),
            "lr_scheduler": self.lr_scheduler.lr_scheduler.state_dict()
            if self.lr_scheduler
            else None,
            "metrics": metrics,
            "replay_buffer": self.train_replay_buffer.state_dict()
            if self.train_replay_buffer
            else None,
        }
        torch.save(checkpoint_dict, checkpoint_dir / f"{checkpoint_name}.pt")

    def train(self) -> Dict[str, float]:
        """
        The main training loop. It samples the training trajectories, computes the objective output, and computes the
        metrics and artifacts. It also performs validation steps and saves the checkpoints.

        Returns:
            a dictionary containing the best validation metrics.
        """
        for i in (pbar := tqdm(range(self.start_iteration, self.n_iterations))):
            self.optimizer.zero_grad()

            hook_update_dict = self.on_start_sampling(i)
            trajectories = self.sample_training_trajectories()
            trajectories = self.trajectory_filter(trajectories)
            hook_update_dict |= self.on_end_sampling(i, trajectories)

            hook_update_dict |= self.on_start_computing_objective(i, trajectories)
            objective = self.objective.compute_objective_output(trajectories=trajectories)
            hook_update_dict |= self.on_end_computing_objective(i, trajectories)

            objective.loss.backward()
            torch.nn.utils.clip_grad_norm_(self.objective.parameters(), self.gradient_clipping_norm)
            self.optimizer.step()
            if self.lr_scheduler is not None:
                self.lr_scheduler.step()

            pbar.set_description(f"Loss: {objective.loss.item():.4f}")
            metrics = (
                self.train_metrics.compute_metrics(trajectories=trajectories)
                | {"loss": objective.loss.item()}
                | objective.metrics
                | hook_update_dict
            )
            self.logger.log_metrics(metrics=metrics, prefix="train")
            artifacts = self.train_artifacts.compute_artifacts(trajectories=trajectories)
            for artifact in artifacts:
                self.logger.log_to_file(content=artifact.content, name=artifact.name)
            self.logger.log_files(self.train_metrics.collect_files())

            if (
                (i > 0 and i % self.valid_every_n_iterations == 0)
                or (i == 0 and self.sanity_check_evaluation)
                or i == self.n_iterations - 1
            ):
                valid_metrics = metrics if self.valid_sampler is None else self.valid_step()
                valid_metrics = valid_metrics | {"epoch": i}

                self.make_checkpoint(checkpoint_name="last_gfn", metrics=valid_metrics)
                if self.checkpoint_mode:
                    if self.metric_direction == "min":
                        is_best = valid_metrics[self.best_metric] < self.best_valid_metrics.get(
                            self.best_metric, float("inf")
                        )
                    else:
                        is_best = valid_metrics[self.best_metric] > self.best_valid_metrics.get(
                            self.best_metric, float("-inf")
                        )
                    if is_best:
                        self.logger.log_metrics(metrics=valid_metrics, prefix="best_valid")
                        self.best_valid_metrics = valid_metrics
                        self.make_checkpoint(
                            checkpoint_name="best_gfn", metrics=self.best_valid_metrics
                        )
                else:
                    self.best_valid_metrics = valid_metrics

        return {k: v for k, v in self.best_valid_metrics.items() if isinstance(v, (float, int))}

    def close(self):
        """
        A method that should be called at the end of training to close the logger.

        Returns:
            None
        """
        self.logger.close()
