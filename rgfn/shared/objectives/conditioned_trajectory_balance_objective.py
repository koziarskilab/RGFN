import gin
import torch

from rgfn.api.objective_base import ObjectiveBase, ObjectiveOutput
from rgfn.api.trajectories import Trajectories
from rgfn.api.type_variables import TAction, TActionSpace, TState


@gin.configurable()
class ConditionedTrajectoryBalanceObjective(ObjectiveBase[TState, TActionSpace, TAction]):
    def compute_objective_output(
        self, trajectories: Trajectories[TState, TActionSpace, TAction]
    ) -> ObjectiveOutput:
        self.assign_log_probs(trajectories)

        source_states = trajectories.get_source_states_flat()  # [n_states]
        source_log_flow = self.forward_policy.compute_states_log_flow(source_states)  # [n_states]
        forward_log_prob = trajectories.get_forward_log_probs_flat()  # [n_actions]
        backward_log_prob = trajectories.get_backward_log_probs_flat()  # [n_actions]
        log_reward = trajectories.get_reward_outputs().log_reward  # [n_trajectories]
        index = trajectories.get_index_flat().to(self.device)  # [n_actions]

        loss = torch.scatter_add(
            input=source_log_flow - log_reward,
            index=index,
            src=forward_log_prob - backward_log_prob,
            dim=0,
        )  # [n_trajectories]
        loss = loss.pow(2).mean()
        return ObjectiveOutput(
            loss=loss,
            metrics={
                "mean_log_flow": source_log_flow.mean().item(),
                "abs_mean_log_flow": source_log_flow.abs().mean().item(),
            },
        )
