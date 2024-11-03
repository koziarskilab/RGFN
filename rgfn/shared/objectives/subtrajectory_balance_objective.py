import gin
import torch
from torch_geometric.utils import to_dense_batch

from rgfn.api.objective_base import ObjectiveBase, ObjectiveOutput
from rgfn.api.policy_base import PolicyBase
from rgfn.api.trajectories import Trajectories
from rgfn.api.type_variables import TAction, TActionSpace, TState


@gin.configurable()
class SubTrajectoryBalanceObjective(ObjectiveBase[TState, TActionSpace, TAction]):
    """
    Sub-trajectory balance objective function for GFN from the paper "Learning GFlowNets from partial episodes for
    improved convergence and stability" (https://arxiv.org/abs/2209.12782).

    Attributes:
        forward_policy: a policy that estimates the probabilities of actions taken in the forward direction.
        backward_policy: a policy that estimates the probabilities of actions taken in the backward direction.
        lambda_coeff: lambda coefficient for the sub-trajectory balance objective.
    """

    def __init__(
        self,
        forward_policy: PolicyBase[TState, TActionSpace, TAction],
        backward_policy: PolicyBase[TState, TActionSpace, TAction],
        lambda_coeff: float,
    ):
        super().__init__(forward_policy=forward_policy, backward_policy=backward_policy)
        self.lambda_coeff = lambda_coeff

    def compute_objective_output(
        self, trajectories: Trajectories[TState, TActionSpace, TAction]
    ) -> ObjectiveOutput:
        """
        Compute the sub-trajectory objective output on a batch of trajectories.

        Args:
            trajectories: the batch of trajectories obtained in the sampling process. It contains the states, actions,
                action spaces in forward and backward directions, and rewards. Other important quantities (e.g. log
                probabilities of taking actions in forward and backward directions) should be assigned in this method
                using appropriate methods (e.g. assign_log_probs).

        Returns:
            The output of the objective function, containing the loss and possibly some metrics.
        """
        self.assign_log_probs(trajectories)
        self.assign_log_flows(trajectories)

        forward_log_prob = trajectories.get_forward_log_probs_flat()  # [n_actions]
        backward_log_prob = trajectories.get_backward_log_probs_flat()  # [n_actions]
        log_flow = trajectories.get_log_flows_flat()  # [n_actions]
        log_reward = trajectories.get_reward_outputs().log_reward  # [n_trajectories]
        index = trajectories.get_index_flat().to(self.device)  # [n_actions]

        log_prob_diff = forward_log_prob - backward_log_prob  # [n_actions]
        log_prob_diff, action_mask = to_dense_batch(
            log_prob_diff, index
        )  # [n_trajectories, max_num_actions]
        log_prob_diff_cumsum = log_prob_diff.cumsum(dim=1)  # [n_trajectories, max_num_actions]
        log_prob_diff_cumsum = torch.cat(
            [torch.zeros_like(log_prob_diff_cumsum[:, :1]), log_prob_diff_cumsum], dim=1
        )  # [n_trajectories, max_num_actions + 1]

        log_flow, _ = to_dense_batch(log_flow, index)  # [n_trajectories, max_num_actions]
        log_flow = torch.cat(
            [log_flow, torch.zeros_like(log_flow[:, :1])], dim=1
        )  # [n_trajectories, max_num_actions + 1]

        # assign log_reward to the last action of each trajectory in log_flow
        reward_index = torch.sum(action_mask, dim=1).long()
        log_flow.scatter_(
            dim=1, index=reward_index.unsqueeze(-1), src=log_reward.unsqueeze(-1)
        )  # [n_trajectories, max_num_actions + 1]
        max_num_actions = log_prob_diff.shape[1]

        diff_list = []
        mask_list = []
        coeff_list = []
        for i in range(max_num_actions):
            for j in range(i + 1, max_num_actions + 1):
                diff = (
                    log_flow[:, i]
                    - log_flow[:, j]
                    + log_prob_diff_cumsum[:, j]
                    - log_prob_diff_cumsum[:, i]
                )
                mask = action_mask[:, i] & action_mask[:, j - 1]
                coeff = self.lambda_coeff ** (j - i)
                diff_list.append(diff)
                mask_list.append(mask)
                coeff_list.append(coeff)
        diff = torch.stack(diff_list, dim=1)
        mask = torch.stack(mask_list, dim=1)
        coeff = torch.tensor(coeff_list).to(self.device)
        coeff = coeff * mask

        loss = torch.pow(diff, 2) * coeff
        loss = torch.sum(loss, dim=1) / torch.sum(coeff, dim=1)
        return ObjectiveOutput(loss=loss.mean(), metrics={})
