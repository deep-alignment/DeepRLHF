from typing import Optional, Tuple

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F

from .utils import masked_mean


class GPTLMLoss(nn.Module):
    """
    GPT Language Model Loss
    """

    def __init__(self):
        super().__init__()
        self.IGNORE_INDEX = -100
        self.loss = nn.CrossEntropyLoss(ignore_index=self.IGNORE_INDEX)

    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        # Flatten the tokens
        return self.loss(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))


class PolicyLoss(nn.Module):
    """
    Policy Loss for PPO
    """

    def __init__(self, clip_eps: float = 0.2) -> None:
        super().__init__()
        self.clip_eps = clip_eps

    def forward(
        self,
        log_probs: torch.Tensor,
        old_log_probs: torch.Tensor,
        advantages: torch.Tensor,
        action_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        ratio = (log_probs - old_log_probs).exp()
        surr1 = ratio * advantages
        surr2 = ratio.clamp(1 - self.clip_eps, 1 + self.clip_eps) * advantages
        loss = -torch.min(surr1, surr2)
        loss = masked_mean(loss, action_mask, dim=-1).mean()
        return loss


class ValueLoss(nn.Module):
    """
    Value Loss for PPO
    """

    def __init__(self, clip_eps: float = None) -> None:
        super().__init__()
        self.clip_eps = clip_eps

    def forward(
        self,
        values: torch.Tensor,
        old_values: torch.Tensor,
        returns: torch.Tensor,
        action_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if self.clip_eps is not None:
            values_clipped = old_values + (values - old_values).clamp(-self.clip_eps, self.clip_eps)
            surr1 = (values_clipped - returns) ** 2
            surr2 = (values - returns) ** 2
            loss = torch.max(surr1, surr2)
        else:
            loss = (values - returns) ** 2

        loss = masked_mean(loss, action_mask, dim=-1).mean()
        return 0.5 * loss


# class PairWiseLoss(nn.Module):
#     """
#     Pairwise Loss for Reward Model
#     """

#     def forward(
#         self, chosen_reward: torch.Tensor, reject_reward: torch.Tensor, margin: torch.Tensor = None
#     ) -> torch.Tensor:
#         if margin is not None:
#             loss = -F.logsigmoid(chosen_reward - reject_reward - margin)
#         else:
#             loss = -F.logsigmoid(chosen_reward - reject_reward)
#         return loss.mean()


class LogExpLoss(nn.Module):
    """
    Pairwise Loss for Reward Model
    Details: https://arxiv.org/abs/2204.05862
    """

    def forward(
        self, chosen_reward: torch.Tensor, reject_reward: torch.Tensor, margin: torch.Tensor = None
    ) -> torch.Tensor:
        loss = torch.log(1 + torch.exp(reject_reward - chosen_reward)).mean()
        return loss


class DPOLoss(nn.Module):
    """
    DPO Loss
    """

    def __init__(self, beta: float, label_smoothing: float = 0.0, ipo: bool = False) -> None:
        super().__init__()
        self.beta = beta
        self.label_smoothing = label_smoothing
        self.ipo = ipo

    def forward(
        self,
        policy_chosen_logps: torch.Tensor,
        policy_rejected_logps: torch.Tensor,
        reference_chosen_logps: torch.Tensor,
        reference_rejected_logps: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        pi_logratios = policy_chosen_logps - policy_rejected_logps
        ref_logratios = reference_chosen_logps - reference_rejected_logps
        logits = pi_logratios - ref_logratios

        if self.ipo:
            losses = (logits - 1 / (2 * self.beta)) ** 2  # Eq. 17 of https://arxiv.org/pdf/2310.12036v2.pdf
        else:
            # Eq. 3 https://ericmitchell.ai/cdpo.pdf; label_smoothing=0 gives original DPO (Eq. 7 of https://arxiv.org/pdf/2305.18290.pdf)
            losses = (
                -F.logsigmoid(self.beta * logits) * (1 - self.label_smoothing)
                - F.logsigmoid(-self.beta * logits) * self.label_smoothing
            )

        loss = losses.mean()
        chosen_rewards = self.beta * (policy_chosen_logps - reference_chosen_logps).detach()
        rejected_rewards = self.beta * (policy_rejected_logps - reference_rejected_logps).detach()

        return loss, chosen_rewards, rejected_rewards


# Adapted from https://github.com/ContextualAI/HALOs/blob/ca9b7e3eeea220c0944ad8095d641da33f907a7e/trainers.py#L742
class VanillaKTOLoss(nn.Module):
    """
    KTO loss for even sampling
    """

    def __init__(self, beta: float) -> None:
        super().__init__()
        self.beta = beta

    def forward(
        self,
        policy_chosen_logps: torch.FloatTensor,
        policy_rejected_logps: torch.FloatTensor,
        reference_chosen_logps: torch.FloatTensor,
        reference_rejected_logps: torch.FloatTensor,
    ) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
        chosen_KL = (policy_chosen_logps - reference_chosen_logps).mean().clamp(min=0)
        rejected_KL = (policy_rejected_logps - reference_rejected_logps).mean().clamp(min=0)

        chosen_logratios = policy_chosen_logps - reference_chosen_logps
        rejected_logratios = policy_rejected_logps - reference_rejected_logps

        losses = torch.cat(
            (
                1 - F.sigmoid(self.beta * (chosen_logratios - rejected_KL)),
                1 - F.sigmoid(self.beta * (chosen_KL - rejected_logratios)),
            ),
            0,
        ).mean()

        chosen_rewards = self.beta * (policy_chosen_logps - reference_chosen_logps).detach()
        rejected_rewards = self.beta * (policy_rejected_logps - reference_rejected_logps).detach()
        return losses, chosen_rewards, rejected_rewards


# Adapted from https://github.com/ContextualAI/HALOs/blob/ca9b7e3eeea220c0944ad8095d641da33f907a7e/trainers.py#L770
class KTOLoss(nn.Module):
    """
    KTO loss for uneven sampling
    """

    def __init__(
        self, beta: float, desirable_weight: float, undesirable_weight: float, world_size: int, device: torch.device
    ) -> None:
        super().__init__()
        self.beta = beta
        self.world_size = world_size
        self.device = device
        self.desirable_weight = desirable_weight
        self.undesirable_weight = undesirable_weight

    def forward(
        self,
        policy_chosen_logps: torch.FloatTensor,
        policy_rejected_logps: torch.FloatTensor,
        policy_KL_logps: torch.FloatTensor,
        reference_chosen_logps: torch.FloatTensor,
        reference_rejected_logps: torch.FloatTensor,
        reference_KL_logps: torch.FloatTensor,
    ) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
        KL = (policy_KL_logps - reference_KL_logps).mean().detach()
        # all_reduce sums up the KL estimates across all devices (gradient will also be scaled by world size)
        dist.all_reduce(KL, op=dist.ReduceOp.SUM)
        # take average (will also scale gradients appropriately)
        KL = (KL / self.world_size).clamp(min=0)

        if policy_chosen_logps.shape[0] != 0:
            chosen_logratios = policy_chosen_logps - reference_chosen_logps
            chosen_losses = 1 - F.sigmoid(self.beta * (chosen_logratios - KL))
            chosen_rewards = self.beta * chosen_logratios.detach()
        else:
            # important to cast to policy_dtype; otherwise error will occur during all_gather
            chosen_losses = torch.Tensor([]).to(policy_rejected_logps.dtype).to(self.device)
            chosen_rewards = torch.Tensor([]).to(policy_rejected_logps.dtype).to(self.device)

        if policy_rejected_logps.shape[0] != 0:
            rejected_logratios = policy_rejected_logps - reference_rejected_logps
            rejected_losses = 1 - F.sigmoid(self.beta * (KL - rejected_logratios))
            rejected_rewards = self.beta * rejected_logratios.detach()
        else:
            # important to cast to policy_dtype; otherwise error will occur during all_gather
            rejected_losses = torch.Tensor([]).to(policy_chosen_logps.dtype).to(self.device)
            rejected_rewards = torch.Tensor([]).to(policy_chosen_logps.dtype).to(self.device)

        losses = torch.cat(
            (self.desirable_weight * chosen_losses, self.undesirable_weight * rejected_losses), 0
        ).mean()
        return losses, chosen_rewards, rejected_rewards, KL


# Adapted from https://github.com/microsoft/LMOps/blob/main/minillm/finetune.py#L166
class KDLoss(nn.Module):
    """
    Language Model Knowledge Distillation Loss
    """

    def __init__(self):
        super().__init__()
        self.IGNORE_INDEX = -100

    def forward(self, logits: torch.Tensor, teacher_logits: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
        teacher_probs = F.softmax(teacher_logits, dim=-1, dtype=torch.float32)
        inf_mask = torch.isinf(logits)
        logprobs = F.log_softmax(logits, dim=-1, dtype=torch.float32)
        prod_probs = torch.masked_fill(teacher_probs * logprobs, inf_mask, 0)
        x = torch.sum(prod_probs, dim=-1).view(-1)
        mask = (label != self.IGNORE_INDEX).int()
        distil_loss = -torch.sum(x * mask.view(-1), dim=0) / torch.sum(mask.view(-1), dim=0)

        return distil_loss


class PRMLoss(nn.Module):
    """
    Process Reward Model Loss
    """

    def __init__(self, placeholder_token_id: int, reward_token_ids: Optional[list[int]] = None):
        super().__init__()
        self.IGNORE_INDEX = -100
        self.loss = nn.CrossEntropyLoss(ignore_index=self.IGNORE_INDEX)
        self.placeholder_token_id = placeholder_token_id
        self.reward_token_ids = reward_token_ids

    def forward(self, inputs: torch.Tensor, logits: torch.Tensor, labels: torch.Tensor, *, return_acc: bool = False):
        placeholder_mask = inputs == self.placeholder_token_id
        logits = logits[placeholder_mask]
        labels = labels[placeholder_mask]

        if labels.dtype == torch.float:
            # soft label
            assert len(self.reward_token_ids) == 2, "reward_token_ids should have 2 tokens for soft labels"
            logits = logits[..., self.reward_token_ids]
            positive_labels = labels.to(logits.dtype)
            negative_labels = 1 - positive_labels
            negative_labels[positive_labels != -100] = 1 - positive_labels[positive_labels != -100]
            labels = torch.stack([positive_labels, negative_labels], dim=-1)
        elif self.reward_token_ids is not None:
            # hard label with reward_token_ids set. (otherwise the whole vocab will be trained together.)
            logits = logits[..., self.reward_token_ids]
            # this is slow....
            for i, token in enumerate(self.reward_token_ids):
                labels = torch.where(labels == token, i, labels)

        loss = self.loss(logits, labels)
        if not return_acc:
            return loss

        if labels.dtype == logits.dtype:
            labels = labels.argmax(dim=-1)
        acc = (logits.argmax(dim=-1) == labels).float().mean()
        return loss, acc

class SFTVanillaLoss(nn.Module):
    """
    SFT Vanilla Regularization Loss
    """
    def __init__(self) -> None:
        super().__init__()
        self.IGNORE_INDEX = 0
        
    def forward(self, logits: torch.Tensor, labels: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        labels = labels[:, 1:].clone()
        logits = logits[:, :-1, :]
        mask = mask.clone().bool()
        mask = mask[:, 1:]
        
        log_probs = F.log_softmax(logits, dim=-1)
        chosen_log_probs = torch.gather(log_probs, 2, labels.unsqueeze(-1)).squeeze(-1)
        chosen_log_probs = chosen_log_probs * mask
        valid_log_probs = chosen_log_probs.sum(dim=1) / mask.sum(dim=1)
        loss = -valid_log_probs.mean()
        
        return loss  
      
class SFTMeanLoss(nn.Module):
    """
    SFT Mean Regularization Loss
    """
    def __init__(self, beta: float = 2.0) -> None:
        super().__init__()
        self.IGNORE_INDEX = 0
        self.beta = beta  
    def forward(self, logits: torch.Tensor, labels: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        labels = labels[:, 1:].clone()
        logits = logits[:, :-1, :]
        mask = mask.clone().bool()
        mask = mask[:, 1:]
        
        log_probs = F.log_softmax(logits, dim=-1)
        chosen_log_probs = torch.gather(log_probs, 2, labels.unsqueeze(-1)).squeeze(-1)
        chosen_log_probs = chosen_log_probs * mask
        valid_log_probs = chosen_log_probs.sum(dim=1) / mask.sum(dim=1)
        loss = -F.logsigmoid(self.beta * valid_log_probs).mean()
        
        return loss  
     
class SFTSumLoss(nn.Module):
    """
    SFT Sum Regularization Loss
    """
    def __init__(self, beta: float = 0.1) -> None:
        super().__init__()
        self.IGNORE_INDEX = 0
        self.beta = beta  
    def forward(self, logits: torch.Tensor, labels: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        labels = labels[:, 1:].clone()
        logits = logits[:, :-1, :]
        mask = mask.clone().bool()
        mask = mask[:, 1:]
        
        log_probs = F.log_softmax(logits, dim=-1)
        chosen_log_probs = torch.gather(log_probs, 2, labels.unsqueeze(-1)).squeeze(-1)
        chosen_log_probs = chosen_log_probs * mask
        valid_log_probs = chosen_log_probs.sum(dim=1)
        loss = -F.logsigmoid(self.beta * valid_log_probs).mean()
        
        return loss
    
class DPORefFreeLoss(nn.Module):
    """
    DPO Reference-free Regularization Loss
    """
    def __init__(self, beta: float = 2.0, margin = 1.0) -> None:
        super().__init__()
        self.IGNORE_INDEX = 0
        self.beta = beta  
        self.margin = margin
        
    def forward(self, chosen_logits: torch.Tensor, chosen_labels: torch.Tensor, chosen_mask: torch.Tensor, rejected_logits: torch.Tensor, rejected_labels: torch.Tensor, rejected_mask: torch.Tensor) -> torch.Tensor:
        chosen_labels = chosen_labels[:, 1:].clone()
        chosen_logits = chosen_logits[:, :-1, :]
        chosen_mask = chosen_mask.clone().bool()
        chosen_mask = chosen_mask[:, 1:]
        rejected_labels = rejected_labels[:, 1:].clone()
        rejected_logits = rejected_logits[:, :-1, :]
        rejected_mask = rejected_mask.clone().bool()
        rejected_mask = rejected_mask[:, 1:]
                
        chosen_log_probs = F.log_softmax(chosen_logits, dim=-1)
        rejected_log_probs = F.log_softmax(rejected_logits, dim=-1)
        chosen_log_probs = torch.gather(chosen_log_probs, 2, chosen_labels.unsqueeze(-1)).squeeze(-1)
        rejected_log_probs = torch.gather(rejected_log_probs, 2, rejected_labels.unsqueeze(-1)).squeeze(-1)
        chosen_log_probs = chosen_log_probs * chosen_mask
        rejected_log_probs = rejected_log_probs * rejected_mask
        chosen_valid_log_probs = chosen_log_probs.sum(dim=1) / chosen_mask.sum(dim=1)
        rejected_valid_log_probs = rejected_log_probs.sum(dim=1) / rejected_mask.sum(dim=1)
        loss = -F.logsigmoid(self.beta * (chosen_valid_log_probs - rejected_valid_log_probs) - self.margin).mean()    
        return loss
    
class PairWiseLoss(nn.Module):
    """
    Pairwise Loss for Reward Model
    """
    def __init__(self, tau: float = 0.1):
        super().__init__()
        self.tau = tau
        
    def forward(
        self, chosen_reward: torch.Tensor, reject_reward: torch.Tensor, margin: torch.Tensor = None
    ) -> torch.Tensor:
        if margin is not None:
            loss = -F.logsigmoid((chosen_reward - reject_reward - margin) / self.tau)
            prob = F.sigmoid((chosen_reward - reject_reward - margin) / self.tau)
        else:
            loss = -F.logsigmoid((chosen_reward - reject_reward) / self.tau)
            prob = F.sigmoid((chosen_reward - reject_reward) / self.tau)
        return loss.mean(), prob.mean() 
    
class PairWiseRegressionLoss(nn.Module):
    """
    Pairwise Loss for Reward Model Regression Loss
    """
    def __init__(self, tau: float = 0.1, target_margin: float = 10.0):
        super().__init__()
        self.tau = tau
        self.target_margin = target_margin
        
    def forward(
        self, chosen_reward: torch.Tensor, reject_reward: torch.Tensor, margin: torch.Tensor = None
    ) -> torch.Tensor:
        if margin is not None:
            margin_scale = 5
            result = chosen_reward - reject_reward
            loss = 0.5 * (result / self.tau - margin_scale * F.sigmoid(margin)) ** 2
            prob = F.sigmoid((result - margin) / self.tau)
        else:
            result = chosen_reward - reject_reward
            loss = 0.5 * (result / self.tau - self.target_margin) ** 2
            prob = F.sigmoid(result / self.tau)
        return loss.mean(), prob.mean()
    
class PairWiseLearnableTauLoss(nn.Module):
    """
    Pairwise Loss for Reward Model with Learnable Tau
    """
    def __init__(self, init_tau: float = -2.25):
        super().__init__()
        # Initialize tau as a learnable parameter
        self.tau = nn.Parameter(torch.tensor(init_tau))

    def forward(self, chosen_reward: torch.Tensor, reject_reward: torch.Tensor, margin: torch.Tensor = None) -> torch.Tensor:
        real_tau = max(-F.logsigmoid(-self.tau), 1e-2)
        if margin is not None:
            scaled_diff = (chosen_reward - reject_reward - margin) / real_tau
            loss = -F.logsigmoid(scaled_diff)
            prob = F.sigmoid(scaled_diff)
        else:
            scaled_diff = (chosen_reward - reject_reward) / real_tau
            loss = -F.logsigmoid(scaled_diff)
            prob = F.sigmoid(scaled_diff)
        
        return loss.mean(), prob.mean()   
     
class PairWiseLearnableTauRegressionLoss(nn.Module):
    """
    Pairwise Loss for Reward Model with Learnable Tau Regression Loss
    """
    def __init__(self, init_tau: float = 2.25,  target_margin: float = 10.0):
        super().__init__()
        # Initialize beta as a learnable parameter
        self.tau = nn.Parameter(torch.tensor(init_tau))
        self.target_margin = target_margin

    def forward(self, chosen_reward: torch.Tensor, reject_reward: torch.Tensor, margin: torch.Tensor = None) -> torch.Tensor:
        real_tau = max(-F.logsigmoid(self.tau), 1e-2)
        if margin is not None:
            scaled_diff = (chosen_reward - reject_reward - margin) / real_tau
            loss = 0.5 * scaled_diff ** 2
            prob = F.sigmoid(scaled_diff)
        else:
            scaled_diff = (chosen_reward - reject_reward) / real_tau
            loss = 0.5 * (scaled_diff - self.target_margin) ** 2
            prob = F.sigmoid(scaled_diff)
        
        return loss.mean(), prob.mean() 
    
class GeneralPreferenceLoss(nn.Module):
    """
    Loss for General Preference Reward Model
    """
    def __init__(self, tau: float = 1):
        super().__init__()
        self.tau = tau
        
    def forward(
        self, chosen_reward: torch.Tensor, reject_reward: torch.Tensor, margin: torch.Tensor = None
    ) -> torch.Tensor:
        if margin is not None:
            result = chosen_reward[:, 0] * reject_reward[:, 1] - chosen_reward[:, 1] * reject_reward[:, 0]
            loss = -F.logsigmoid((result - margin) / self.tau)
            prob = F.sigmoid((result - margin) / self.tau)
        else:
            result = chosen_reward[:, 0] * reject_reward[:, 1] - chosen_reward[:, 1] * reject_reward[:, 0]
            loss = -F.logsigmoid(result / self.tau)
            prob = F.sigmoid(result / self.tau)
        return loss.mean(), prob.mean()
    
class GeneralPreferenceRegressionLoss(nn.Module):
    """
    Loss for General Preference Reward Model Regression Loss
    """
    def __init__(self, tau: float = 1, target_margin: float = 10.0):
        super().__init__()
        self.tau= tau
        self.target_margin = target_margin
        
    def forward(
        self, chosen_reward: torch.Tensor, reject_reward: torch.Tensor, margin: torch.Tensor = None
    ) -> torch.Tensor:
        if margin is not None:
            margin_scale = 5
            result = chosen_reward[:, 0] * reject_reward[:, 1] - chosen_reward[:, 1] * reject_reward[:, 0]
            loss = 0.5 * (result / self.tau - margin_scale * F.sigmoid(margin)) ** 2
            prob = F.sigmoid((result - margin) / self.tau)
        else:
            result = chosen_reward[:, 0] * reject_reward[:, 1] - chosen_reward[:, 1] * reject_reward[:, 0]
            loss = 0.5 * (result / self.tau - self.target_margin) ** 2
            prob = F.sigmoid(result / self.tau)
        return loss.mean(), prob.mean()
    
class GeneralPreferenceLearnableTauLoss(nn.Module):
    """
    Loss for General Preference Reward Model with Learnable Tau
    """
    def __init__(self, init_tau: float = -2.25):
        super().__init__()
        # Initialize tau as a learnable parameter
        self.tau = nn.Parameter(torch.tensor(init_tau))

    def forward(self, chosen_reward: torch.Tensor, reject_reward: torch.Tensor, margin: torch.Tensor = None) -> torch.Tensor:
        real_tau = max(-F.logsigmoid(-self.tau), 1e-2)
        if margin is not None:
            result = chosen_reward[:, 0] * reject_reward[:, 1] - chosen_reward[:, 1] * reject_reward[:, 0]
            loss = -F.logsigmoid((result - margin) / real_tau)
            prob = F.sigmoid((result - margin) / real_tau)
        else:
            result = chosen_reward[:, 0] * reject_reward[:, 1] - chosen_reward[:, 1] * reject_reward[:, 0]
            loss = -F.logsigmoid(result / real_tau)
            prob = F.sigmoid(result / real_tau)
        
        return loss.mean(), prob.mean()
    
class GeneralPreferenceLearnableTauRegressionLoss(nn.Module):
    """
    Loss for General Preference Reward Model with Learnable Tau Regression Loss
    """
    def __init__(self, init_tau: float = -2.25, target_margin: float = 10.0):
        super().__init__()
        # Initialize tau as a learnable parameter
        self.tau = nn.Parameter(torch.tensor(init_tau))
        self.target_margin = target_margin

    def forward(self, chosen_reward: torch.Tensor, reject_reward: torch.Tensor, margin: torch.Tensor = None) -> torch.Tensor:
        real_tau = max(-F.logsigmoid(-self.tau), 1e-2)
        if margin is not None:
            result = chosen_reward[:, 0] * reject_reward[:, 1] - chosen_reward[:, 1] * reject_reward[:, 0]
            loss = 0.5 * ((result - margin) / real_tau) ** 2
            prob = F.sigmoid((result - margin) / real_tau)
        else:
            result = chosen_reward[:, 0] * reject_reward[:, 1] - chosen_reward[:, 1] * reject_reward[:, 0]
            loss = 0.5 * (result / real_tau - self.target_margin) ** 2
            prob = F.sigmoid(result / real_tau)
        
        return loss.mean(), prob.mean()
    
class HighDimGeneralPreferenceLoss(nn.Module):
    """
    Loss for General Preference Reward Model with high dimension value_head
    """
    def __init__(self, tau: float = 0.1, value_head_dim: int = 4, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.tau = tau
        self.value_head_dim = value_head_dim
        
    def create_skew_symmetric_block_matrix(self, dim, device, dtype):
        matrix = torch.zeros((dim, dim), device=device, dtype=dtype)
        for i in range(0, dim, 2):
            matrix[i, i+1] = -1 
            matrix[i+1, i] = 1            
        return matrix

    def forward(
        self, chosen_reward: torch.Tensor, reject_reward: torch.Tensor, margin: torch.Tensor = None 
    ) -> torch.Tensor:
        if margin is not None:        
            R_matrix = self.create_skew_symmetric_block_matrix(self.value_head_dim, chosen_reward.device, chosen_reward.dtype)
            if chosen_reward.device == reject_reward.device == R_matrix.device:
                transformed_chosen = torch.matmul(chosen_reward, R_matrix.T)
                result = torch.bmm(transformed_chosen.view(chosen_reward.shape[0], 1, self.value_head_dim), reject_reward.view(reject_reward.shape[0], self.value_head_dim, 1))
                result = result.view(chosen_reward.shape[0])              
            loss = -F.logsigmoid((result - margin) / self.tau) 
            prob = F.sigmoid((result - margin) / self.tau)
        else:
            R_matrix = self.create_skew_symmetric_block_matrix(self.value_head_dim, chosen_reward.device, chosen_reward.dtype)
            if chosen_reward.device == reject_reward.device == R_matrix.device:
                transformed_chosen = torch.matmul(chosen_reward, R_matrix.T)
                result = torch.bmm(transformed_chosen.view(chosen_reward.shape[0], 1, self.value_head_dim), reject_reward.view(reject_reward.shape[0], self.value_head_dim, 1))
                result = result.view(chosen_reward.shape[0])  
            loss = -F.logsigmoid(result / self.tau)
            prob = F.sigmoid(result / self.tau)
        return loss.mean(), prob.mean()  
    
class HighDimGeneralPreferenceRegressionLoss(nn.Module):
    """
    Loss for General Preference Reward Model with high dimension value_head
    """
    def __init__(self, tau: float = 0.1, target_margin: float = 10.0, value_head_dim: int = 4, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.tau = tau
        self.value_head_dim = value_head_dim
        self.target_margin = target_margin
        
    def create_skew_symmetric_block_matrix(self, dim, device, dtype):
        matrix = torch.zeros((dim, dim), device=device, dtype=dtype)
        for i in range(0, dim, 2):
            matrix[i, i+1] = -1 
            matrix[i+1, i] = 1            
        return matrix

    def forward(
        self, chosen_reward: torch.Tensor, reject_reward: torch.Tensor, margin: torch.Tensor = None 
    ) -> torch.Tensor:
        if margin is not None:        
            R_matrix = self.create_skew_symmetric_block_matrix(self.value_head_dim, chosen_reward.device, chosen_reward.dtype)
            if chosen_reward.device == reject_reward.device == R_matrix.device:
                transformed_chosen = torch.matmul(chosen_reward, R_matrix.T)
                result = torch.bmm(transformed_chosen.view(chosen_reward.shape[0], 1, self.value_head_dim), reject_reward.view(reject_reward.shape[0], self.value_head_dim, 1))
                result = result.view(chosen_reward.shape[0])              
            loss = 0.5 * ((result - margin) / self.tau) ** 2
            prob = F.sigmoid((result - margin) / self.tau)
        else:
            R_matrix = self.create_skew_symmetric_block_matrix(self.value_head_dim, chosen_reward.device, chosen_reward.dtype)
            if chosen_reward.device == reject_reward.device == R_matrix.device:
                transformed_chosen = torch.matmul(chosen_reward, R_matrix.T)
                result = torch.bmm(transformed_chosen.view(chosen_reward.shape[0], 1, self.value_head_dim), reject_reward.view(reject_reward.shape[0], self.value_head_dim, 1))
                result = result.view(chosen_reward.shape[0])  
            loss = 0.5 * (result / self.tau - self.target_margin) ** 2
            prob = F.sigmoid(result / self.tau)
        return loss.mean(), prob.mean()
    
class HighDimGeneralPreferenceLearnableTauLoss(nn.Module):
    """
    Loss for General Preference Reward Model with high dimension value_head with Learnable Tau
    """
    def __init__(self, value_head_dim: int = 4, init_tau: float = 2.25, scale: float = 1.0, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.value_head_dim = value_head_dim
        self.scale = scale
        # Initialize learnable taus
        self.taus = nn.Parameter(torch.full((value_head_dim // 2,), init_tau))

    def create_skew_symmetric_block_matrix(self, dim, device, dtype):
        assert dim % 2 == 0, "Dimension must be even"
        matrix = torch.zeros((dim, dim), device=device, dtype=dtype)
        
        # Iterate through the pairs and set the matrix elements with learnable taus
        for i in range(0, dim, 2):
            tau_i = self.taus[i // 2]
            transform_value = 1 / max(-torch.logsigmoid(tau_i), 1e-2)
            matrix[i, i+1] = -transform_value
            matrix[i+1, i] = transform_value
        
        return matrix

    def forward(
        self, chosen_reward: torch.Tensor, reject_reward: torch.Tensor, margin: torch.Tensor = None 
    ) -> torch.Tensor:
        if margin is not None:        
            R_matrix = self.create_skew_symmetric_block_matrix(self.value_head_dim, chosen_reward.device, chosen_reward.dtype)
            if chosen_reward.device == reject_reward.device == R_matrix.device:
                transformed_chosen = torch.matmul(chosen_reward, R_matrix.T)
                result = torch.bmm(transformed_chosen.view(chosen_reward.shape[0], 1, self.value_head_dim), reject_reward.view(reject_reward.shape[0], self.value_head_dim, 1))
                result = result.view(chosen_reward.shape[0])              
            loss = -F.logsigmoid((result - margin) / self.scale) 
            prob = F.sigmoid((result - margin) / self.scale)
        else:
            R_matrix = self.create_skew_symmetric_block_matrix(self.value_head_dim, chosen_reward.device, chosen_reward.dtype)
            if chosen_reward.device == reject_reward.device == R_matrix.device:
                transformed_chosen = torch.matmul(chosen_reward, R_matrix.T)
                result = torch.bmm(transformed_chosen.view(chosen_reward.shape[0], 1, self.value_head_dim), reject_reward.view(reject_reward.shape[0], self.value_head_dim, 1))
                result = result.view(chosen_reward.shape[0])  
            loss = -F.logsigmoid(result / self.scale)
            prob = F.sigmoid(result / self.scale)
        return loss.mean(), prob.mean()
    
class HighDimGeneralPreferenceMoELoss(nn.Module):
    """
    Loss for General Preference Reward Model with high dimension value_head and Data Dependent MoE
    """
    def __init__(self, model, value_head_dim: int = 4, softmax_tau: float = 1.0, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.model = model
        self.value_head_dim = value_head_dim
        self.softmax_tau = softmax_tau

    def forward(
        self, chosen_reward: torch.Tensor, reject_reward: torch.Tensor, prompt_hidden_states: torch.Tensor, margin: torch.Tensor = None) -> torch.Tensor:
        if margin is not None:        
            R_matrix = self.model.create_skew_symmetric_block_matrix(self.value_head_dim, chosen_reward.device, chosen_reward.dtype, prompt_hidden_states)
            if chosen_reward.device == reject_reward.device == R_matrix.device:
                # Batched matrix multiplication with .view() instead of unsqueeze
                transformed_chosen = torch.bmm(chosen_reward.view(chosen_reward.shape[0], 1, self.value_head_dim), R_matrix.transpose(1, 2))
                result = torch.bmm(transformed_chosen, reject_reward.view(reject_reward.shape[0], self.value_head_dim, 1))
                result = result.view(chosen_reward.shape[0])  
            loss = -F.logsigmoid((result - margin) / self.softmax_tau) 
            prob = F.sigmoid((result - margin) / self.softmax_tau)
        else:
            R_matrix = self.model.create_skew_symmetric_block_matrix(self.value_head_dim, chosen_reward.device, chosen_reward.dtype, prompt_hidden_states)
            if chosen_reward.device == reject_reward.device == R_matrix.device:
                transformed_chosen = torch.bmm(chosen_reward.view(chosen_reward.shape[0], 1, self.value_head_dim), R_matrix.transpose(1, 2))
                result = torch.bmm(transformed_chosen, reject_reward.view(reject_reward.shape[0], self.value_head_dim, 1))
                result = result.view(chosen_reward.shape[0])  
            loss = -F.logsigmoid(result / self.softmax_tau)
            prob = F.sigmoid(result / self.softmax_tau)
        return loss.mean(), prob.mean()
    
class HighDimGeneralPreferenceRegressionMoELoss(nn.Module):
    """
    Loss for General Preference Reward Model with high dimension value_head and Data Dependent MoE
    """
    def __init__(self, model, value_head_dim: int = 4, target_margin: float = 10.0, softmax_tau: float = 1.0, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.model = model
        self.value_head_dim = value_head_dim
        self.target_margin = target_margin
        self.softmax_tau = softmax_tau

    def forward(
        self, chosen_reward: torch.Tensor, reject_reward: torch.Tensor, prompt_hidden_states: torch.Tensor, margin: torch.Tensor = None) -> torch.Tensor:
        if margin is not None:        
            R_matrix = self.model.create_skew_symmetric_block_matrix(self.value_head_dim, chosen_reward.device, chosen_reward.dtype, prompt_hidden_states)
            if chosen_reward.device == reject_reward.device == R_matrix.device:
                # Batched matrix multiplication with .view() instead of unsqueeze
                transformed_chosen = torch.bmm(chosen_reward.view(chosen_reward.shape[0], 1, self.value_head_dim), R_matrix.transpose(1, 2))
                result = torch.bmm(transformed_chosen, reject_reward.view(reject_reward.shape[0], self.value_head_dim, 1))
                result = result.view(chosen_reward.shape[0])  
            loss = 0.5 * ((result - margin) / self.softmax_tau) ** 2
            prob = F.sigmoid((result - margin) / self.softmax_tau)
        else:
            R_matrix = self.model.create_skew_symmetric_block_matrix(self.value_head_dim, chosen_reward.device, chosen_reward.dtype, prompt_hidden_states)
            if chosen_reward.device == reject_reward.device == R_matrix.device:
                transformed_chosen = torch.bmm(chosen_reward.view(chosen_reward.shape[0], 1, self.value_head_dim), R_matrix.transpose(1, 2))
                result = torch.bmm(transformed_chosen, reject_reward.view(reject_reward.shape[0], self.value_head_dim, 1))
                result = result.view(chosen_reward.shape[0])  
            loss = 0.5 * (result / self.softmax_tau - self.target_margin) ** 2
            prob = F.sigmoid(result / self.softmax_tau)
        return loss.mean(), prob.mean()
