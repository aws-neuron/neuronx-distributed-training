import neuronx_distributed as nxd
import torch
from transformers import LlamaConfig
import torch.nn.functional as F
import os
import torch_xla.core.xla_model as xm
from neuronx_distributed.parallel_layers.loss_functions import from_parallel_logits_to_logprobs
from .base_dpo import DPOBaseModel
from omegaconf import DictConfig, open_dict
from pytorch_lightning.trainer.trainer import Trainer
import numpy as np
from neuronx_distributed.parallel_layers.loss_functions import parallel_cross_entropy

class ORPOBaseModel(DPOBaseModel):

    def __init__(self, cfg: DictConfig, trainer: Trainer, model=None):
        try:
            global trl
            import trl
        except ImportError:
            raise ImportError("trl is required for the ORPO algorithm, but it is not available. Please install the library to continue.")

    def on_train_start(self, trainer, model, config) -> None:
        pass

    def model_fwd_calc_loss(self, model, batch, config):

        policy_chosen_logps, policy_rejected_logps, all_logits = self.policy_model_concatenated_forward(model, batch, average_log_probs=True)

        # orpo chosen nll loss is computed over the full prompt and chosen response
        policy_chosen_nll_loss = -policy_chosen_logps.mean(0)
        
        # odds ratio loss
        log_odds = (policy_chosen_logps - policy_rejected_logps) - (torch.log(1 - torch.exp(policy_chosen_logps)) - torch.log(1 - torch.exp(policy_rejected_logps)))
        sig_ratio = torch.nn.functional.sigmoid(log_odds)
        ratio = torch.log(sig_ratio)
        losses = config.model_alignment_strategy.orpo.beta * ratio
        
        # Calculate the ORPO loss and reward metrics
        orpo_loss = torch.mean(policy_chosen_nll_loss - losses.mean())
        chosen_rewards = config.model_alignment_strategy.orpo.beta * (policy_chosen_logps.to(all_logits.device)).detach()
        rejected_rewards = config.model_alignment_strategy.orpo.beta * (policy_rejected_logps.to(all_logits.device)).detach()

        misc_metrics = {'chosen_rewards': chosen_rewards, 'rejected_rewards': rejected_rewards}

        return orpo_loss, misc_metrics