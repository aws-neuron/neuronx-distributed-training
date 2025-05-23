import neuronx_distributed as nxd
import torch
from transformers import LlamaConfig
import torch.nn.functional as F
import os
import torch_xla.core.xla_model as xm
from neuronx_distributed.parallel_layers.loss_functions import from_parallel_logits_to_logprobs
from .base import BaseModelModule
from omegaconf import DictConfig, open_dict
from lightning.pytorch.trainer.trainer import Trainer
import numpy as np
from lightning.pytorch.trainer.connectors.data_connector import _DataLoaderSource

class DPOBaseModel(BaseModelModule):

    def __init__(self, cfg: DictConfig, trainer: Trainer, model=None):
        try:
            global trl
            import trl
        except ImportError:
            raise ImportError("trl is required for the DPO algorithm, but it is not available. Please install the library to continue.")
        
    @torch.no_grad()
    def on_train_start(self, trainer, model, config, average_log_probs=False) -> None:
        device = xm.xla_device()
        model.eval()
        reference_chosen_logps, reference_rejected_logps = [], []
        from torch_xla.distributed.parallel_loader import MpDeviceLoader
        with torch.no_grad():
            dl = MpDeviceLoader(trainer.train_dataloader, xm.xla_device())
            for batch_idx, batch in enumerate(dl):
                len_chosen = batch["chosen_labels"].shape[0]
                # batch size is doubled here, since we concatenate chosen and rejected responses on the batch dimension
                self.concatenated_batch = trl.DPOTrainer.concatenated_inputs(batch, device=device)
                outputs = model(self.concatenated_batch['concatenated_input_ids'], attention_mask=self.concatenated_batch['concatenated_attention_mask'], use_cache=False)
                all_logits = outputs[0].clone().detach()
                try:
                    all_logps = from_parallel_logits_to_logprobs(vocab_parallel_logits=all_logits, target=self.concatenated_batch['concatenated_labels'].to(device), inference=False)
                    loss_mask = self.concatenated_batch['concatenated_labels'][:, 1:] != -100
                    mask_sum = loss_mask.sum(-1)
                    if average_log_probs:
                        all_logps = (all_logps * loss_mask).sum(-1) / torch.where(mask_sum < 1, torch.ones_like(mask_sum), mask_sum)
                    else:
                        all_logps = (all_logps * loss_mask).sum(-1)
                except Exception:
                    all_logps, _ = self.get_batch_logps(all_logits, self.concatenated_batch['concatenated_labels'].to(device))

                # splitting chosen/rejected responses from concatenated output
                chosen_logps, rejected_logps = torch.split(all_logps, [len_chosen, len(all_logps) - len_chosen])
            
                chosen_logps, rejected_logps = chosen_logps.cpu(), rejected_logps.cpu()
                reference_chosen_logps.append(chosen_logps)
                reference_rejected_logps.append(rejected_logps)
                del chosen_logps, rejected_logps
                del all_logits, all_logps
            
            xm.mark_step()
            all_reference_chosen_logps = torch.cat(reference_chosen_logps).numpy()
            all_reference_rejected_logps = torch.cat(reference_rejected_logps).numpy()

        trainer.datamodule.data_sets['train'] = trainer.datamodule.data_sets['train'].add_column("reference_chosen_logps", all_reference_chosen_logps)
        trainer.datamodule.data_sets['train'] = trainer.datamodule.data_sets['train'].add_column("reference_rejected_logps", all_reference_rejected_logps)
        trainer.fit_loop.setup_data(updated_data_source=_DataLoaderSource(trainer.datamodule, "train_dataloader"))
        del reference_chosen_logps, reference_rejected_logps   
        del all_reference_chosen_logps, all_reference_rejected_logps
        model.train()

    def policy_model_concatenated_forward(self, model, batch, average_log_probs=False):
        import trl
        device = xm.xla_device()
        len_chosen = batch["chosen_labels"].shape[0]
        self.concatenated_batch = trl.DPOTrainer.concatenated_inputs(batch, device=device)
        outputs = model(self.concatenated_batch['concatenated_input_ids'], attention_mask=self.concatenated_batch['concatenated_attention_mask'], use_cache=False)
        all_logits = outputs[0]
        try:
            all_logps = from_parallel_logits_to_logprobs(vocab_parallel_logits=all_logits, target=self.concatenated_batch['concatenated_labels'].to(device), inference=False)
            loss_mask = self.concatenated_batch['concatenated_labels'][:, 1:] != -100
            mask_sum = loss_mask.sum(-1)
            if average_log_probs:
                all_logps = (all_logps * loss_mask).sum(-1) / torch.where(mask_sum < 1, torch.ones_like(mask_sum), mask_sum)
            else:
                all_logps = (all_logps * loss_mask).sum(-1)
        except Exception:
            all_logps, _ = self.get_batch_logps(all_logits, self.concatenated_batch['concatenated_labels'].to(device))

        policy_chosen_logps, policy_rejected_logps = torch.split(all_logps, [len_chosen, len(all_logps) - len_chosen])

        return policy_chosen_logps, policy_rejected_logps, all_logits

    def model_fwd_calc_loss(self, model, batch, config):
        device = xm.xla_device()
        policy_chosen_logps, policy_rejected_logps, _ = self.policy_model_concatenated_forward(model, batch, average_log_probs=False)

        reference_chosen_logps, reference_rejected_logps = batch["reference_chosen_logps"], batch["reference_rejected_logps"]
        kl_beta = config.model_alignment_strategy.dpo.kl_beta

        pi_logratios = policy_chosen_logps - policy_rejected_logps
    
        ref_logratios = reference_chosen_logps - reference_rejected_logps

        pi_logratios = pi_logratios.to(device)
        ref_logratios = ref_logratios.to(device)
        logits = pi_logratios - ref_logratios
        loss = -torch.nn.functional.logsigmoid(kl_beta * logits).mean(0)
        chosen_rewards = kl_beta * (policy_chosen_logps.to(device) - reference_chosen_logps.to(device)).detach()
        rejected_rewards = kl_beta * (policy_rejected_logps.to(device) - reference_rejected_logps.to(device)).detach()
        misc_metrics = {'chosen_rewards': chosen_rewards, 'rejected_rewards': rejected_rewards}

        return loss, misc_metrics

    def get_batch_logps(
        self,
        logits,
        labels,
        label_pad_token_id: int = -100,
    ):
        """Compute the log probabilities of the given labels under the given logits.

        Args:
            logits: Logits of the model (unnormalized). Shape: (batch_size, sequence_length, vocab_size)
            labels: Labels for which to compute the log probabilities. Label tokens with a value of label_pad_token_id are ignored. Shape: (batch_size, sequence_length)
            label_pad_token_id: The label pad token id.
            is_encoder_decoder: Whether the model is an encoder-decoder model.

        Returns:
            A Tuple of two tensor of shape ((batch_size,), (batch_size,)) containing the sum of log probabilities of the given labels under the given logits in the first tensor and the number of non-masked tokens in the second tensor.
        """
        if logits.shape[:-1] != labels.shape:
            raise ValueError(
                f"Logits (batch and sequence length dim) {logits.shape[:-1]} and labels must have the same shape {labels.shape}."
            )
       
        labels = labels[:, 1:].clone()
        logits = logits[:, :-1, :]
        loss_mask = labels != label_pad_token_id

        # dummy token; we'll ignore the losses on these tokens later
        labels = torch.where(labels == label_pad_token_id, torch.tensor(0, device=labels.device), labels)

        per_token_logps = torch.gather(logits.log_softmax(-1), dim=2, index=labels.unsqueeze(2)).squeeze(2)

        return (per_token_logps * loss_mask).sum(-1), loss_mask.sum(-1)
