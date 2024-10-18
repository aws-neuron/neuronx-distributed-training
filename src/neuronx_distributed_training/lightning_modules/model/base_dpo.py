import neuronx_distributed as nxd
import torch
from transformers import LlamaConfig
import torch.nn.functional as F
import os
import torch_xla.core.xla_model as xm
from neuronx_distributed.parallel_layers.loss_functions import from_parallel_logits_to_logprobs
import logging
from .base import BaseModelModule
from omegaconf import DictConfig, open_dict
from pytorch_lightning.trainer.trainer import Trainer
import numpy as np

logger = logging.getLogger(__name__)


class DPOBaseModel(BaseModelModule):

    def __init__(self, cfg: DictConfig, trainer: Trainer, model=None):
        try:
            global trl
            import trl
        except ImportError:
            raise ImportError("trl is required for the DPO algorithm, but it is not available. Please install the library to continue.")
        
    @torch.no_grad()
    def on_train_start(self, trainer, model, config) -> None:
        device = xm.xla_device()
        model.eval()
        reference_chosen_logps, reference_rejected_logps = [], []
        with torch.no_grad():
            for batch_idx, batch in enumerate(trainer.train_dataloader):
                len_chosen = batch["chosen_labels"].shape[0]
                # batch size is doubled here, since we concatenate chosen and rejected responses on the batch dimension
                concatenated_batch = trl.DPOTrainer.concatenated_inputs(batch, device=device)
                outputs = model(concatenated_batch['concatenated_input_ids'], attention_mask=concatenated_batch['concatenated_attention_mask'], use_cache=False)
                xm.mark_step()
                try:
                    all_logits = outputs[0].clone().detach()
                    all_logps, _ = self.get_batch_logps(all_logits, concatenated_batch['concatenated_labels'].to(device))
                except Exception:
                    all_logps = from_parallel_logits_to_logprobs(vocab_parallel_logits=outputs[0][..., :-1, :], target=concatenated_batch['concatenated_labels'].to(device), inference=True)
                    loss_mask = (concatenated_batch['concatenated_labels'][..., 1:] > -1).to(torch.bfloat16)
                    all_logps = (all_logps * loss_mask).sum(-1)
                
                # splitting chosen/rejected responses from concatenated output
                chosen_logps, rejected_logps = torch.split(all_logps, [len_chosen, len(all_logps) - len_chosen])
               
                chosen_logps, rejected_logps = chosen_logps.cpu(), rejected_logps.cpu()
                reference_chosen_logps.append(chosen_logps)
                reference_rejected_logps.append(rejected_logps)
                del chosen_logps, rejected_logps
                del all_logits, all_logps
        
            all_reference_chosen_logps = torch.cat(reference_chosen_logps).numpy()
            all_reference_rejected_logps = torch.cat(reference_rejected_logps).numpy()

        trainer.datamodule.data_sets['train'] = trainer.datamodule.data_sets['train'].add_column("reference_chosen_logps", all_reference_chosen_logps)
        trainer.datamodule.data_sets['train'] = trainer.datamodule.data_sets['train'].add_column("reference_rejected_logps", all_reference_rejected_logps)
        trainer.reset_train_dataloader(trainer.lightning_module)
        del reference_chosen_logps, reference_rejected_logps   
        del all_reference_chosen_logps, all_reference_rejected_logps                
        model.train()

    def model_fwd_calc_loss(self, model, batch, config):
        device = xm.xla_device()
        len_chosen = batch["chosen_labels"].shape[0]
        concatenated_batch = trl.DPOTrainer.concatenated_inputs(batch, device=device)
        outputs = model(concatenated_batch['concatenated_input_ids'], attention_mask=concatenated_batch['concatenated_attention_mask'], use_cache=False)
        xm.mark_step()
        try:
            all_logits = outputs[0]
            all_logps, _ = self.get_batch_logps(all_logits, concatenated_batch['concatenated_labels'].to(device))
        except Exception:
            all_logps = from_parallel_logits_to_logprobs(vocab_parallel_logits=outputs[0][..., :-1, :], target=concatenated_batch['concatenated_labels'].to(device), inference=False)
            loss_mask = (concatenated_batch['concatenated_labels'][..., 1:] > -1).to(torch.bfloat16)
            all_logps = (all_logps * loss_mask).sum(-1)

        policy_chosen_logps, policy_rejected_logps = torch.split(all_logps, [len_chosen, len(all_logps) - len_chosen])

        reference_chosen_logps, reference_rejected_logps = batch["reference_chosen_logps"], batch["reference_rejected_logps"]
        self.beta = config.data.alignment_strategy.dpo.kl_beta
        pi_logratios = policy_chosen_logps - policy_rejected_logps
    
        ref_logratios = reference_chosen_logps - reference_rejected_logps

        pi_logratios = pi_logratios.to(device)
        ref_logratios = ref_logratios.to(device)
        logits = pi_logratios - ref_logratios
       
        loss = -torch.nn.functional.logsigmoid(self.beta * logits).mean(0)

        return loss

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
