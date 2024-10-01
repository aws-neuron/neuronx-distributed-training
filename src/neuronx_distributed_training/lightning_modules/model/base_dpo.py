import neuronx_distributed as nxd
import torch
from transformers import LlamaConfig
import torch.nn.functional as F
import os
import torch_xla.core.xla_model as xm
from neuronx_distributed.parallel_layers.loss_functions import from_parallel_logits_to_logprobs
import trl
from .base import BaseModelModule
from omegaconf import DictConfig, open_dict
from pytorch_lightning.trainer.trainer import Trainer
import numpy as np

class DPOBaseModel(BaseModelModule):

    def __init__(self, cfg: DictConfig, trainer: Trainer, model=None):
        pass
        
    @torch.no_grad()
    def on_train_start(self, trainer, model, config) -> None:
        device = xm.xla_device()
        model.eval()
        reference_chosen_logps, reference_rejected_logps = [], []
        with torch.no_grad():
            for batch_idx, batch in enumerate(trainer.train_dataloader):
                len_chosen = batch["chosen_labels"].shape[0]           
                concatenated_batch = trl.DPOTrainer.concatenated_inputs(batch, device=device)
                outputs = model(concatenated_batch['concatenated_input_ids'], attention_mask=concatenated_batch['concatenated_attention_mask'], use_cache=False)
                
                
                all_logps = from_parallel_logits_to_logprobs(vocab_parallel_logits=outputs[0], target=concatenated_batch['concatenated_labels'].to(device))
                loss_mask = (concatenated_batch['concatenated_labels'].to(device) > -1).float()
                all_logps = (all_logps * loss_mask).sum(-1)
               
                chosen_logps, rejected_logps = torch.split(all_logps.float(), [len_chosen, len(all_logps) - len_chosen])
               
                chosen_logps, rejected_logps = chosen_logps.cpu(), rejected_logps.cpu()
                reference_chosen_logps.append(chosen_logps)
                reference_rejected_logps.append(rejected_logps)
        
            all_reference_chosen_logps = torch.cat(reference_chosen_logps).float().numpy()
            all_reference_rejected_logps = torch.cat(reference_rejected_logps).float().numpy()

        trainer.datamodule.data_sets['train'] = trainer.datamodule.data_sets['train'].add_column("reference_chosen_logps", all_reference_chosen_logps)
        trainer.datamodule.data_sets['train'] = trainer.datamodule.data_sets['train'].add_column("reference_rejected_logps", all_reference_rejected_logps)
        trainer.reset_train_dataloader(trainer.lightning_module)
        del reference_chosen_logps, reference_rejected_logps   
        del all_reference_chosen_logps, all_reference_rejected_logps                
        xm.mark_step()
        model.train()

    def model_fwd_calc_loss(self, model, batch, config):
        device = xm.xla_device()
       
        concatenated_batch = trl.DPOTrainer.concatenated_inputs(batch, device=device)
        outputs = model(concatenated_batch['concatenated_input_ids'], attention_mask=concatenated_batch['concatenated_attention_mask'], use_cache=False)
        
        all_logps = from_parallel_logits_to_logprobs(vocab_parallel_logits=outputs[0], target=concatenated_batch['concatenated_labels'].to(device))
        loss_mask = (concatenated_batch['concatenated_labels'].to(device) > -1).float()
        all_logps = (all_logps * loss_mask).sum(-1)

        xm.mark_step()
      
        policy_chosen_logps = all_logps[0]
        policy_rejected_logps = all_logps[1]
        reference_chosen_logps, reference_rejected_logps = batch["reference_chosen_logps"], batch["reference_rejected_logps"]
        self.beta = config.data.alignment_strategy.dpo.kl_beta
        pi_logratios = policy_chosen_logps - policy_rejected_logps
    
        ref_logratios = reference_chosen_logps - reference_rejected_logps

        pi_logratios = pi_logratios.to(device)
        ref_logratios = ref_logratios.to(device)
        logits = pi_logratios - ref_logratios
       
        loss = -torch.nn.functional.logsigmoid(self.beta * logits).mean(0)

        return loss
        