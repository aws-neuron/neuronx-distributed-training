import neuronx_distributed as nxd
import torch
from transformers.models.mixtral.configuration_mixtral import MixtralConfig

from neuronx_distributed.modules.moe.model import MoE
from neuronx_distributed.parallel_layers import mappings 
from neuronx_distributed.modules.moe.loss_function import load_balancing_loss_func
from neuronx_distributed_training.models.hf_models.modeling_mixtral import (
    CoreAttention,
    MixtralDecoderLayer,
    MixtralForCausalLM,
    MixtralRMSNorm,
    LlamaMLP
)

from .base_model import BaseHfModel

class HFMixtralModule(BaseHfModel):
    def _get_model(self):
        config = MixtralConfig.from_pretrained(self.config.model.model_config)
        config.return_dict = False
        config.sequence_parallel_enabled = self.config.distributed_strategy.get("sequence_parallel", False)
        config.qkv_linear = self.config.model.get("qkv_linear", False)
        config.kv_shared_group_size = self.config.distributed_strategy.get("kv_replicator", 1)
        config.max_position_embeddings = max(config.max_position_embeddings, self.config.model.get("max_position_embeddings"))
        config.use_flash_attention = self.config.model.fusions.flash_attention
        if self.config.model.get('num_layers', -1) != -1:
            config.num_hidden_layers = self.config.model.get('num_layers')
        if self.config.model.get('hidden_size', -1) != -1:
            config.hidden_size = self.config.model.get('hidden_size')
        if self.config.model.get('rope_theta', -1) != -1:
            config.rope_theta = self.config.model.get('rope_theta')

        config.num_local_experts = self.config.model.moe.get('num_experts', 1)
        config.moe_frequency = self.config.model.moe.get('frequency', 1)
        config.attention_dropout = self.config.model.moe.get('dropout', 0)
        config.capacity_factor = self.config.model.moe.get('capacity_factor', 1.0)
        config.num_experts_per_tok = self.config.model.moe.get('top_k', 1)
        config.output_router_logits = self.config.model.moe.get('output_router_logits', True)
        config.router_aux_loss_coef = self.config.model.moe.get('router_aux_loss_coef', 0.02)
        config.normalize_top_k_affinities = self.config.model.moe.get('normalize_top_k_affinities', True)

        leaf_module_cls = [MixtralRMSNorm.__name__]
        if self.config.model.get("activations_checkpoint_granularity", None) == "selective":
            if self.config.model.get("activations_checkpoint_recompute_mlp", False) and self.config.model.encoder_seq_length>=8192:
                self.nxd_config["activation_checkpoint_config"] = (CoreAttention, MoE)
            else:
                self.nxd_config["activation_checkpoint_config"] = CoreAttention
        elif self.config.model.get("activations_checkpoint_granularity", None) == "full":
            self.nxd_config["activation_checkpoint_config"] = "full"
        self.nxd_config["pipeline_config"].update(
            {
                "transformer_layer_cls": MixtralDecoderLayer,
                "output_loss_value_spec": (True, False, False, False),
                "input_names": ["input_ids", "labels"],
                "leaf_module_cls": leaf_module_cls,
            }
        )
        return nxd.initialize_parallel_model(self.nxd_config, self.model_provider_func, config)

    def model_provider_func(self, config):
        return MixtralForCausalLM(config)
    
    def init_weights(self, module, device):
        """
        Re-init weights after partition
        Referred from HF transformers https://github.com/huggingface/transformers/blob/v4.36.1/src/transformers/models/mixtral/modeling_mixtral.py#L849
        """
        # Last else should always call super().init_weights() to allow initializing
        # pre-defined layers.
        if isinstance(module, MixtralRMSNorm):
            module.weight.data.fill_(1.0)
        else:
            super().init_weights(module, device)
