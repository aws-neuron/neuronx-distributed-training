"""Megatron style model checkpoint conversion with HF (Mixtral)
a) NxDT -> HF

.. code-block:: console

    python3 hf_nxdt_mixtral_ckpt_converter.py --model_style megatron --convert_to_full_state --input_dir mixtral_checkpoints --output_dir converted_checkpoints --nxdt_yaml_config conf/mixtral_8x7b_config.yaml --load_xser True --tp_size 32 --pp_size 4 --ep_size 1 --n_layers 32

b) HF -> NxDT

.. code-block:: console

    python3 hf_nxdt_mixtral_ckpt_converter.py --model_style megatron --input_dir mixtral-8x7B-hf --output_dir converted_checkpoints --config mixtral-8x7B-hf/config.json --save_xser True --convert_from_full_state --tp_size 32 --pp_size 4 --ep_size 1 --n_layers 32

If you do not have a config.json, e.g. because of using an NxDT custom model, and want to convert to some other checkpoint,
use the ``--nxdt_yaml_config <path to yaml>`` option, and it will convert the yaml into the json format the script
uses to read the config.
"""

import torch
import json
import argparse
from neuronx_distributed.scripts.checkpoint_converter import CheckpointConverterBase


class CheckpointConverterNxDTMixtral(CheckpointConverterBase):
    """This function uses the NxD checkpoint converter to convert from
    HF to Megatron NxDT checkpoint and back. It overrides the following
    two functions in order to account for the naming differences in the Mixtral
    models between HF and Megatron.
    """

    # ExpertFusedColumnParallelLinear
    gate_up_proj_partition_dim = 2
    # ExpertFusedRowParallelLinear
    down_proj_partition_dim = 1

    def pre_process_full_state_before_tp_conversion(self, state_dict, args):
        """Stack the MLP weights across experts as expected by the MoE module."""

        with open(args.config, "r") as f:
            config = json.load(f)

        for i in range(config["num_hidden_layers"]):
            router_weight = state_dict.pop(f"model.layers.{i}.block_sparse_moe.gate.weight")

            gate_proj_per_expert = []
            up_proj_per_expert = []
            down_proj_per_expert = []
            for j in range(config["num_local_experts"]):
                gate_proj_per_expert.append(state_dict.pop(f"model.layers.{i}.block_sparse_moe.experts.{j}.w1.weight"))
                down_proj_per_expert.append(state_dict.pop(f"model.layers.{i}.block_sparse_moe.experts.{j}.w2.weight"))
                up_proj_per_expert.append(state_dict.pop(f"model.layers.{i}.block_sparse_moe.experts.{j}.w3.weight"))
            gate_proj = torch.stack(gate_proj_per_expert)
            up_proj = torch.stack(up_proj_per_expert)
            down_proj = torch.stack(down_proj_per_expert)

            state_dict[f"model.layers.{i}.mlp.moe.router.linear_router.weight"] = router_weight
            state_dict[f"model.layers.{i}.mlp.moe.expert_mlps.mlp_op.gate_proj.weight"] = gate_proj
            state_dict[f"model.layers.{i}.mlp.moe.expert_mlps.mlp_op.up_proj.weight"] = up_proj
            state_dict[f"model.layers.{i}.mlp.moe.expert_mlps.mlp_op.down_proj.weight"] = down_proj

        return state_dict

    def post_process_full_state_after_tp_conversion(self, state_dict, args):
        """Split the MLP weights across experts."""

        with open(args.config, "r") as f:
            config = json.load(f)

        for i in range(config["num_hidden_layers"]):
            router_weight = state_dict.pop(f"model.layers.{i}.mlp.moe.router.linear_router.weight")
            gate_proj = state_dict.pop(f"model.layers.{i}.mlp.moe.expert_mlps.mlp_op.gate_proj.weight")
            up_proj = state_dict.pop(f"model.layers.{i}.mlp.moe.expert_mlps.mlp_op.up_proj.weight")
            down_proj = state_dict.pop(f"model.layers.{i}.mlp.moe.expert_mlps.mlp_op.down_proj.weight")

            gate_proj_per_expert = torch.unbind(gate_proj)
            up_proj_per_expert = torch.unbind(up_proj)
            down_proj_per_expert = torch.unbind(down_proj)
            for j in range(config["num_local_experts"]):
                state_dict[f"model.layers.{i}.block_sparse_moe.experts.{j}.w1.weight"] = gate_proj_per_expert[j]
                state_dict[f"model.layers.{i}.block_sparse_moe.experts.{j}.w2.weight"] = down_proj_per_expert[j]
                state_dict[f"model.layers.{i}.block_sparse_moe.experts.{j}.w3.weight"] = up_proj_per_expert[j]
            state_dict[f"model.layers.{i}.block_sparse_moe.gate.weight"] = router_weight

        return state_dict


if __name__ == "__main__":
    checkpoint_converter = CheckpointConverterNxDTMixtral()
    parser = checkpoint_converter.get_arg_parser()
    args, _ = parser.parse_known_args()
    checkpoint_converter.run(args)