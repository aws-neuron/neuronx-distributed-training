'''
Usage::

1. HF style model (GQA models tested - llama3)
1a) HF -> nxdt ckpt (uses Nxd)
python3 ckpt_convert.py --model_style hf --input_dir /home/ubuntu/pretrained_llama_3_8B_hf/pytorch_model.bin --output_dir /home/ubuntu/converted_hf_style_hf_to_nxdt_tp8pp4/ --save_xser True --convert_from_full_state --config /home/ubuntu/pretrained_llama_3_8B_hf/config.json --tp_size 8 --pp_size 4 --n_layers 32 --kv_size_multiplier 1 --qkv_linear True --convert_from_full_state
1b) nxdt -> HF ckpt (uses Nxd)
python3 ckpt_convert.py --model_style hf --convert_to_full_state --input_dir ~/examples/nemo_experiments/hf_llama3_8B_SFT/2024-07-19_23-07-40/checkpoints/hf_llama3_8B--step=5-consumed_samples=160.0.ckpt/model --output_dir ~/converted_hf_style_nxdt_to_hf_tp8pp4/ --qkv_linear True --config ~/config.json --load_xser True --tp_size 8 --pp_size 4 --kv_size_multiplier 1

2. Megatron style model (Non-GQA models tested - Llama2)
2a) Nxdt -> HF ckpt (source code for nnm to nxdt is nnm_model_ckpt_to_nxdt_model_ckpt_converter)
python3 ckpt_convert.py --model_style megatron --convert_to_full_state --input_dir ~/examples/nemo_experiments/megatron_llama/2024-07-19_23-45-54/checkpoints/megatron_llama--step=5-consumed_samples=5120.0.ckpt/model --output_dir ~/megatron-tp8pp4-nxdt-to-hf3 --config ~/llama_tokenizer/config.json --load_xser True --tp_size 8 --pp_size 4 --kv_size_multiplier 1
2b) Hf->Nxdt Megatron ckpt
python3 ckpt_convert.py --model_style megatron --input_dir ~/megatron-tp8pp4-nxdt-to-hf3/checkpoint.pt --output_dir ~/meg_nxdt_hf3_nxdt2 --config ~/llama_tokenizer/config.json --save_xser True --convert_from_full_state --tp_size 8 --pp_size 4 --n_layers 32 --kv_size_multiplier 1 

3. Megatron style model (GQA models tested - LLama3)
3a) NXDT -> HF
python3 ckpt_convert.py  --model_style megatron --convert_to_full_state --input_dir ~/examples/nemo_experiments/megatron_llama/2024-07-23_21-07-30/checkpoints/megatron_llama--step=5-consumed_samples=5120.0.ckpt/model --output_dir ~/megatron-tp8pp4-nxdt-to-hf4 --config ~/llama_gqa/config.json --load_xser True --tp_size 8 --pp_size 4 --kv_size_multiplier 1 --qkv_linear True 
3b) HF->Nxdt
python3 ckpt_convert.py --model_style megatron --input_dir ~/megatron-tp8pp4-nxdt-to-hf4/checkpoint.pt --output_dir ~/meg_nxdt_hf3_nxdt3 --config ~/llama_gqa/config.json --save_xser True --convert_from_full_state --tp_size 8 --pp_size 4 --n_layers 32 --kv_size_multiplier 1 --model_style megatron --qkv_linear True
'''

import argparse
from neuronx_distributed.scripts.checkpoint_converter import CheckpointConverterBase

if __name__ == "__main__":
    ckpt_converter_obj = CheckpointConverterBase()
    parser = ckpt_converter_obj.get_arg_parser()

    args = parser.parse_args()
    ckpt_converter_obj.run(args)
