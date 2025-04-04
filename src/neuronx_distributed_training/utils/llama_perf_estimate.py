import argparse
import json
from transformers import LlamaConfig

def get_attention_flop(hidden_size, num_heads, num_key_value_heads, seq_len):
    """
    Calculate FLOPs for attention mechanism.
    Self attention:
    1. QKV https://tiny.amazon.com/1ethtm1t1
    for a linear layer we have the following relationship
    FWD_MACs = num_params * auxiliary_dimensions
    FWD_FLOPs = 2 * num_params * auxiliary_dimensions
    """
    head_size = hidden_size // num_heads
    query_param_size = hidden_size**2
    key_value_param_size = num_key_value_heads * head_size * hidden_size 
    
    # Query, Key, and Value projections
    flops_query = 2 * query_param_size * seq_len
    flops_key = 2 * key_value_param_size * seq_len
    flops_value = 2 * key_value_param_size * seq_len
    
    # Attention operations
    # QK^T https://tiny.amazon.com/13ga4z1h6 
    flops_attn_qk = 2 * (seq_len**2) * hidden_size
    # softmax - 2 is because of float32
    flops_attn_softmax = 2*3 * num_heads * (seq_len**2)
    # softmax * V https://tiny.amazon.com/zvtjn1du
    flops_attn_soft_v = 2 * (seq_len**2) * hidden_size
    
    # Final projection
    # Final linear https://tiny.amazon.com/1rva783v
    # Again this follows the linear layer rule
    flops_attn_final = 2 * (hidden_size**2) * seq_len

    return (flops_query + flops_key + flops_value + flops_attn_qk + 
            flops_attn_softmax + flops_attn_soft_v + flops_attn_final)

def get_mlp_flops(hidden_size, intermediate_size, seq_len):
    """
    Calculate FLOPs for MLP layers.
    """
    num_params = intermediate_size * hidden_size
    flops_per_linear = 2 * num_params * seq_len
    # Three linear layers (gate_proj, up_proj, down_proj)
    return flops_per_linear * 3

def llama2_flops_per_seq(config, seq_len, output_logits=True):
    """
    Calculate total FLOPs for a LLaMA model forward and backward pass.
    https://tiny.amazon.com/147m7kwhy
    """
    att_flops = get_attention_flop(config.hidden_size, config.num_attention_heads, 
                                   config.num_key_value_heads, seq_len)
    mlp_flops = get_mlp_flops(config.hidden_size, config.intermediate_size, seq_len)
    
    # Input embedding
    flops_input_embb = 2 * (config.vocab_size * config.hidden_size) * seq_len
    
    # Output logits (if required)
    flops_output_embb = 2 * (config.vocab_size * config.hidden_size) * seq_len if output_logits else 0
    
    # Forward pass
    FWD = config.num_hidden_layers * (att_flops + mlp_flops) + flops_input_embb + flops_output_embb
    
    # Backward pass (approximately 2x forward + MLP gradients)
    BWD = 2 * FWD

    return FWD + BWD

def calculate_mfu(config_path, batch_size, throughput, num_nodes, seq_len, hw_backend):
    """
    Calculate Model FLOPs Utilization (MFU) for LLaMA models.
    """
    # Load model configuration
    with open(config_path, 'r') as f:
        config_dict = json.load(f)
    config = LlamaConfig(**config_dict)

    # Calculate FLOPs per sequence
    flops_per_seq = llama2_flops_per_seq(config, seq_len)
    tflops_per_second = flops_per_seq / 10**12

    # Calculate throughput and utilization
    time_per_batch = batch_size / throughput
    seq_per_second_per_node = batch_size / (time_per_batch * num_nodes)
    throughput_per_node = (batch_size / num_nodes) * (tflops_per_second / time_per_batch)

    # Set peak FLOPS based on hardware backend
    if hw_backend == 'trn1':
        # 95 TFLOPs per core, 32 devices per node
        peak_flops = 3040
    elif hw_backend == 'trn2':
        # 667 TFLOPs per 8 cores, 128 devices per node
        peak_flops = 10672
    elif hw_backend == 'p5':
        peak_flops = 8000
    else:
        raise ValueError("Unsupported hardware backend")

    model_flops_utilization = throughput_per_node * 100 / peak_flops

    return model_flops_utilization, seq_per_second_per_node, throughput_per_node, tflops_per_second, time_per_batch

def main():
    parser = argparse.ArgumentParser(description="Calculate MFU for LLaMA models")
    parser.add_argument("--config", required=True, help="Path to the model config.json file")
    parser.add_argument("--batch_size", type=int, required=True, help="Batch size")
    parser.add_argument("--throughput", type=float, required=True, help="Throughput in sequences/second")
    parser.add_argument("--num_nodes", type=int, required=True, help="Number of nodes")
    parser.add_argument("--seq_len", type=int, required=True, help="Sequence length")
    parser.add_argument("--hw_backend", choices=['trn1', 'trn2', 'p5'], required=True, help="Hardware backend")

    args = parser.parse_args()

    # Print initial parameters
    print(f"Hardware Backend: {args.hw_backend}")
    print(f"Number of Nodes: {args.num_nodes}")
    print(f"Batch Size: {args.batch_size}")
    print(f"Throughput: {args.throughput} sequences/second")
    print(f"Sequence Length: {args.seq_len}")
    print("-" * 40)

    mfu, seq_per_second, throughput_per_node, tflops_per_second, time_per_batch = calculate_mfu(
        args.config, args.batch_size, args.throughput, args.num_nodes, args.seq_len, args.hw_backend
    )

    print(f"TFLOPs per second: {tflops_per_second:.2f}")
    print(f"Time per batch: {time_per_batch:.2f} seconds")
    print(f"Sequences per second per node: {seq_per_second:.2f}")
    print(f"Throughput per node (TFLOP/s): {throughput_per_node:.2f}")
    print(f"Model FLOPs Utilization (MFU): {mfu:.2f}%")

if __name__ == "__main__":
    main()