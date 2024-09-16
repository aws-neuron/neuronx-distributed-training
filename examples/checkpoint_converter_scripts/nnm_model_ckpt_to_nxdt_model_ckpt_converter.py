'''
Example Usage:
1. Parallel processing: python nnm_nxdt_ckpt_converter.py --tp 8 --pp 4 --n_layers 32 --nnm_ckpt_path {path_to_ckpt}/ckpt/nnm --nxdt_ckpt_path {path_to_ckpt}/nnm-converted-nxdt-ckpt/ --enable_parallel_processing True --num_parallel_processes 8
2. Non-parallel: python nnm_nxdt_ckpt_converter.py --tp 8 --pp 4 --n_layers 32 --nnm_ckpt_path {path_to_ckpt}/ckpt/nnm --nxdt_ckpt_path {path_to_ckpt}/ckpt/nnm-converted-nxdt-ckpt/ 

Note: DP is not yet handled, also PP=1 scenario
'''


import argparse
import torch
import os
from torch_xla.utils.serialization import TensorReference
import time
from multiprocessing import Pool
from itertools import product
from tqdm import tqdm


def parse_arguments():
    """
    Parses command-line arguments for TP, PP, checkpoint paths, and number of parallel processes.
    
    Returns:
      argparse.Namespace: An object containing parsed arguments or None if help is requested.
    """
    
    parser = argparse.ArgumentParser(description="Script description (replace with your description)")
    
    # Required arguments
    parser.add_argument("--tp", type=int, required=True, help="Number of TP ranks")
    parser.add_argument("--pp", type=int, required=True, help="Number of PP ranks")
    parser.add_argument("--n_layers", type=int, required=True, help="Number of total layers in the model")
    parser.add_argument("--nnm_ckpt_path", type=str, required=True, help="Path to the NNM checkpoint directory")
    parser.add_argument("--nxdt_ckpt_path", type=str, required=True, help="Path to the NxDT checkpoint directory")
    
    # Optional arguments with default value
    parser.add_argument("--enable_parallel_processing", type=bool, default=False, help="Runs in parallel with specified number of threads")
    parser.add_argument("--num_parallel_processes", type=int, default=os.cpu_count()-5, help="Number of parallel processes (default: number of CPU cores)")
    
    args = parser.parse_args()
    return args

def check_and_create_output_folder(output_path):
    """
    Checks the output folder and creates it if it doesn't exist.
    If it exists, checks if it's empty. If not empty, creates a version folder
    with an incremented version number.
    
    Args:
      output_path (str): Path to the desired output folder.
    
    Returns:
      str: The final output folder path (including version if created).
    """
    
    # Create base path if it doesn't exist
    if not os.path.exists(output_path):
        os.makedirs(output_path)
        return output_path
    
    # Check if base path is empty
    if not os.listdir(output_path):
        return output_path
    
    # Generate version folder name with incremented number
    version = 1
    version_folder = f"{output_path}_v{version}"
    while os.path.exists(version_folder):
        version += 1
        version_folder = f"{output_path}_v{version}"
    
    # Create version folder
    os.makedirs(version_folder)
    return version_folder

def populate_each_worker(nnm_model_folder, nxdt_model_folder, dp, tp, pp, num_layers_per_pipeline):
    """
    Populates each worker with three files:
    1. dp_rank_00_tp_rank_00_pp_rank_00.pt: tid mapped
    2. dp_rank_00_tp_rank_00_pp_rank_00.pt.info.pt: tid shape and dtype mapped
    3. dp_rank_00_tp_rank_00_pp_rank_00.pt.tensors/tensor_{tid}.pt: actual tensor
    """
    
    # Load the NNM model state dictionary efficiently
    nnm_worker_pt = torch.load(f"{nnm_model_folder}/tp_rank_{tp:02d}_pp_rank_{pp:03d}/model_optim_rng.ckpt")["state_dict"]
    
    # Create tid mappers with dictionary comprehension
    tid_mapper = {k: i for i, k in enumerate(nnm_worker_pt.keys())}
    tid_nxdt_mapper = {
        modify_layer_string(k, pp * num_layers_per_pipeline): TensorReference(v)
        for k, v in tid_mapper.items()
    }
    
    # Save tid mappers with a single call
    torch.save(tid_nxdt_mapper, f"{nxdt_model_folder}/dp_rank_{dp:02d}_tp_rank_{tp:02d}_pp_rank_{pp:02d}.pt")
    
    # Create tid_dtype_shape_mapper with dictionary comprehension
    tid_dtype_shape_mapper = {v: {'dtype': nnm_worker_pt[k].dtype, 'shape': nnm_worker_pt[k].shape}
                           for k, v in tid_mapper.items()}
    
    # Save tid_dtype_shape_mapper with a single call
    torch.save(tid_dtype_shape_mapper, f"{nxdt_model_folder}/dp_rank_{dp:02d}_tp_rank_{tp:02d}_pp_rank_{pp:02d}.info.pt")
    
    # Create the tensor folder if it doesn't exist (avoid redundant checks)
    tensor_folder = os.path.join(nxdt_model_folder, f"dp_rank_{dp:02d}_tp_rank_{tp:02d}_pp_rank_{pp:02d}.pt.tensors")
    os.makedirs(tensor_folder, exist_ok=True)

    # Save tensors using dictionary comprehension and string formatting
    for k, v in tid_mapper.items():
        torch.save(nnm_worker_pt[k], os.path.join(tensor_folder, f"tensor_{v}.pt"))

def populate_workers(nnm_ckpt_path, nxdt_model_folder, dp, TP, PP, num_layers_per_pipeline):
    """
    Populates all workers with the corresponding files.
    
    Args:
      nnm_ckpt_path (str): Path to the NNM checkpoint directory.
      nxdt_model_folder (str): Base path for the NxDT model directory.
      dp (int, optional): Device placement (default: 0).
    """
        
    # Use product for efficient iteration over all TP and PP combinations
    total_workers = TP * PP
    with tqdm(total=total_workers, desc="Progress of Workers") as pbar:
        for tp, pp in product(range(TP), range(PP)):
            populate_each_worker(nnm_ckpt_path, nxdt_model_folder, dp, tp, pp, num_layers_per_pipeline)
            pbar.update(1)

def populate_workers_parallel(nnm_ckpt_path, nxdt_model_folder, dp, tp, pp, num_layers_per_pipeline):
    populate_each_worker(nnm_ckpt_path, nxdt_model_folder, dp, tp, pp, num_layers_per_pipeline)

def modify_layer_string(input_string, layer_offset):
    """
    Modifies a string representing a layer path in a model by adjusting the layer index.
    
    Args:
      input_string (str): The original string representing the layer path (e.g., 'model.language_model.encoder.layers.0.self_attention.dense.weight').
      layer_offset (int): The offset value to apply to the layer index (e.g., 0 for no change, 8 for layers.8 to layers.15).
    
    Returns:
      str: The modified string with the adjusted layer index.
    """
    
    input_string=input_string.replace("model.language_model", "language_model")
    parts = input_string.split('.')
    for i, part in enumerate(parts):
        if part.isdigit():
            # Found layer index
            new_layer_index = int(part) + int(layer_offset)
            parts[i] = str(new_layer_index)
            break  # Modify only the first layer index occurrence
    return '.'.join(parts)

if __name__ == "__main__":
    parsed_args = parse_arguments()
    
    # Exit if help was requested (--help)
    if parsed_args is None:
        exit(0)
    
    # Access arguments
    TP = parsed_args.tp
    PP = parsed_args.pp
    n_layers = parsed_args.n_layers
    num_layers_per_pipeline = n_layers/PP
    nnm_ckpt_path = parsed_args.nnm_ckpt_path
    nxdt_ckpt_path = parsed_args.nxdt_ckpt_path
    num_parallel_processes = parsed_args.num_parallel_processes
    enable_parallel_processing = parsed_args.enable_parallel_processing

    # validate the TP, PP folders in the nnm folder

    actual = set(os.listdir(nnm_ckpt_path))
    expected = set([f"tp_rank_{i:02d}_pp_rank_{j:03d}" for i in range(TP) for j in range(PP)])
    assert expected == actual, "The folders expected and actual inside nnm directory are not the same. expected contains: {} and actual contains: {}".format(expected, actual) 

    # check output folder and create one if it doesnt exists, if exists check empty, if not empty create version folder
    nxdt_ckpt_path = check_and_create_output_folder(nxdt_ckpt_path)
    print(f"Output folder chosen: {nxdt_ckpt_path}")

    # create checkpoint and done files with 1 inside
    with open(f"{nxdt_ckpt_path}/done", "w") as file:
      file.write("1")
    
    with open(f"{nxdt_ckpt_path}/checkpoint", "w") as file:
      file.write("1")

    # create ckpt model folder
    nxdt_model_folder = f"{nxdt_ckpt_path}/model"
    if not os.path.exists(nxdt_model_folder):
        os.makedirs(nxdt_model_folder)

    # call workers for convertion
    start=time.time()
    if enable_parallel_processing is False:
        populate_workers(nnm_ckpt_path, nxdt_model_folder, 0, TP, PP, num_layers_per_pipeline) #
    else:
        print(f"Started Parallel processing, sometimes workers might not get pooled in, so do check the progress in output path={nxdt_ckpt_path}")
        with Pool(processes=num_parallel_processes) as pool:
            # Unpack the tuple elements into separate arguments
            pool.starmap(populate_workers_parallel, [(nnm_ckpt_path, nxdt_model_folder, 0, tp, pp, num_layers_per_pipeline) for tp in range(TP) for pp in range(PP)])
        pool.close()
    print(f"Successfully completed convertion from NNM to NxD in {(time.time()-start)} seconds and output is in {nxdt_ckpt_path}")

