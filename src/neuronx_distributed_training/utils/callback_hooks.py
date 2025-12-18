import logging
import os
from typing import Any, Dict, List, Tuple, Union

import lightning
import numpy as np
import torch
import torch_xla.core.xla_model as xm
import torch_xla.runtime as xr
from neuronx_distributed.parallel_layers.parallel_state import (
    get_data_parallel_rank, get_pipeline_model_parallel_rank,
    get_tensor_model_parallel_group, get_tensor_model_parallel_rank,
    get_tensor_model_parallel_size)
from pytorch_lightning.callbacks import Callback
from torch.distributed import all_reduce


class NeuronHooksCallback(Callback, lightning.Callback):
    """
    A PyTorch Lightning callback for capturing and saving neural network layer activations and gradients.
    
    This callback provides functionality to hook into module boundaries of a model
    and capture their input/output activations during forward pass and gradients during backward pass.
    The captured data can be saved as raw tensors or statistical summaries for debugging and analysis.
    """
    
    def __init__(self, cfg: Any) -> None:
        """
        Initialize the NeuronHooksCallback.
        
        Args:
            cfg: Training configuration object containing hooks settings such as target layers,
                 dump directories, and processing options.
        """
        super().__init__()
        # Check if hooks configuration is not set and log error
        if cfg.hooks is None:
            logging.error("Hooks configuration not found. Cannot initialize NeuronHooksCallback.")
            raise ValueError("Hooks Configuration cannot be None")
        # Maps to store captured activations and gradients by layer name
        self.activations_map = {}
        self.gradients_map = {}
        # Set dump directory based on SLURM job ID if available
        self.hooks_dump_base_directory = f"./hooks_dump/{os.environ.get('SLURM_JOB_ID')}" \
                if os.environ.get('SLURM_JOB_ID') is not None else "./hooks_dump"
        self.cfg = cfg
        self.pl_module = None

    def process_inputs_outputs(self, inputs: Any, outputs: Any) -> Tuple[Union[Tuple[torch.Tensor, ...], Dict[str, Tuple[torch.Tensor, ...]]], Union[Tuple[torch.Tensor, ...], Dict[str, Tuple[torch.Tensor, ...]]]]:
        """
        Process layer inputs and outputs by extracting tensors and computing statistics or gathering data.
        
        This method handles the processing of layer inputs and outputs based on configuration settings.
        It can either compute statistical summaries (norms, means, max, min) or return raw tensors
        with optional all-gather operations for distributed training.
        
        Args:
            inputs: Raw inputs from the layer 
            outputs: Raw outputs from the layer
            
        Returns:
            A tuple containing processed inputs and outputs. The format depends on configuration:
            - If dump_only_summaries=True: Returns dictionaries with statistical summaries
            - Otherwise: Returns tuples of tensors (potentially all-gathered across tensor parallel ranks)
        """
        # recursively ensure only torch.Tensors exist in input/output lists
        def _extract_tensor_list(data: Any) -> List[torch.Tensor]:
            """
            Recursively extract torch.Tensor objects from nested data structures.
            
            Args:
                data: Input data that may contain tensors in nested structures
                
            Returns:
                List of extracted torch.Tensor objects
            """
            list_of_tensors = []
            if isinstance(data, torch.Tensor):
                data_detach = data.detach().clone()
                list_of_tensors.append(data_detach)
            elif isinstance(data, (list, tuple)):
                for item in data:
                    list_of_tensors.extend(_extract_tensor_list(item))
            else:
                # Log non-tensor types for debugging but continue execution
                logging.warning(f"Encountered non-tensor type in layer data: {type(data)} with value: {data}")
            return list_of_tensors
        
        _inputs = tuple(_extract_tensor_list(inputs))
        _outputs = tuple(_extract_tensor_list(outputs))

        if self.cfg.hooks.dump_only_summaries:
            # Compute statistical summaries instead of saving raw tensors
            _inputs_stats = dict()
            
            # Calculate L2 norms and synchronize across tensor parallel ranks
            _inputs_norm_squares = tuple(torch.square(i.data.norm()) for i in _inputs)
            for t in _inputs_norm_squares:
                all_reduce(t, op=torch.distributed.ReduceOp.SUM, group=get_tensor_model_parallel_group())
            _inputs_stats['norms'] = tuple(torch.pow(i, 0.5) for i in _inputs_norm_squares) 

            # Calculate means and synchronize across tensor parallel ranks
            size = get_tensor_model_parallel_size()
            _inputs_mean = tuple(torch.mean(i.data) for i in _inputs)
            for t in _inputs_mean:
                all_reduce(t, op=torch.distributed.ReduceOp.SUM, group=get_tensor_model_parallel_group())
            _inputs_stats['means'] = tuple(i/size for i in _inputs_mean) 

            # Calculate max values across tensor parallel ranks
            _inputs_max = tuple(torch.max(i.data) for i in _inputs)
            for t in _inputs_max:
                all_reduce(t, op=torch.distributed.ReduceOp.MAX, group=get_tensor_model_parallel_group())
            _inputs_stats['max'] = _inputs_max

            # Calculate min values across tensor parallel ranks
            _inputs_min = tuple(torch.min(i.data) for i in _inputs)
            for t in _inputs_min:
                all_reduce(t, op=torch.distributed.ReduceOp.MIN, group=get_tensor_model_parallel_group())
            _inputs_stats['min'] = _inputs_min

            # Repeat the same statistical calculations for outputs
            _outputs_stats = dict()
            _outputs_norm_squares = tuple(torch.square(i.data.norm()) for i in _outputs)
            for t in _outputs_norm_squares:
                all_reduce(t, op=torch.distributed.ReduceOp.SUM, group=get_tensor_model_parallel_group())
            _outputs_stats['norms'] = tuple(torch.pow(i, 0.5) for i in _outputs_norm_squares) 
            
            _outputs_mean = tuple(torch.mean(i.data) for i in _outputs)
            for t in _outputs_mean:
                all_reduce(t, op=torch.distributed.ReduceOp.SUM, group=get_tensor_model_parallel_group())
            _outputs_stats['means'] = tuple(i/size for i in _outputs_mean) 

            _outputs_max = tuple(torch.max(i.data) for i in _outputs)
            for t in _outputs_max:
                all_reduce(t, op=torch.distributed.ReduceOp.MAX, group=get_tensor_model_parallel_group())
            _outputs_stats['max'] = _outputs_max

            _outputs_min = tuple(torch.min(i.data) for i in _outputs)
            for t in _outputs_min:
                all_reduce(t, op=torch.distributed.ReduceOp.MIN, group=get_tensor_model_parallel_group())
            _outputs_stats['min'] = _outputs_min

            return _inputs_stats, _outputs_stats
        else:
            # Return raw tensors based on configuration
            if self.cfg.hooks.dump_only_master_rank:
                return _inputs, _outputs
            elif self.cfg.hooks.allgather_activations:
                # Gather tensors from all tensor parallel ranks
                _inputs_allgather = []
                _outputs_allgather = []
                size = get_tensor_model_parallel_size()
                for t in _inputs:
                    t_allgather = xm.all_gather(
                        t,
                        dim=self.cfg.hooks.allgather_dim_for_module_inputs,
                        groups=get_tensor_model_parallel_group(as_list=True),
                        pin_layout=True,
                    )
                    _inputs_allgather.append(t_allgather)
                for t in _outputs:
                    t_allgather = xm.all_gather(
                        t,
                        dim=self.cfg.hooks.allgather_dim_for_module_outputs,
                        groups=get_tensor_model_parallel_group(as_list=True),
                        pin_layout=True,
                    )
                    _outputs_allgather.append(t_allgather)
                return _inputs_allgather, _outputs_allgather
            else:
                return _inputs, _outputs


    def save_activations_map(self) -> None:
        """
        Save the recorded activations (layer input, layer output) for each target layer(s) in the model.
        
        Each activation is saved to a unique directory based on the layer name and global step.
        The activations are stored in .npy format for raw data or .npz format for statistical summaries.
        The function clears the activations map after saving to prevent re-saving the same tensors
        on successive calls.
        
        The saved files include rank information (data parallel, pipeline parallel, tensor parallel)
        in the filename for distributed training scenarios.
        """
        for layer_name, activation_list in self.activations_map.items():
            for activation_data in activation_list:
                global_step, layer_inputs, layer_outputs = activation_data
                # Get distributed training rank information for filename
                data_parallel_rank = get_data_parallel_rank()
                pipeline_parallel_rank = get_pipeline_model_parallel_rank()
                tensor_parallel_rank = get_tensor_model_parallel_rank()
                output_directory = f"{self.hooks_dump_base_directory}/{layer_name}/global_step_{global_step}"
                os.makedirs(output_directory, exist_ok=True)
                dp_pp_tp_config = f"dp_{data_parallel_rank}_pp_{pipeline_parallel_rank}_tp_{tensor_parallel_rank}"

                if not self.cfg.hooks.dump_only_summaries:
                    # Save raw activation tensors
                    for activation_type, activations in (("activation-input", layer_inputs), ("activation-output", layer_outputs)):
                        for i, activation in enumerate(activations):
                            arr = activation.detach().cpu().float().numpy()
                            filename = f"{output_directory}/{dp_pp_tp_config}_{activation_type}_raw_{i}.npy"
                            # Only save from rank 0 if allgather was used, otherwise save from all ranks
                            if not self.cfg.hooks.allgather_activations or tensor_parallel_rank == 0:
                                np.save(filename, arr)
                            # Log histograms to tensorboard if enabled
                            if self.cfg.hooks.plot_hist and xr.global_ordinal() == 0:
                                self.trainer.logger.experiment.add_histogram(
                                    f"{layer_name}-{activation_type}-{i}", arr, global_step
                                )
                elif self.pl_module is not None:
                    # Save statistical summaries instead of raw tensors
                    summary_dict = {}
                    for activation_type, activations in (("activation-input", layer_inputs), ("activation-output", layer_outputs)):
                        for stat_name in ("norms", "means", "max", "min"):
                            for i, value in enumerate(activations[stat_name]):
                                key = f"{layer_name}-{activation_type}-{i}-{stat_name[:-1]}"  # strip plural 's'
                                summary_dict[key] = value.detach().cpu().float().numpy()
                    summary_filename = f"{output_directory}/{dp_pp_tp_config}_summary_dict.npz"
                    if tensor_parallel_rank == 0:
                        np.savez(summary_filename, **summary_dict)
                   

        self.activations_map.clear()



    def create_forward_hook(self, layer_name: str, pl_module: Any) -> callable:
        """
        Create a forward hook function to record the input & output activations of target layer(s) during the forward pass.
        
        The hook function captures layer inputs and outputs at specified intervals and stores them
        in the activations map for later saving. The hook respects configuration settings for
        dumping frequency and rank-specific filtering.
        
        Args:
            layer_name: Name identifier for the target layer
            pl_module: PyTorch Lightning module containing the model and global step information
            
        Returns:
            A hook function that can be registered with a PyTorch module to capture activations
        """
        def hook(module, inputs, outputs):
            # Only capture activations at specified intervals
            if pl_module.global_step % self.cfg.hooks.hooks_interval == 0:
                layer_identifier = f"{layer_name}"
                if layer_identifier not in self.activations_map:
                    self.activations_map[layer_identifier] = []

                processed_inputs, processed_outputs = self.process_inputs_outputs(inputs, outputs)

                # Filter by rank if configured to dump only from master rank
                if (not self.cfg.hooks.dump_only_master_rank) or (self.cfg.hooks.dump_only_master_rank and xr.global_ordinal() == 0):
                    self.activations_map[layer_identifier].append((
                        pl_module.global_step,
                        processed_inputs,
                        processed_outputs
                    ))

                # Schedule tensor saving after XLA step completion
                xm.add_step_closure(self.save_activations_map)
        return hook


    def register_forward_hook_wrapper(self, layer_name: str, layer: torch.nn.Module, pl_module: Any) -> None:
        """
        Register the forward hook function to the target layer.
        
        This method creates and registers a forward hook that will capture activations
        for the given layer during the forward pass of the model.
        
        Args:
            layer_name: Name identifier for the target layer
            layer: The PyTorch module/layer to attach the hook to
            pl_module: PyTorch Lightning module containing the model and global step information
        """
        layer.register_forward_hook(self.create_forward_hook(layer_name, pl_module))


    def save_gradients_map(self) -> None:
        """
        Save the recorded gradients for each target layer(s) in the model.
        
        Each gradient is saved to a unique directory based on the layer name and global step.
        The gradients are stored in .npy format for raw data or .npz format for statistical summaries.
        The function clears the gradients map after saving to prevent re-saving the same tensors
        on successive calls.
        
        The saved files include rank information (data parallel, pipeline parallel, tensor parallel)
        in the filename for distributed training scenarios.
        """
        for layer_name, gradient_list in self.gradients_map.items():
            for gradient_data in gradient_list:
                global_step, grad_inputs, grad_outputs = (
                    gradient_data[0],
                    gradient_data[1],
                    gradient_data[2],
                )
                data_parallel_rank = get_data_parallel_rank()
                pipeline_parallel_rank = get_pipeline_model_parallel_rank()
                tensor_parallel_rank = get_tensor_model_parallel_rank()
                output_directory = f"{self.hooks_dump_base_directory}/{layer_name}/global_step_{global_step}"
                os.makedirs(output_directory, exist_ok=True)
                #tensor_parallel_rank = torch.distributed.get_rank()
                dp_pp_tp_config = f"dp_{data_parallel_rank}_pp_{pipeline_parallel_rank}_tp_{tensor_parallel_rank}"
                if not self.cfg.hooks.dump_only_summaries:
                    for grad_type, grads in (("gradient-input", grad_inputs), ("gradient-output", grad_outputs)):
                        for i, grad in enumerate(grads):
                            arr = grad.detach().cpu().float().numpy()
                            filename = f"{output_directory}/{dp_pp_tp_config}_{grad_type}_raw_{i}.npy"
                            if not self.cfg.hooks.allgather_activations or tensor_parallel_rank == 0:
                                np.save(filename, arr)
                            if self.cfg.hooks.plot_hist and xr.global_ordinal() == 0:
                                self.trainer.logger.experiment.add_histogram(
                                    f'{layer_name}-{grad_type}-{i}', arr, global_step
                                )
                elif self.pl_module is not None:
                    summary_dict = {}
                    for grad_type, grads in (("gradient-input", grad_inputs), ("gradient-output", grad_outputs)):
                        for stat_name in ("norms", "means", "max", "min"):
                            for i, value in enumerate(grads[stat_name]):
                                key = f"{layer_name}-{grad_type}-{i}-{stat_name[:-1]}"  # strip plural 's'
                                summary_dict[key] = value.detach().cpu().float().numpy()
                    if tensor_parallel_rank == 0:
                        np.savez(f"{output_directory}/{dp_pp_tp_config}_summary_dict.npz", **summary_dict)
                
                    
        self.gradients_map.clear()


    def create_backward_hook(self, layer_name: str, pl_module: Any) -> callable:
        """
        Create a backward hook function to record the gradients for target layer(s) during the backward pass.
        
        The hook function captures gradient inputs and outputs at specified intervals and stores them
        in the gradients map for later saving. The hook respects configuration settings for
        dumping frequency and rank-specific filtering.
        
        Args:
            layer_name: Name identifier for the target layer
            pl_module: PyTorch Lightning module containing the model and global step information
            
        Returns:
            A hook function that can be registered with a PyTorch module to capture gradients
        """
        def hook(module, grad_inputs, grad_outputs):
            # Only capture gradients at specified intervals
            if pl_module.global_step % self.cfg.hooks.hooks_interval == 0:
                layer_identifier = f"{layer_name}"
                if layer_identifier not in self.gradients_map:
                    self.gradients_map[layer_identifier] = []

                processed_inputs, processed_outputs = self.process_inputs_outputs(grad_inputs, grad_outputs)

                # Filter by rank if configured to dump only from master rank
                if (not self.cfg.hooks.dump_only_master_rank) or (self.cfg.hooks.dump_only_master_rank and xr.global_ordinal() == 0):
                    self.gradients_map[layer_identifier].append((
                        pl_module.global_step,
                        processed_inputs,
                        processed_outputs
                    ))

                # Schedule tensor saving after XLA step completion
                xm.add_step_closure(self.save_gradients_map)
        return hook


    def register_backward_hook_wrapper(self, layer_name: str, layer: torch.nn.Module, pl_module: Any) -> None:
        """
        Register the backward hook function to the target layer.
        
        This method creates and registers a backward hook that will capture gradients
        for the given layer during the backward pass of the model.
        
        Args:
            layer_name: Name identifier for the target layer
            layer: The PyTorch module/layer to attach the hook to
            pl_module: PyTorch Lightning module containing the model and global step information
        """
        layer.register_full_backward_hook(self.create_backward_hook(layer_name, pl_module))


    def on_train_start(self, trainer: Any, pl_module: Any) -> None:
        """
        PyTorch Lightning callback method called at the start of training.
        
        This method initializes the callback by storing references to the trainer and model,
        printing the model architecture, and registering hooks on the specified target layers
        based on the configuration settings.
        
        Args:
            trainer: PyTorch Lightning trainer instance
            pl_module: PyTorch Lightning module containing the model to be hooked
        """
        self.pl_module = pl_module
        self.trainer = trainer
        print('LOGGER:{}'.format(pl_module.logger))
        
        # Print model architecture for debugging
        for layer_name, layer in pl_module.model.named_modules():
            print(f"{layer_name} | {type(layer)}")

        # Parse target layers from configuration and register hooks
        target_layers = self.cfg.hooks.target_layers.split(",")
        target_layers = [s.strip() for s in target_layers]
        for layer_name, layer in pl_module.model.named_modules():
            layer_name = layer_name.strip()
            if layer_name in target_layers:
                # Register hooks based on configuration flags
                if self.cfg.hooks.enable_activation_dumps:
                    self.register_forward_hook_wrapper(layer_name, layer, pl_module)
                if self.cfg.hooks.enable_grad_dumps:
                    self.register_backward_hook_wrapper(layer_name, layer, pl_module)
