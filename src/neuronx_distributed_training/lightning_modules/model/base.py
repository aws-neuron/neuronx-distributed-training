# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import itertools
import math
import numbers
import os
import time
from typing import Any, Dict, Optional, Union

import neuronx_distributed as nxd
import torch
import torch_xla.core.xla_model as xm
from lightning_utilities.core.apply_func import apply_to_collection
from nemo.collections.nlp.models.nlp_model import NLPModel
from nemo.core.classes.common import PretrainedModelInfo
from nemo.core.optim import get_optimizer, prepare_lr_scheduler
from neuronx_distributed.modules.qkv_linear import GQAQKVColumnParallelLinear
from neuronx_distributed.parallel_layers import mappings, parallel_state
from neuronx_distributed.parallel_layers.layers import (
    ColumnParallelLinear,
    ParallelEmbedding,
    RowParallelLinear,
)
from neuronx_distributed.parallel_layers.utils import (
    param_is_not_tensor_parallel_duplicate,
)
from neuronx_distributed.modules.moe.loss_function import load_balancing_loss_func
from neuronx_distributed.utils.batch_utils import get_batch_on_this_context_parallel_rank
from omegaconf import DictConfig
from pytorch_lightning.trainer.connectors.logger_connector.fx_validator import (
    _FxValidator,
)
from lightning.fabric.utilities.distributed import _sync_ddp
from neuronx_distributed_training.utils import _distributed_available
from pytorch_lightning.utilities.exceptions import MisconfigurationException
from pytorch_lightning.utilities.rank_zero import rank_zero_warn
from pytorch_lightning.utilities.signature_utils import is_param_in_hook_signature
from torch import Tensor
from torchmetrics import Metric
from torch_xla import runtime

from neuronx_distributed_training.models.megatron.module import param_is_not_shared
from neuronx_distributed_training.utils import Throughput
from neuronx_distributed_training.utils import get_attribute_from_cfg

from .megatron_init import initialize_model_parallel_for_nemo

class BaseModelModule(NLPModel):
    def __init__(self, cfg, trainer, no_lm_init=True):
        super().__init__(cfg.model, trainer, no_lm_init=no_lm_init)
        self.config = cfg
        self.trainer = trainer
        dp_size = runtime.world_size() / (
            self.config.distributed_strategy.get("tensor_model_parallel_size") * self.config.distributed_strategy.get("pipeline_model_parallel_size") * get_attribute_from_cfg(self.config, "context_parallel_size", 1)
        )
        self.num_microbatches = int(self.config.data.global_batch_size / (self.config.data.micro_batch_size * dp_size))

        # TODO: remove this when PTL 1.7.3 is released
        _FxValidator.functions["configure_gradient_clipping"] = {
            "allowed_on_step": (False, True),
            "allowed_on_epoch": (False, True),
            "default_on_step": True,
            "default_on_epoch": False,
        }
        self.log_parameter_norm = self.config.exp_manager.get("log_parameter_norm", False)
        self.log_gradient_norm = self.config.exp_manager.get("log_gradient_norm", False)
        self.save_logits = cfg.model.get("save_logits", False)
        self.grad_clip_pl_default = False  # use pytorch default for gradient clipping. Default False

        self._validate_and_override_config()

        # buffer used during train_step for logging average loss over gradient accumulation steps
        self._reduced_loss_buffer = []

        self._initialize_nxd_config()
        self.throughput = Throughput(10)

        initialize_model_parallel_for_nemo(
            world_size=trainer.world_size,
            global_rank=trainer.global_rank,
            local_rank=trainer.local_rank,
            tensor_model_parallel_size=self.config.distributed_strategy.get("tensor_model_parallel_size", 1),
            pipeline_model_parallel_size=self.config.distributed_strategy.get("pipeline_model_parallel_size", 1),
            virtual_pipeline_model_parallel_size=self.config.distributed_strategy.get("virtual_pipeline_model_parallel_size", 1),
            pipeline_model_parallel_split_rank=self.config.distributed_strategy.get("pipeline_model_parallel_split_rank", 0),
            context_parallel_size=get_attribute_from_cfg(self.config, "context_parallel_size", 1),
            micro_batch_size=self.config.data.get("micro_batch_size"),
            global_batch_size=self.config.data.get("global_batch_size"),
            seed=cfg.get("seed", None),
        )

    def setup(self, stage=None):
        """PTL hook that is executed after DDP spawns.
            See https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html#setup for more information.
        Args:
            stage (str, optional): Can be 'fit', 'validate', 'test' or 'predict'. Defaults to None.
        """
        self.model = self.build_model()
        self.init_global_step = self.trainer.global_step
        super().setup(stage)

    def _validate_and_override_config(self):
        """Certain configurations might be incompatible or discouraged.
        We can check for them here and override if necessary.
        """

    def _initialize_nxd_config(self):
        if self.config.distributed_strategy.get("pipeline_model_parallel_size", 1) > 1:
            model_init_config = {
                "sequential_move_factor": self.config.trainer.get("sequential_move_factor", 11),
                "meta_device_init": True,
                "param_init_fn": self.init_weights,
            }
        else:
            model_init_config = None

        precision_type = self.config.precision.get("type")

        if precision_type == "manual":
            use_master_weights = self.config.precision.get("master_weights", False)
            use_fp32_grad_acc = self.config.precision.get("fp32_grad_acc", False)
        else:
            use_master_weights = precision_type == "mixed_precision" or precision_type == "mixed_precisionSR" or precision_type == "autocast"
            use_fp32_grad_acc = precision_type == "mixed_precision" or precision_type == "mixed_precisionSR" or precision_type == "autocast"
        
        zero_one_enabled = self.config.distributed_strategy.get("zero1", True)
        mixed_precision_config = {
            "use_master_weights": use_master_weights and zero_one_enabled,
            "use_fp32_grad_acc": use_fp32_grad_acc and zero_one_enabled,
            "use_master_weights_in_ckpt": self.config.exp_manager.checkpoint_callback_params.get("use_master_weights_in_ckpt", False) and zero_one_enabled,
        }

        self.nxd_config = nxd.neuronx_distributed_config(
            tensor_parallel_size=self.config.distributed_strategy.get("tensor_model_parallel_size", 1),
            pipeline_parallel_size=self.config.distributed_strategy.get("pipeline_model_parallel_size", 1),
            expert_parallel_size=self.config.distributed_strategy.get("expert_model_parallel_size", 1),
            context_parallel_size=get_attribute_from_cfg(self.config, "context_parallel_size", 1),
            optimizer_config={
                "zero_one_enabled": zero_one_enabled,
                "grad_clipping": self.trainer.gradient_clip_val is not None,
                "max_grad_norm": self.trainer.gradient_clip_val,
            },
            sequence_parallel=self.config.distributed_strategy.get("sequence_parallel", False),
            activation_checkpoint_config=None,  # We set None here and let individual models have their own
            pipeline_config={
                "num_microbatches": self.num_microbatches,
                "auto_partition": self.config.model.get("pipeline_cuts", None) is None,
                "param_init_fn": None,
                "autowrap_modules": [mappings],
                "autowrap_functions": [load_balancing_loss_func],
                "use_zero1_optimizer": self.config.distributed_strategy.get("zero1"),
                "use_optimizer_wrapper": True,
                "return_loss_on_cpu": False,
                "virtual_pipeline_size": self.config.distributed_strategy.get("virtual_pipeline_model_parallel_size", 1),
                "pipeline_cuts": self.config.model.get("pipeline_cuts", None),
            },
            model_init_config=model_init_config,
            mixed_precision_config=mixed_precision_config,
            lnc_size=self.trainer.lnc,
        )

    def _get_parameters(self):
        """
        private method to load all the trainable parameters from optimizer param groups
        """
        params = []
        for param_group in self._optimizer_param_groups:
            for param in param_group["params"]:
                params.append(param)
        return params

    def setup_optimizer_param_groups(self):
        raise NotImplementedError

    def on_train_start(self) -> None:
        super().on_train_start()
        self.init_global_step = self.trainer.global_step

    def training_step(self, batch, batch_idx):
        """
        Our dataloaders produce a micro-batch and then we fetch
        a number of microbatches depending on the global batch size and model parallel size
        from the dataloader to produce a list of microbatches.
        Batch should be a list of microbatches and those microbatches should on CPU.
        Microbatches are then moved to GPU during the pipeline.
        The list of microbatches is then piped through the pipeline using Apex fwd/bwd functions.
        """
        full_log = 0 == self.global_step % self.trainer.log_every_n_steps  # always dump at least a partial log

        # we zero grads here because we also call backward in the apex fwd/bwd functions
        self._optimizer.zero_grad()
        
        #batch['input_ids'] has shape [B,Seq_len]
        if 'input_ids' in batch:
            assert batch['input_ids'].shape[1] == self.config.model.get("encoder_seq_length"), f"Mismatch in input data {batch['input_ids'].shape[1]} and sequence length {self.config.model.get('encoder_seq_length')}"

        # Split batch if CP is enabled
        batch = get_batch_on_this_context_parallel_rank(batch)

        with torch.autocast(enabled=self.config.precision.get("type") == "autocast", dtype=torch.bfloat16, device_type="cuda"):
            loss_mean, misc_metrics = self.forward_backward_step(batch, is_training=True)

        with torch.no_grad():
            full_log = 0 == self.global_step % self.trainer.log_every_n_steps  # dump at least a partial log
            lr = self._optimizer.param_groups[0]["lr"]
            consumed_samples = self.trainer.datamodule.compute_consumed_samples(
                self.trainer.global_step + 1 - self.init_global_step
            )

            if self.throughput.seqs_per_iteration is None:
                self.throughput.set_seqs_per_iteration(
                    self.config.data.micro_batch_size, parallel_state.get_data_parallel_size(), self.num_microbatches
                )
            throughput = self.throughput.get_throughput()
            throughput_peak = self.throughput.throughput_peak
            if throughput > throughput_peak:
                self.throughput.throughput_peak = throughput
            self.throughput.throughput_sum += throughput

            param_norm = None
            grad_norm = None
            # only the last stages of the pipeline return losses
            if self.log_parameter_norm:
                param_norm = self.calculate_parameter_norm(self.parameters())
            if self.log_gradient_norm:
                grad_norm = self._optimizer.grad_norm

            ## logging
            # we can only log on one rank if it is rank zero so we broadcast from last rank
            # we can avoid this broadcast by updating the PTL log function to accept specific ranks
            torch.distributed.all_reduce(loss_mean, group=parallel_state.get_pipeline_model_parallel_group())
        if full_log:
            # TDOD : Consider using run_async = True on step closure. Avoiding that not to minimize functional risk
            xm.add_step_closure(
                self.log_metrics,
                (
                    self.log,
                    loss_mean,
                    misc_metrics,
                    lr,
                    float(self.trainer.global_step),
                    float(consumed_samples),
                    grad_norm,
                    param_norm,
                    float(throughput),
                    float(throughput_peak),
                    self.trainer,
                ),
            )
        xm.mark_step()

        # update for new PTL
        return loss_mean

    def on_train_batch_end(self, outputs, batch, batch_idx: int, unused: Optional[int] = 0) -> None:
        super().on_train_batch_end(outputs, batch, batch_idx)

    def get_batch_length(self, batch):
        return self.trainer.datamodule.get_batch_length(batch)

    def validation_step(self, batch, batch_idx):
        """
        Our dataloaders produce a micro-batch and then we fetch
        a number of microbatches depending on the global batch size and model parallel size
        from the dataloader to produce a list of microbatches.
        """
        outputs, length = self.forward_backward_step(batch), self.get_batch_length(batch)
        self.validation_step_outputs.append((outputs, length))
        return outputs, length

    # update
    def on_validation_epoch_end(self):
        # We want to take the average of all the losses reported in the validation epoch
        # outputs is a list of nested tuples containing (loss, misc_metrics)
        losses_across_val_epoch = [x[0][0][0] for x in self.validation_step_outputs]
        averaged_loss_val_epoch = sum(losses_across_val_epoch) / len(losses_across_val_epoch)
        # we can only log on one rank if it is rank zero so we all_reduce from last rank
        # (effectively a broadcast since we are all_reducing with a zero tensor)
        torch.distributed.all_reduce(averaged_loss_val_epoch, group=parallel_state.get_pipeline_model_parallel_group())

        def _log_val_loss(log_fn, loss):
            log_fn("val_loss", loss.cpu(), prog_bar=True, on_step=True, rank_zero_only=True, batch_size=1)

        xm.add_step_closure(
            _log_val_loss,
            (
                self.log,
                averaged_loss_val_epoch.detach(),
            ),
        )

    def setup_optimization(
        self,
        optim_config: Optional[Union[DictConfig, Dict]] = None,
        optim_kwargs: Optional[Dict[str, Any]] = None,
    ):
        optim_kwargs = {} if optim_kwargs is None else optim_kwargs.copy()
        return super().setup_optimization(optim_config=optim_config, optim_kwargs=optim_kwargs)

    def configure_optimizers(self):
        self.setup_optimization()
        self._optimizer = nxd.initialize_parallel_optimizer(
            self.nxd_config,
            get_optimizer(self.config.model.optim["name"]),
            self._optimizer_param_groups,
            **self._optimizer.defaults,
        )
        assert self._trainer.max_steps is not None, "'max_steps' is missing in trainer config."
        if hasattr(self.config.model.optim, "sched"):
            sched_config = dict(self.config.model.optim.sched)
            sched_config["max_steps"] = self._trainer.max_steps
            self._scheduler = prepare_lr_scheduler(
                optimizer=self._optimizer, scheduler_config=sched_config, train_dataloader=self.trainer.datamodule.train_dataloader()
            )
        if self._scheduler is None:
            return self._optimizer
        else:
            return [self._optimizer], [self._scheduler]

    def configure_gradient_clipping(self, *args, **kwargs):
        """LightningModule hook to clip grads.
        We want this to do nothing as grads are clipped by NxD's optimizer wrapper
        """
        return

    def get_num_microbatches(self):
        return self.num_microbatches

    def get_batch_iterator(self, batch):
        """Create a list of microbatches from a list of local minibatches.

        This function creates a list of `k`th microbatches from a list of local minibatches.
        `a local minibatch` consists of `global_batch_size / data_parallel_size` samples.
        """
        from torch_xla.distributed.parallel_loader import MpDeviceLoader

        micro_batch_size = self.config.data.micro_batch_size
        all_batches = []
        for k in range(self.num_microbatches):
            start = k * micro_batch_size
            end = start + micro_batch_size
            microbatch = {}
            for x, y in batch.items():
                size = len(y)
                assert size > start and size >= end, "size issue microbatch"
                microbatch[x] = y[start:end]
            assert len(microbatch) > 0, "Microbatch lenght less than 0"
            all_batches.append(microbatch)
        return MpDeviceLoader(all_batches, xm.xla_device())

    def model_fwd_calc_loss(self, batch):
        output = self.model(**batch)
        return output[0], None

    def forward_backward_step(self, batch, is_training=False):
        # TODO: Needs override
        device = xm.xla_device()
        misc_metrics = None
        running_loss = torch.zeros(1, device=device, dtype=torch.bfloat16)
        if parallel_state.get_pipeline_model_parallel_size() == 1:
            batch_for_pipeline = self.get_batch_iterator(self.process_global_batch(batch))
            total_batches = len(batch_for_pipeline)
            for batch in batch_for_pipeline:
                loss, misc_metrics = self.model_fwd_calc_loss(batch)
                
                # Want to run the loss in fp32 so that the division and sum are not affect by dp degree
                if os.environ.get("XLA_DOWNCAST_BF16", None) == "1":
                    loss = loss.to(torch.float64)
                loss = loss / total_batches
                if is_training:
                    loss.backward()
                running_loss += loss.detach()
        else:
            batch = self.process_global_batch(batch)
            fwd_bwd_fn = self.model.run_train if is_training else self.model.run_eval
            output = fwd_bwd_fn(**batch)
            if parallel_state.get_pipeline_model_parallel_rank() == (
                parallel_state.get_pipeline_model_parallel_size() - 1
            ):
                running_loss = output[0].detach() if self.save_logits else output.detach()
        # model loss can be of dtype of double. Hence last pp stage would have dtype double where as pp=0 to pp_size-2,
        # the running loss would be 0.0 with dtype of float32. Hence, when we do allreduce to broadcast the loss,
        # it can cause mismatch in HLOs. To avoid this, we just typecast the loss to float64
        running_loss = running_loss.to(torch.float64)
        xm.mark_step()
        self.reduce_loss_from_context_parallel_group(running_loss)
        torch.distributed.all_reduce(running_loss, group=parallel_state.get_data_parallel_group())
        loss_mean = running_loss / torch.distributed.get_world_size(group=parallel_state.get_data_parallel_group())
        return loss_mean, misc_metrics
    
    def reduce_loss_from_context_parallel_group(self, running_loss):
        if parallel_state.get_context_model_parallel_size() > 1:
            torch.distributed.all_reduce(running_loss, group=parallel_state.get_context_model_parallel_group())
            running_loss /= parallel_state.get_context_model_parallel_size()

    def calculate_parameter_norm(self, parameters, norm_type=2):
        """Calculate parameter norms across model parallel ranks
        Arguments:
            parameters (Iterable[Tensor] or Tensor): an iterable of Tensors or a
                single Tensor
            norm_type (float or int): type of the used p-norm. Can be ``'math.inf'`` for
                infinity norm.
            Total norm of the parameters (viewed as a single vector).
        """
        if isinstance(parameters, torch.Tensor):
            parameters = [parameters]
        # Norm parameters.
        norm_type = float(norm_type)
        total_norm = torch.tensor([float(0.0)], device=xm.xla_device())
        params_to_norm = []

        # Filter parameters based on:
        #   - parameter should not be shared
        #   - should not be a replica due to tensor model parallelism
        for param in parameters:
            is_not_shared = param_is_not_shared(param)
            is_not_tp_duplicate = param_is_not_tensor_parallel_duplicate(param)
            if is_not_shared and is_not_tp_duplicate:
                params_to_norm.append(param)

        # Calculate norm.
        if norm_type == math.inf:
            total_norm = max(torch.abs(param) for param in params_to_norm)
            total_norm = torch.tensor([float(total_norm)], device=xm.xla_device())
            # Take max across all model-parallel TPUs.
            torch.distributed.all_reduce(
                total_norm, op=torch.distributed.ReduceOp.MAX, group=parallel_state.get_tensor_model_parallel_group()
            )
            torch.distributed.all_reduce(
                total_norm, op=torch.distributed.ReduceOp.MAX, group=parallel_state.get_context_model_parallel_group()
            )
            torch.distributed.all_reduce(
                total_norm, op=torch.distributed.ReduceOp.MAX, group=parallel_state.get_pipeline_model_parallel_group()
            )
            total_norm = total_norm[0]
        else:
            for param in params_to_norm:
                param_norm = torch.norm(param, norm_type)
                total_norm += param_norm**norm_type
            # Sum across all model-parallel TPUs.
            torch.distributed.all_reduce(
                total_norm, op=torch.distributed.ReduceOp.SUM, group=parallel_state.get_tensor_model_parallel_group()
            )
            torch.distributed.all_reduce(
                total_norm, op=torch.distributed.ReduceOp.SUM, group=parallel_state.get_context_model_parallel_group()
            )
            torch.distributed.all_reduce(
                total_norm, op=torch.distributed.ReduceOp.SUM, group=parallel_state.get_pipeline_model_parallel_group()
            )
            total_norm = torch.pow(total_norm, 1.0 / norm_type)
        return total_norm

    def backward(self, *args, **kwargs):
        """LightningModule hook to do backward.
        We want this to do nothing since we run backward training_step
        No need to call it here.
        """
        return

    def optimizer_zero_grad(self, *args, **kwargs):
        """LightningModule hook to zero grad.
        We want this to do nothing as we are zeroing grads during the training_step.
        """
        return

    def _to_tensor(self, value, name: str):
        value = value.clone().detach() if isinstance(value, torch.Tensor) else torch.tensor(value)
        if not torch.numel(value) == 1:
            raise ValueError(
                f"`self.log({name}, {value})` was called, but the tensor must have a single element."
                f" You can try doing `self.log({name}, {value}.mean())`"
            )
        value = value.squeeze()
        return value

    def log(
        self,
        name: str,
        value,
        prog_bar: bool = False,
        logger: bool = True,
        on_step: Optional[bool] = None,
        on_epoch: Optional[bool] = None,
        reduce_fx="mean",
        enable_graph: bool = False,
        sync_dist: bool = False,
        sync_dist_group: Optional[Any] = None,
        add_dataloader_idx: bool = True,
        batch_size: Optional[int] = None,
        metric_attribute: Optional[str] = None,
        rank_zero_only: bool = False,
    ) -> None:
        """Log a key, value pair.

        Example::

            self.log('train_loss', loss)

        The default behavior per hook is documented here: :ref:`extensions/logging:Automatic Logging`.

        Args:
            name: key to log.
            value: value to log. Can be a ``float``, ``Tensor``, ``Metric``, or a dictionary of the former.
            prog_bar: if ``True`` logs to the progress bar.
            logger: if ``True`` logs to the logger.
            on_step: if ``True`` logs at this step. The default value is determined by the hook.
                See :ref:`extensions/logging:Automatic Logging` for details.
            on_epoch: if ``True`` logs epoch accumulated metrics. The default value is determined by the hook.
                See :ref:`extensions/logging:Automatic Logging` for details.
            reduce_fx: reduction function over step values for end of epoch. :meth:`torch.mean` by default.
            enable_graph: if ``True``, will not auto detach the graph.
            sync_dist: if ``True``, reduces the metric across devices. Use with care as this may lead to a significant
                communication overhead.
            sync_dist_group: the DDP group to sync across.
            add_dataloader_idx: if ``True``, appends the index of the current dataloader to
                the name (when using multiple dataloaders). If False, user needs to give unique names for
                each dataloader to not mix the values.
            batch_size: Current batch_size. This will be directly inferred from the loaded batch,
                but for some data structures you might need to explicitly provide it.
            metric_attribute: To restore the metric state, Lightning requires the reference of the
                :class:`torchmetrics.Metric` in your model. This is found automatically if it is a model attribute.
            rank_zero_only: Whether the value will be logged only on rank 0. This will prevent synchronization which
                would produce a deadlock as not all processes would perform this log call.
        """
        # check for invalid values
        apply_to_collection(value, dict, self._LightningModule__check_not_nested, name)
        apply_to_collection(
            value,
            object,
            self._LightningModule__check_allowed,
            name,
            value,
            wrong_dtype=(numbers.Number, Metric, Tensor, dict),
        )

        if self._trainer is None:
            # not an error to support testing the `*_step` methods without a `Trainer` reference
            rank_zero_warn(
                "You are trying to `self.log()` but the `self.trainer` reference is not registered on the model yet."
                " This is most likely because the model hasn't been passed to the `Trainer`"
            )
            return
        results = self.trainer._results
        if results is None:
            raise MisconfigurationException(
                "You are trying to `self.log()` but the loop's result collection is not registered"
                " yet. This is most likely because you are trying to log in a `predict` hook,"
                " but it doesn't support logging"
            )
        if self._current_fx_name is None:
            raise MisconfigurationException(
                "You are trying to `self.log()` but it is not managed by the `Trainer` control flow"
            )

        on_step, on_epoch = _FxValidator.check_logging_and_get_default_levels(
            self._current_fx_name, on_step=on_step, on_epoch=on_epoch
        )

        # make sure user doesn't introduce logic for multi-dataloaders
        if "/dataloader_idx_" in name:
            raise MisconfigurationException(
                f"You called `self.log` with the key `{name}`"
                " but it should not contain information about `dataloader_idx`"
            )

        value = apply_to_collection(value, (torch.Tensor, numbers.Number), self._to_tensor, name)

        if self.trainer._logger_connector.should_reset_tensors(self._current_fx_name):
            # if we started a new epoch (running its first batch) the hook name has changed
            # reset any tensors for the new hook name
            results.reset(metrics=False, fx=self._current_fx_name)

        if metric_attribute is None and isinstance(value, Metric):
            if self._metric_attributes is None:
                # compute once
                self._metric_attributes = {
                    id(module): name for name, module in self.named_modules() if isinstance(module, Metric)
                }
                if not self._metric_attributes:
                    raise MisconfigurationException(
                        "Could not find the `LightningModule` attribute for the `torchmetrics.Metric` logged."
                        " You can fix this by setting an attribute for the metric in your `LightningModule`."
                    )
            # try to find the passed metric in the LightningModule
            metric_attribute = self._metric_attributes.get(id(value), None)
            if metric_attribute is None:
                raise MisconfigurationException(
                    "Could not find the `LightningModule` attribute for the `torchmetrics.Metric` logged."
                    f" You can fix this by calling `self.log({name}, ..., metric_attribute=name)` where `name` is one"
                    f" of {list(self._metric_attributes.values())}"
                )

        if (
            self.trainer.training
            and is_param_in_hook_signature(self.training_step, "dataloader_iter", explicit=True)
            and batch_size is None
        ):
            raise MisconfigurationException(
                "With `def training_step(self, dataloader_iter)`, `self.log(..., batch_size=...)` should be provided."
            )

        results.log(
            self._current_fx_name,
            name,
            value,
            prog_bar=prog_bar,
            logger=logger,
            on_step=on_step,
            on_epoch=on_epoch,
            reduce_fx=reduce_fx,  # type: ignore[arg-type]
            enable_graph=enable_graph,
            add_dataloader_idx=add_dataloader_idx,
            batch_size=batch_size,
            sync_dist=sync_dist and _distributed_available(),
            sync_dist_fn=self.trainer.strategy.reduce or _sync_ddp,
            sync_dist_group=sync_dist_group,
            metric_attribute=metric_attribute,
            rank_zero_only=rank_zero_only,
        )

        self.trainer._logger_connector._current_fx = self._current_fx_name

    def log_metrics(
        self,
        log_fn,
        loss_mean,
        misc_metrics,
        lr,
        global_step,
        consumed_samples,
        grad_norm,
        param_norm,
        throughput,
        throughput_peak,
        trainer,
    ):
        loss_cpu = loss_mean.detach().cpu()
        # trainer.fit_loop.epoch_loop.batch_loop.running_loss.append(loss_cpu)
        log_fn("reduced_train_loss", loss_cpu, prog_bar=True, rank_zero_only=True)
        if misc_metrics is not None:
            for key, value in misc_metrics.items():
                if isinstance(value, torch.Tensor):
                    value = value.detach().cpu()
                log_fn(key, value, prog_bar=True, rank_zero_only=True)
        log_fn("lr", lr, prog_bar=True, rank_zero_only=True)
        if param_norm:
            log_fn("parameter_norm", param_norm.detach().cpu(), prog_bar=True, rank_zero_only=True)
        if grad_norm:
            log_fn("gradient_norm", grad_norm.detach().cpu(), prog_bar=True, rank_zero_only=True)
        log_fn("global_step", global_step, prog_bar=True, rank_zero_only=True)
        log_fn("consumed_samples", consumed_samples, prog_bar=True, rank_zero_only=True)
        log_fn("throughput", throughput, prog_bar=True, rank_zero_only=True)
        log_fn("throughput_peak", throughput_peak, prog_bar=True, rank_zero_only=True)

    def setup_training_data(self, cfg):
        # We do a pass, since we are setting this up as part of data_module
        pass

    def setup_validation_data(self, cfg):
        # We do a pass, since we are setting this up as part of data_module
        pass

    def is_data_parallel_rank_zero(self):
        return parallel_state.get_data_parallel_rank() == 0

    def parameters(self):
        if isinstance(self.model, list):
            return itertools.chain.from_iterable(module.parameters() for module in self.model)
        else:
            return self.model.parameters()

    def load_state_dict(self, state_dict, strict=False):
        load_result = self.model.load_state_dict(state_dict, strict=strict)
        return load_result

    def state_dict(self, *args, **kwargs):
        return self.model.state_dict(*args, **kwargs)

    def train_dataloader(self):
        return self.trainer.datamodule.train_dataloader()

    def test_step(self, batch, batch_idx):
        return self.validation_step(batch, batch_idx)

    def test_epoch_end(self, outputs):
        return self.validation_epoch_end(outputs)

    @classmethod
    def list_available_models(cls) -> Optional[PretrainedModelInfo]:
        return []

    def init_weights(self, module, device):
        """
        Re-init weights after partition
        Referred from HF transformers https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/modeling_llama.py#L690
        """
        if hasattr(module, "init_weight_cpu"):
            module.init_weight_cpu()
            if hasattr(module, "bias") and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, GQAQKVColumnParallelLinear):
            module.initialize_weight_biases()
        elif isinstance(module, torch.nn.Linear):
            module.reset_parameters()
        elif len(module._parameters):
            # If there is no init provided for any of the module, we want to raise
            # an exception to alert the user that there might be some weights that 
            # are not initialized after meta-device init.
            raise NotImplementedError(
                f"{module._get_name()} is not initialized. Please provide an init method as part of init_weights API"
            )
