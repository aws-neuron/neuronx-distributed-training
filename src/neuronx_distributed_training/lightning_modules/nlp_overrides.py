# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import itertools
import os
import shutil
import tempfile
from collections import defaultdict
from datetime import timedelta
from functools import partial
from pathlib import Path
from typing import (
    Any,
    Callable,
    Dict,
    Generator,
    Iterable,
    List,
    Mapping,
    Optional,
    Union,
)

import neuronx_distributed as nxd
import lightning.pytorch as pl
import torch
import torch.multiprocessing as mp
import torch_xla.core.xla_model as xm
from lightning.fabric.plugins.environments import ClusterEnvironment
from lightning.pytorch.plugins.io import XLACheckpointIO
from lightning.fabric.plugins.environments import XLAEnvironment
from lightning.fabric.strategies.launchers.xla import _rank_teardown
from lightning.fabric.utilities.types import _PATH
from lightning.pytorch.core.optimizer import Optimizable
from lightning_utilities.core.apply_func import (
    apply_to_collection,
    apply_to_collections,
)
from lightning_utilities.core.imports import RequirementCache
from nemo.collections.nlp.modules.common.megatron.module import Float16Module
from nemo.utils import AppState, logging

# This file is copied as it is from NNM, needs some cleanup.
from neuronx_distributed.parallel_layers import parallel_state
from omegaconf import OmegaConf
from pytorch_lightning.accelerators import Accelerator
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.loggers import Logger
from lightning.pytorch.loops import (
    _TrainingEpochLoop as TrainingEpochLoop,
    _PredictionLoop as PredictionLoop,
    _EvaluationLoop as EvaluationLoop,
    _FitLoop as FitLoop,
)
from lightning.pytorch.loops.utilities import _select_data_fetcher
from lightning.fabric.utilities.data import (
    _auto_add_worker_init_fn,
    has_iterable_dataset,
    _set_sampler_epoch,
    sized_len
)
from lightning.fabric.plugins import CheckpointIO
from lightning.pytorch.plugins import _PLUGIN_INPUT as PLUGIN_INPUT
from lightning.pytorch.plugins.precision import Precision as PrecisionPlugin
from lightning.pytorch.plugins.precision import XLAPrecision as XLAPrecisionPlugin
from pytorch_lightning.profilers import Profiler
from lightning.pytorch.strategies import Strategy, XLAStrategy
from pytorch_lightning.strategies.launchers.xla import _XLALauncher
from pytorch_lightning.trainer import call, setup
from lightning.pytorch.trainer.connectors.accelerator_connector import (
    _LITERAL_WARN,
    _PRECISION_INPUT,
    _AcceleratorConnector as AcceleratorConnector,
)
from lightning.pytorch.trainer.connectors.callback_connector import (
    _CallbackConnector as CallbackConnector,
)
from lightning.pytorch.trainer.connectors.checkpoint_connector import (
    _CheckpointConnector as CheckpointConnector,
)
from lightning.pytorch.trainer.connectors.data_connector import ( 
    _DataConnector as DataConnector,
    _request_dataloader,
    _resolve_overfit_batches,
    _check_dataloader_iterable,
    _parse_num_batches,
    _worker_check,
)
from lightning.pytorch.trainer.connectors.logger_connector import (
    _LoggerConnector as LoggerConnector,
)
from lightning.pytorch.trainer.connectors.logger_connector.result import (
    _ResultCollection,
    _ResultMetric,
)
from pytorch_lightning.trainer.connectors.signal_connector import (
    _SignalConnector as SignalConnector
)
from lightning.pytorch.utilities import CombinedLoader
from lightning.pytorch.trainer.states import RunningStage, TrainerFn, TrainerState, TrainerStatus

from lightning.pytorch.trainer.trainer import Trainer
from pytorch_lightning.utilities.argparse import _defaults_from_env_vars
from pytorch_lightning.utilities.exceptions import MisconfigurationException
from pytorch_lightning.utilities.rank_zero import rank_zero_warn
from pytorch_lightning.utilities.warnings import PossibleUserWarning
from lightning.pytorch.utilities.model_helpers import is_overridden
from torch import Tensor
from torch.utils.data import BatchSampler, DataLoader, IterableDataset, RandomSampler, Sampler, SequentialSampler
from torch_xla.distributed.zero_redundancy_optimizer import ZeroRedundancyOptimizer
from torchmetrics import Metric
from neuronx_distributed_training.utils import get_lnc_size, get_attribute_from_cfg


def _is_dataloader_shuffled_patched(dataloader: object) -> bool:
    '''
    This function patches _is_data_loader_shuffled from PTL 2.5
    '''
    if hasattr(dataloader, "__pl_saved_kwargs"):
        # this attribute is not part of PyTorch's DataLoader, but could have been set by
        # our `_replace_init_method` context manager
        if "shuffle" in dataloader.__pl_saved_kwargs:
            return dataloader.__pl_saved_kwargs["shuffle"]
        if "shuffle" in dataloader.__pl_saved_arg_names:
            return dataloader.__pl_saved_args[dataloader.__pl_saved_arg_names.index("shuffle")]
    if hasattr(dataloader, "dataset") and isinstance(dataloader.dataset, IterableDataset):
        # shuffling is useless with iterable datasets
        return False
    if not hasattr(dataloader, "sampler"):
        # shuffling is enabled via a sampler. No sampler, no shuffling
        return False
    sampler = dataloader.batch_sampler
    # sampler = batch_sampler if batch_sampler is not None else dataloader.sampler
    if isinstance(sampler, SequentialSampler):
        return False
    return isinstance(sampler, RandomSampler)


def has_len_all_ranks_patched(
    dataloader,
    strategy,
    model,
) -> bool:
    """Checks if a given Dataloader has ``__len__`` method implemented i.e. if it is a finite dataloader or
    infinite dataloader."""
    try:
        local_length = len(dataloader)  # type: ignore [arg-type] # we are checking with duck-typing
        total_length, = strategy.reduce(torch.tensor([local_length], device=strategy.root_device), reduce_op="sum")

        if total_length == 0:
            rank_zero_warn(
                f"Total length of `{dataloader.__class__.__name__}` across ranks is zero."
                " Please make sure this was your intention."
            )
        if total_length > 0 and local_length == 0:
            if model.allow_zero_length_dataloader_with_multiple_devices:
                rank_zero_warn(
                    f"Total length of `{dataloader.__class__.__name__}` across ranks is zero, but local rank has zero"
                    " length. Please be cautious of uneven batch length."
                )
                has_len = False
            else:
                raise MisconfigurationException(
                    f"`{dataloader.__class__.__name__}` within local rank has zero length."
                    " Please make sure that it returns at least 1 batch."
                )
        else:
            has_len = True

    except (TypeError, NotImplementedError):
        has_len = False

    # we are checking using lightning_lite, which doesn't know CombinedLoader
    if has_len and has_iterable_dataset(dataloader):  # type: ignore [arg-type]
        rank_zero_warn(
            "Your `IterableDataset` has `__len__` defined."
            " In combination with multi-process data loading (when num_workers > 1),"
            " `__len__` could be inaccurate if each worker is not configured independently"
            " to avoid having duplicate data."
        )
    if os.environ.get("NEURON_EXTRACT_GRAPHS_ONLY", None):
        return True
    return has_len


class TRNPrecisionPlugin(XLAPrecisionPlugin):
    """Precision plugin for TPU integration."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)

    def optimizer_step(  # type: ignore[override]
        self,
        optimizer: Optimizable,
        model: "pl.LightningModule",
        closure: Callable[[], Any],
        **kwargs: Any,
    ) -> Any:
        """Hook to run the optimizer step."""

        if not isinstance(optimizer, ZeroRedundancyOptimizer):
            #closure = partial(self._xla_wrap_closure, optimizer, closure)
            closure = partial(self._wrap_closure, model, optimizer, closure)

        return optimizer.step(closure=closure, **kwargs)


class _NLPXLALauncher(_XLALauncher):
    def launch(self, function: Callable, *args: Any, trainer=None, **kwargs: Any) -> Any:
        """Launches processes that run the given function in parallel.

        The function is allowed to have a return value. However, when all processes join, only the return value
        of worker process 0 gets returned from this `launch` method in the main process.

        Arguments:
            function: The entry point for all launched processes.
            *args: Optional positional arguments to be passed to the given function.
            trainer: Optional reference to the :class:`~pytorch_lightning.trainer.trainer.Trainer` for which
                a selected set of attributes get restored in the main process after processes join.
            **rning AMI GPU PyTorch 1.13.1 (Ubuntu 20.04) 20230519
            kwargs: Optional keyword arguments to be passed to the given function.
        """

        if not self._strategy.cluster_environment.creates_processes_externally:
            context = mp.get_context(self._start_method)
            return_queue = context.SimpleQueue()
            import torch_xla.distributed.xla_multiprocessing as xmp

            xmp.spawn(
                self._wrapping_function,
                args=(trainer, function, args, kwargs, return_queue),
                nprocs=self._strategy.num_processes,
                start_method=self._start_method,
            )
        else:
            process_idx = int(os.environ.get("LOCAL_RANK"))
            self._strategy._local_rank = process_idx
            _ = function(*args, **kwargs)
            _rank_teardown(process_idx)

        return None

    def _wrapping_function(
        self, process_idx: int, trainer, function, args, kwargs, return_queue, global_states=None
    ) -> None:
        self._strategy._local_rank = process_idx
        _ = function(*args, **kwargs)

        #### NEURON: Avoiding moving data from device to CPU
        _rank_teardown(process_idx)


class _NLPResultCollection(_ResultCollection):
    def register_key(self, key: str, meta, value) -> None:
        """Create one _ResultMetric object per value.

        Value can be provided as a nested collection
        """

        def fn(v):
            metric = _ResultMetric(meta, isinstance(v, Tensor))
            ### NEURON: Do not move metrics to device, results in unnnecessary compiles
            return metric

        self.update(apply_to_collection(value, (Tensor, Metric), fn))

    def update_metrics(self, key: str, value, batch_size: int) -> None:
        def fn(result_metric, v):
            # performance: avoid calling `__call__` to avoid the checks in `torch.nn.Module._call_impl`
            ### NEURON: Do not move metrics to device, results in unnnecessary compiles
            result_metric.forward(v, batch_size)
            result_metric.has_reset = False

        apply_to_collections(self[key], value, _ResultMetric, fn)


class NLPEvaluationLoop(EvaluationLoop):
    def __init__(
        self,
        trainer: "pl.Trainer",
        trainer_fn: TrainerFn,
        stage: RunningStage,
        verbose: bool = True,
        inference_mode: bool = True,
    ) -> None:
        # We override this class to make strainer, trainer_fn, stage, verbose: bool = True) -> None:
        super().__init__(trainer, trainer_fn, stage, verbose, inference_mode)
        self._results = _NLPResultCollection(training=False)

    def teardown(self) -> None:
        if self._data_fetcher is not None:
            self._data_fetcher.teardown()
            self._data_fetcher = None

    def _on_evaluation_start(self, *args: Any, **kwargs: Any) -> None:
        """Runs ``on_{validation/test}_start`` hooks."""
        assert self._results is not None

        hook_name = "on_test_start" if self.trainer.testing else "on_validation_start"
        call._call_callback_hooks(self.trainer, hook_name, *args, **kwargs)
        call._call_lightning_module_hook(self.trainer, hook_name, *args, **kwargs)
        call._call_strategy_hook(self.trainer, hook_name, *args, **kwargs)

    def setup_data(self) -> None:
        '''
        This function is overrides setup_data() from PTL 2.5
        '''
        trainer = self.trainer
        trainer_fn = self._trainer_fn
        if self._combined_loader is not None and trainer_fn == TrainerFn.FITTING and not self._should_reload_val_dl:
            return

        pl_module = trainer.lightning_module
        limit_batches = trainer.limit_test_batches if trainer.testing else trainer.limit_val_batches
        hook_name = "test_step" if trainer.testing else "validation_step"
        if limit_batches == 0 or not is_overridden(hook_name, pl_module):
            return

        # store epoch of dataloader reset for reload_dataloaders_every_n_epochs
        # it should not reload again if it has already reloaded during sanity_check
        if trainer_fn == TrainerFn.FITTING and (
            (trainer.sanity_checking and trainer.fit_loop.epoch_loop._should_check_val_epoch())
            or not trainer.sanity_checking
        ):
            self._last_val_dl_reload_epoch = trainer.current_epoch

        stage = self._stage
        source = self._data_source
        dataloaders = _request_dataloader(source)
        trainer.strategy.barrier(f"{stage.dataloader_prefix}_dataloader()")

        if not isinstance(dataloaders, CombinedLoader):
            combined_loader = CombinedLoader(dataloaders, "sequential")
        else:
            combined_loader = dataloaders

        if trainer_fn == TrainerFn.FITTING and trainer.overfit_batches > 0:
            _resolve_overfit_batches(combined_loader, stage)

        dataloaders = []
        for dl in combined_loader.flattened:
            _check_dataloader_iterable(dl, source, trainer_fn)
            dl = NLPDataConnector._process_dataloader(trainer, trainer_fn, stage, dl)
            dataloaders.append(dl)
        combined_loader.flattened = dataloaders
        self._combined_loader = combined_loader

        allow_zero_length = pl_module.allow_zero_length_dataloader_with_multiple_devices
        if trainer.datamodule is not None:
            allow_zero_length |= trainer.datamodule.allow_zero_length_dataloader_with_multiple_devices

        self._max_batches = []
        for dl in combined_loader.flattened:
            # determine number of batches
            length = len(dl) if has_len_all_ranks_patched(dl, trainer.strategy, pl_module) else float("inf")
            limit_batches = getattr(trainer, f"limit_{stage.dataloader_prefix}_batches")
            num_batches = _parse_num_batches(stage, length, limit_batches)
            self._max_batches.append(num_batches)

        # this depends on the data used, so reset it too
        self._seen_batches_per_dataloader = defaultdict(int)


class NLPTrainingEpochLoop(TrainingEpochLoop):
    def __init__(self, trainer: "pl.Trainer", min_steps: Optional[int] = None, max_steps: int = -1) -> None:
        super().__init__(trainer, min_steps, max_steps)
        self.val_loop = NLPEvaluationLoop(
            trainer, TrainerFn.FITTING, RunningStage.VALIDATING, verbose=False, inference_mode=False
        )
        self._results = _NLPResultCollection(training=True)


class NLPPredictionLoop(PredictionLoop):
    # We override setup data to ensure has_len_all_ranks is using the patched version
    def setup_data(self) -> None:
        trainer = self.trainer
        # a default `predict_step` exists in the LightningModule, so no need to check if it's overridden
        if trainer.limit_predict_batches == 0:
            return

        source = self._data_source
        dataloaders = _request_dataloader(source)
        trainer.strategy.barrier("predict_dataloader()")

        if not isinstance(dataloaders, CombinedLoader):
            combined_loader = CombinedLoader(dataloaders, "sequential")
        else:
            combined_loader = dataloaders

        allow_zero_length = trainer.lightning_module.allow_zero_length_dataloader_with_multiple_devices
        if trainer.datamodule is not None:
            allow_zero_length |= trainer.datamodule.allow_zero_length_dataloader_with_multiple_devices

        trainer_fn = TrainerFn.PREDICTING
        stage = RunningStage.PREDICTING
        dataloaders = []
        self.max_batches = []
        for dl in combined_loader.flattened:
            _check_dataloader_iterable(dl, source, trainer_fn)
            dl = NLPDataConnector._process_dataloader(trainer, trainer_fn, stage, dl)
            dataloaders.append(dl)

            # determine number of batches
            length = len(dl) if has_len_all_ranks_patched(dl, trainer.strategy, allow_zero_length) else float("inf")
            num_batches = _parse_num_batches(stage, length, trainer.limit_predict_batches)
            self.max_batches.append(num_batches)
        combined_loader.flattened = dataloaders
        self._combined_loader = combined_loader


class NLPFitLoop(FitLoop):
    # We override setup data to ensure has_len_all_ranks is using the patched version
    def setup_data(self, updated_data_source=None) -> None:
        if self._combined_loader is not None and not self._should_reload_train_dl and updated_data_source is None:
            return

        trainer = self.trainer
        pl_module = trainer.lightning_module
        if trainer.limit_train_batches == 0 or not is_overridden("training_step", pl_module):
            return

        logging.debug(f"{self.__class__.__name__}: resetting train dataloader")

        source = updated_data_source if updated_data_source is not None else self._data_source
        train_dataloader = _request_dataloader(source)
        trainer.strategy.barrier("train_dataloader()")

        if not isinstance(train_dataloader, CombinedLoader):
            combined_loader = CombinedLoader(train_dataloader, "max_size_cycle")
        else:
            combined_loader = train_dataloader

        if trainer.overfit_batches > 0:
            _resolve_overfit_batches(combined_loader, mode=RunningStage.TRAINING)

        trainer_fn = TrainerFn.FITTING
        stage = RunningStage.TRAINING
        dataloaders = []
        for dl in combined_loader.flattened:
            _check_dataloader_iterable(dl, source, trainer_fn)
            dl = NLPDataConnector._process_dataloader(trainer, trainer_fn, stage, dl)
            dataloaders.append(dl)
        combined_loader.flattened = dataloaders
        self._combined_loader = combined_loader

        allow_zero_length = pl_module.allow_zero_length_dataloader_with_multiple_devices
        if trainer.datamodule is not None:
            allow_zero_length |= trainer.datamodule.allow_zero_length_dataloader_with_multiple_devices

        limits = []
        for dl in combined_loader.flattened:
            # determine number of batches
            length = len(dl) if has_len_all_ranks_patched(dl, trainer.strategy, allow_zero_length) else float("inf")
            num_batches = _parse_num_batches(stage, length, trainer.limit_train_batches)
            limits.append(num_batches)

        combined_loader.limits = limits

        self._load_combined_loader_states()

        self._data_fetcher = _select_data_fetcher(trainer, RunningStage.TRAINING)
        self._data_fetcher.setup(combined_loader)
        iter(self._data_fetcher)  # creates the iterator inside the fetcher
        max_batches = sized_len(combined_loader)
        self.max_batches = max_batches if max_batches is not None else float("inf")
        has_len_all_ranks_ = has_len_all_ranks_patched(combined_loader, trainer.strategy, allow_zero_length)

        if self.max_batches == 0:
            return

        # store epoch of dataloader reset for reload_dataloaders_every_n_epochs
        self._last_train_dl_reload_epoch = trainer.current_epoch

        if isinstance(trainer.val_check_interval, int):
            trainer.val_check_batch = trainer.val_check_interval
            if trainer.val_check_batch > self.max_batches and trainer.check_val_every_n_epoch is not None:
                raise ValueError(
                    f" `val_check_interval` ({trainer.val_check_interval}) must be less than or equal"
                    f" to the number of the training batches ({self.max_batches})."
                    " If you want to disable validation set `limit_val_batches` to 0.0 instead."
                    " If you want to validate based on the total training batches, set `check_val_every_n_epoch=None`."
                )
        else:
            if not has_len_all_ranks_:
                if trainer.val_check_interval == 1.0:
                    trainer.val_check_batch = float("inf")
                else:
                    raise MisconfigurationException(
                        "When using an IterableDataset for `train_dataloader`,"
                        " `Trainer(val_check_interval)` must be `1.0` or an int. An int k specifies"
                        " checking validation every k training batches."
                    )
            else:
                trainer.val_check_batch = int(self.max_batches * trainer.val_check_interval)
                trainer.val_check_batch = max(1, trainer.val_check_batch)

        if trainer.loggers and self.max_batches < trainer.log_every_n_steps and not trainer.fast_dev_run:
            rank_zero_warn(
                f"The number of training batches ({self.max_batches}) is smaller than the logging interval"
                f" Trainer(log_every_n_steps={trainer.log_every_n_steps}). Set a lower value for log_every_n_steps if"
                " you want to see logs for the training epoch.",
                category=PossibleUserWarning,
            )

    # We override this class to make sure results are on CPU on run start
    def on_run_start(self) -> None:
        """Calls the ``on_train_start`` hook."""
        # update the current_epoch in-case of checkpoint reload
        if not self._iteration_based_training():
            self.epoch_progress.current.completed = self.epoch_progress.current.processed

        if self.epoch_loop._should_check_val_epoch() and self.trainer.val_dataloaders is None:
            self.trainer.validating = True
            self.epoch_loop.val_loop.setup_data()
            self.trainer.training = True

        self._results.cpu()

        call._call_callback_hooks(self.trainer, "on_train_start")
        call._call_lightning_module_hook(self.trainer, "on_train_start")
        call._call_strategy_hook(self.trainer, "on_train_start")


class NLPCheckpointIO(XLACheckpointIO):
    """
    This class overrides PTL's XLACheckpointIO in order to use NxD's
    checkpoint APIs. For more information on XLAChekpointIO please see:
    https://lightning.ai/docs/pytorch/2.5.0/api/lightning.pytorch.plugins.io.XLACheckpointIO.html
    """
    def __init__(self, async_save=False, weight_init_only=False, ptl_version: str = "1.8.6"):
        super().__init__()
        self._async_save = async_save
        self._weight_init_only = weight_init_only
        self.ptl_version = ptl_version

    def load_checkpoint(self, checkpoint_path: _PATH, load_type_xser: bool) -> Dict[str, Any]:
        """PTL override to accomodate model parallel checkpoints"""

        model_state_dict = {}
        optimizer_state_dict = {}
        checkpoint_dir = os.path.dirname(checkpoint_path)
        checkpoint_tag = os.path.basename(checkpoint_path)
        checkpoint = nxd.load_checkpoint(
            checkpoint_dir, checkpoint_tag, model=model_state_dict, optimizer=None if self._weight_init_only else optimizer_state_dict
        )
        
        if checkpoint is None: # usercontent.pt non existing then create an empty dict
            checkpoint = {}
        
        checkpoint["state_dict"] = model_state_dict # model shards should always exist
        if not checkpoint.get("pytorch-lightning_version", None):
            checkpoint["pytorch-lightning_version"] = self.ptl_version

        if not self._weight_init_only: # if loading weights only is not True then load optim states also along with model states.
            checkpoint["optimizer_states"] = [optimizer_state_dict]
        else:        
            logging.info("Weight Init only case, so not loading optim states, optim states would be initialized")
            
        return checkpoint

    def _remove_version_count(self, filepath):
        vercnt_bgn = filepath.find("-v")
        if vercnt_bgn == -1:
            return filepath

        vercnt_end = filepath.rfind(".")
        return filepath[0:vercnt_bgn] + filepath[vercnt_end:]

    def save_checkpoint(
        self, checkpoint: Dict[str, Any], filepath: _PATH, save_type_xser: bool, storage_options: Optional[Any] = None
    ) -> None:
        """Save model/training states as a checkpoint file through state-dump and file-write.

        Args:
            checkpoint: dict containing model and trainer state
            path: write-target path
            storage_options: not used in ``XLACheckpointIO.save_checkpoint``

        Raises:
            TypeError:
                If ``storage_options`` arg is passed in
        """

        filepath = self._remove_version_count(filepath)

        if storage_options is not None:
            raise TypeError(
                "`Trainer.save_checkpoint(..., storage_options=...)` with `storage_options` arg"
                f" is not supported for `{self.__class__.__name__}`. Please implement your custom `CheckpointIO`"
                " to define how you'd like to use `storage_options`."
            )

        # PTL override to accomodate model parallel checkpoints
        if RequirementCache("omegaconf"):
            # workaround for https://github.com/pytorch/xla/issues/2773
            from omegaconf import DictConfig, ListConfig, OmegaConf

            checkpoint = apply_to_collection(checkpoint, (DictConfig, ListConfig), OmegaConf.to_container)

        model = checkpoint.pop("state_dict")
        optimizer_states = checkpoint.pop("optimizer_states")
        if len(optimizer_states) > 1:
            raise RuntimeError("Error: currently nxd checkpoint does not support more than 1 optimizer")

        checkpoint_dir = os.path.dirname(filepath)
        checkpoint_tag = os.path.basename(filepath)
        nxd.save_checkpoint(
            checkpoint_dir,
            checkpoint_tag,
            model=model,
            optimizer=optimizer_states[0],
            user_content=checkpoint,
            use_xser=save_type_xser,
            async_save=self._async_save,
            zero1_optimizer=self._is_checkpoint_using_zero1_optimizer(checkpoint),
        )

    def _is_checkpoint_using_zero1_optimizer(self, checkpoint: dict):
        return checkpoint.get("hyper_parameters", {}).get("cfg", {}).get("wrap_with_zero", False)

    def remove_checkpoint(self, filepath: _PATH) -> None:
        if nxd.trainer.checkpoint.g_iostate is None:
            return
        checkpoint_tag = os.path.basename(filepath)
        nxd.trainer.checkpoint.g_iostate.submit_remove(-1, self._async_save, [checkpoint_tag])

    def teardown(self):
        nxd.finalize_checkpoint()


class NLPCheckpointConnector(CheckpointConnector):
    def restore_loops(self) -> None:
        """Restores the loop progress from the pre-loaded checkpoint.

        Calls hooks on the loops to give it a chance to restore its state from the checkpoint.
        """
        if not self._loaded_checkpoint:
            return

        fit_loop = self.trainer.fit_loop
        pl_module = self.trainer.lightning_module
        assert pl_module is not None

        global_step = self._loaded_checkpoint.get("global_step", 0)
        epoch = self._loaded_checkpoint.get("epoch", 0)
        # set the `global_step` value for checkpoints before v1.6 without the progress tracking state.
        # it will be overwritten by the loop's state if it was also saved
        batch_loop = fit_loop.epoch_loop
        if pl_module.automatic_optimization:
            batch_loop.automatic_optimization.optim_progress.optimizer.step.total.completed = global_step
        else:
            batch_loop.manual_optimization.optim_step_progress.total.completed = global_step

        # set the `current_epoch` value for checkpoints before v1.6 without the progress tracking state.
        # it will be overwritten by the loop's state if it was also saved
        fit_loop.epoch_progress.current.completed = epoch

        assert self.trainer.state.fn is not None
        state_dict = self._loaded_checkpoint.get("loops")
        if state_dict is not None:
            if self.trainer.state.fn == TrainerFn.FITTING:
                fit_loop.load_state_dict(state_dict["fit_loop"])
            elif self.trainer.state.fn == TrainerFn.VALIDATING:
                self.trainer.validate_loop.load_state_dict(state_dict["validate_loop"])
            elif self.trainer.state.fn == TrainerFn.TESTING:
                self.trainer.test_loop.load_state_dict(state_dict["test_loop"])
            elif self.trainer.state.fn == TrainerFn.PREDICTING:
                self.trainer.predict_loop.load_state_dict(state_dict["predict_loop"])

        if self.trainer.state.fn != TrainerFn.FITTING:
            return

        # crash if max_epochs is lower then the current epoch from the checkpoint
        if (
            self.trainer.max_epochs != -1
            and self.trainer.max_epochs is not None
            and self.trainer.current_epoch > self.trainer.max_epochs
        ):
            raise MisconfigurationException(
                f"You restored a checkpoint with current_epoch={self.trainer.current_epoch},"
                f" but you have set Trainer(max_epochs={self.trainer.max_epochs})."
            )

    def restore_optimizers_and_schedulers(self) -> None:
        """Restores the optimizers and learning rate scheduler states from the pre-loaded checkpoint."""
        if not self._loaded_checkpoint:
            return
        if self.trainer.strategy.lightning_restore_optimizer:
            # validation
            if "optimizer_states" not in self._loaded_checkpoint:
                logging.warning(
                    "Trying to restore optimizer state but checkpoint contains only the model."
                    " This is probably due to `ModelCheckpoint.save_weights_only` being set to `True`."
                )
                return
            self.restore_optimizers()

        if "lr_schedulers" not in self._loaded_checkpoint:
            logging.warning(
                "Trying to restore learning rate scheduler state but checkpoint contains only the model."
                " This is probably due to `ModelCheckpoint.save_weights_only` being set to `True`."
            )
            return
        self.restore_lr_schedulers()


class NLPAcceleratorConnector(AcceleratorConnector):
    def _validate_precision_choice(self) -> None:
        """Validate the combination of choices for precision, AMP type, and accelerator."""
        if self._precision_flag == 64:
            raise MisconfigurationException(
                "`Trainer(accelerator='tpu', precision=64)` is not implemented."
                " Please, open an issue in `https://github.com/Lightning-AI/lightning/issues`"
                " requesting this feature."
            )

    def _lazy_init_strategy(self) -> None:
        """Lazily set missing attributes on the previously instantiated strategy."""
        self.strategy.accelerator = self.accelerator
        if self.precision_plugin:
            # self.strategy.precision_plugin = self.precision_plugin
            self.strategy.precision_plugin = TRNPrecisionPlugin()

        if self.checkpoint_io:
            self.strategy.checkpoint_io = self.checkpoint_io
        if hasattr(self.strategy, "cluster_environment"):
            self.strategy.cluster_environment = self.cluster_environment
        if hasattr(self.strategy, "parallel_devices"):
            if self.strategy.parallel_devices:
                self._parallel_devices = self.strategy.parallel_devices
            else:
                self.strategy.parallel_devices = self._parallel_devices
        if hasattr(self.strategy, "num_nodes"):
            self.strategy._num_nodes = self._num_nodes_flag
        if hasattr(self.strategy, "_layer_sync"):
            self.strategy._layer_sync = self._layer_sync
        if hasattr(self.strategy, "set_world_ranks"):
            self.strategy.set_world_ranks()
        self.strategy._configure_launcher()


class NLPDataConnector(DataConnector):
    def _reset_eval_dataloader(self, mode, model):
        """Generic method to reset a dataloader for evaluation.

        Args:
            mode: The running stage of the ``Trainer``
            model: The ``LightningModule`` if calling this outside of the trainer scope.

        Returns:
            Tuple (num_batches, dataloaders)
        """
        assert mode.evaluating or mode == RunningStage.PREDICTING

        # always get the loaders first so we can count how many there are
        dataloaders = self._request_dataloader(mode)

        if self.trainer.overfit_batches > 0:
            dataloaders = self._resolve_overfit_batches(dataloaders, mode)

        if not isinstance(dataloaders, list):
            dataloaders = [dataloaders]  # type: ignore[assignment]

        if any(dl is None for dl in dataloaders):
            rank_zero_warn("One of given dataloaders is None and it will be skipped.")

        for loader in dataloaders:
            apply_to_collection(
                loader.loaders if isinstance(loader, CombinedLoader) else loader,
                DataLoader,
                self._check_eval_shuffling,
                mode=mode,
            )

        # add samplers
        dataloaders = [self._prepare_dataloader(dl, mode=mode) for dl in dataloaders if dl is not None]

        # add worker_init_fn for correct seeding in worker processes
        apply_to_collection(
            dataloaders, dtype=DataLoader, function=_auto_add_worker_init_fn, rank=self.trainer.global_rank
        )

        loader_num_batches: List[Union[int, float]] = []

        # determine number of batches
        module = model or self.trainer.lightning_module or self.datamodule
        if len(dataloaders) != 0:
            for i, dataloader in enumerate(dataloaders):
                orig_num_batches = num_batches = (
                    len(dataloader)
                    if has_len_all_ranks_patched(dataloader, self.trainer.strategy, module)
                    else float("inf")
                )

                if orig_num_batches == 0:
                    assert isinstance(orig_num_batches, int)
                    loader_num_batches.append(orig_num_batches)
                    continue

                self._worker_check(dataloader, f"{mode.dataloader_prefix}_dataloader {i}")

                # percent or num_steps
                limit_eval_batches = getattr(self.trainer, f"limit_{mode.dataloader_prefix}_batches")

                # limit num batches either as a percent or num steps
                if isinstance(limit_eval_batches, int):
                    num_batches = min(orig_num_batches, limit_eval_batches)
                elif isinstance(limit_eval_batches, float) and orig_num_batches != float("inf"):
                    num_batches = int(orig_num_batches * limit_eval_batches)
                elif limit_eval_batches != 1.0:
                    raise MisconfigurationException(
                        f"When using an `IterableDataset`, `Trainer(limit_{mode.dataloader_prefix}_batches)` must be"
                        f" `1.0` or an int. An int specifies `num_{mode.dataloader_prefix}_batches` to use."
                    )

                if (
                    num_batches == 0
                    and limit_eval_batches > 0.0
                    and isinstance(limit_eval_batches, float)
                    and orig_num_batches != float("inf")
                ):
                    min_percentage = 1.0 / orig_num_batches
                    raise MisconfigurationException(
                        f"You requested to check {limit_eval_batches} of the `{mode.dataloader_prefix}_dataloader` but"
                        f" {limit_eval_batches} * {orig_num_batches} < 1. Please increase the"
                        f" `limit_{mode.dataloader_prefix}_batches` argument. Try at least"
                        f" `limit_{mode.dataloader_prefix}_batches={min_percentage}`"
                    )

                loader_num_batches.append(num_batches)

        return loader_num_batches, dataloaders


    def _process_dataloader(
        trainer: "pl.Trainer", trainer_fn: TrainerFn, stage: RunningStage, dataloader: object
    ) -> object:
        '''
        This function is overrides _process_dataloader() from PTL 2.5, so that it correctly
        uses the Megatron custom sampler.
        '''
        if stage != RunningStage.TRAINING:
            is_shuffled = _is_dataloader_shuffled_patched(dataloader)
            # limit this warning only for samplers assigned automatically when shuffle is set
            if is_shuffled:
                rank_zero_warn(
                    f"Your `{stage.dataloader_prefix}_dataloader`'s sampler has shuffling enabled,"
                    " it is strongly recommended that you turn shuffling off for val/test dataloaders.",
                    category=PossibleUserWarning,
                )
        else:
            is_shuffled = True

        # automatically add samplers
        dataloader = trainer._data_connector._prepare_dataloader(dataloader, shuffle=is_shuffled, mode=stage)

        # let the strategy inject its logic
        dataloader = trainer.strategy.process_dataloader(dataloader)

        # check the workers
        _worker_check(
            trainer=trainer,
            dataloader=dataloader,
            name=f"{stage.dataloader_prefix}_dataloader",
        )

        # add worker_init_fn for correct seeding in worker processes
        _auto_add_worker_init_fn(dataloader, trainer.global_rank)

        if trainer_fn != TrainerFn.FITTING:  # if we are fitting, we need to do this in the loop
            # some users want validation shuffling based on the training progress
            _set_sampler_epoch(dataloader, trainer.fit_loop.epoch_progress.current.processed)

        return dataloader
    

class NLPTrainer(Trainer):
    @_defaults_from_env_vars
    def __init__(
        self,
        logger: Optional[Union[Logger, Iterable[Logger], bool]] = None,
        enable_checkpointing: Optional[bool] = False,
        callbacks: Optional[Union[List[Callback], Callback]] = None,
        default_root_dir: Optional[_PATH] = None,
        gradient_clip_val: Optional[Union[int, float]] = None,
        gradient_clip_algorithm: Optional[str] = None,
        num_nodes: int = 1,
        num_processes: Optional[int] = None,
        devices: Union[List[int], str, int] = "auto",
        enable_progress_bar: Optional[bool] = True,
        overfit_batches: Union[int, float] = 0.0,
        track_grad_norm: Union[int, float, str] = -1,
        check_val_every_n_epoch: Optional[int] = 1,
        fast_dev_run: Union[int, bool] = False,
        accumulate_grad_batches: int = 1,
        max_epochs: Optional[int] = None,
        min_epochs: Optional[int] = None,
        max_steps: int = -1,
        min_steps: Optional[int] = None,
        max_time: Optional[Union[str, timedelta, Dict[str, int]]] = None,
        limit_train_batches: Optional[Union[int, float]] = None,
        limit_val_batches: Optional[Union[int, float]] = None,
        limit_test_batches: Optional[Union[int, float]] = None,
        limit_predict_batches: Optional[Union[int, float]] = None,
        val_check_interval: Optional[Union[int, float]] = None,
        log_every_n_steps: Optional[int] = 50,
        accelerator: Union[str, Accelerator] = "tpu",
        strategy: Union[str, Strategy] = "auto",
        sync_batchnorm: bool = False,
        precision: Optional[_PRECISION_INPUT] = None,
        enable_model_summary: Optional[bool] = False,
        num_sanity_val_steps: Optional[int] = 2,
        profiler: Optional[Union[Profiler, str]] = None,
        benchmark: Optional[bool] = None,
        deterministic: Optional[Union[bool, _LITERAL_WARN]] = None,
        reload_dataloaders_every_n_epochs: int = 0,
        auto_lr_find: Union[bool, str] = False,
        detect_anomaly: bool = False,
        auto_scale_batch_size: Union[str, bool] = False,
        plugins: Optional[Union[PLUGIN_INPUT, List[PLUGIN_INPUT]]] = None,
        move_metrics_to_cpu: bool = False,
        multiple_trainloader_mode: str = "max_size_cycle",
        inference_mode: bool = True,
        lnc: int = None,
        sequential_move_factor: Optional[int] = None,
        barebones: bool = False,
        use_distributed_sampler: bool = False,
    ) -> None:
        logging.info(f"{self.__class__.__name__}: Initializing trainer with parameters: {locals()}")
        self.state = TrainerState()
        self.lnc = get_lnc_size(lnc)
        self.sequential_move_factor = sequential_move_factor
        self.barebones = barebones

        if default_root_dir is not None:
            default_root_dir = os.fspath(default_root_dir)

        if accumulate_grad_batches and accumulate_grad_batches > 1:
            raise ValueError(
                "gradient accumulation is handled by setting the global batchsize and micro-batchsize automatically"
            )

        if enable_model_summary:
            raise ValueError("enable_model_summary is not supported.")

        # init connectors
        self._data_connector = NLPDataConnector(self)
        self.multiple_trainloader_mode = multiple_trainloader_mode

        self._accelerator_connector = NLPAcceleratorConnector(
            devices=devices,
            accelerator=accelerator,
            strategy=strategy,
            num_nodes=num_nodes,
            sync_batchnorm=sync_batchnorm,
            benchmark=benchmark,
            deterministic=deterministic,
            precision=precision,
            plugins=plugins,
            use_distributed_sampler=use_distributed_sampler
        )
        self._logger_connector = LoggerConnector(self)
        self._callback_connector = CallbackConnector(self)
        self._checkpoint_connector = NLPCheckpointConnector(self)
        self._signal_connector = SignalConnector(self)

        # move training epoch loop into fitloop
        self.fit_loop = NLPFitLoop(self, min_epochs=min_epochs, max_epochs=max_epochs)

        # default .fit() loop
        self.fit_loop.epoch_loop = NLPTrainingEpochLoop(self, min_steps=min_steps, max_steps=max_steps)
        # default .validate() loop
        self.validate_loop = NLPEvaluationLoop(
            self, trainer_fn=TrainerFn.VALIDATING, stage=RunningStage.VALIDATING, inference_mode=inference_mode
        )

        # default .test() loop
        self.test_loop = NLPEvaluationLoop(self, trainer_fn=TrainerFn.TESTING, stage=RunningStage.TESTING, inference_mode=inference_mode)
        
        # default .predict() loop
        self.predict_loop = PredictionLoop(self, inference_mode=inference_mode)

        # set when a checkpoint is loaded via `Trainer.{fit,validate,test,predict}`.
        self._ckpt_path: Optional[str] = None

        # init callbacks

        # Declare attributes to be set in _callback_connector on_trainer_init
        self.accumulate_grad_batches = accumulate_grad_batches
        self._callback_connector.on_trainer_init(
            callbacks,
            enable_checkpointing,
            enable_progress_bar,
            default_root_dir,
            enable_model_summary,
            max_time,
        )

        # init data flags
        self.check_val_every_n_epoch: Optional[int]
        self._data_connector.on_trainer_init(
            val_check_interval,
            reload_dataloaders_every_n_epochs,
            check_val_every_n_epoch,
        )

        # gradient clipping
        if gradient_clip_val is not None and not isinstance(gradient_clip_val, (int, float)):
            raise TypeError(f"`gradient_clip_val` should be an int or a float. Got {gradient_clip_val}.")
        
        if gradient_clip_algorithm is not None:
            raise MisconfigurationException(
                "`gradient_clip_algorithm` is not supported through Pytorch-lightning"
            )

        # gradient norm tracking
        if track_grad_norm != -1 and not (
            (isinstance(track_grad_norm, (int, float)) or track_grad_norm == "inf") and float(track_grad_norm) > 0
        ):
            raise MisconfigurationException(
                f"`track_grad_norm` must be a positive number or 'inf' (infinity norm). Got {track_grad_norm}."
            )

        self.gradient_clip_val: Optional[Union[int, float]] = gradient_clip_val
        # self.gradient_clip_algorithm: Optional[GradClipAlgorithmType] = (
        #     GradClipAlgorithmType(gradient_clip_algorithm.lower()) if gradient_clip_algorithm is not None else None
        # )
        self.gradient_clip_algorithm = None
        self.track_grad_norm: float = float(track_grad_norm)

        self._inference_mode: bool = inference_mode

        self._detect_anomaly: bool = detect_anomaly

        setup._log_device_info(self)
        self.should_stop = False

        # configure profiler
        setup._init_profiler(self, profiler)

        # init logger flags
        self._loggers: List[Logger]
        self._logger_connector.on_trainer_init(logger, log_every_n_steps)

        # init debugging flags
        self.val_check_batch: Union[int, float]
        self.val_check_interval: Union[int, float]
        self.num_sanity_val_steps: Union[int, float]
        self.limit_train_batches: Union[int, float]
        self.limit_val_batches: Union[int, float]
        self.limit_test_batches: Union[int, float]
        self.limit_predict_batches: Union[int, float]
        setup._init_debugging_flags(
            self,
            limit_train_batches,
            limit_val_batches,
            limit_test_batches,
            limit_predict_batches,
            fast_dev_run,
            overfit_batches,
            val_check_interval,
            num_sanity_val_steps,
        )

    def _restore_modules_and_callbacks(self, checkpoint_path: Optional[_PATH] = None) -> None:
        self._checkpoint_connector.resume_start(checkpoint_path)
        self._checkpoint_connector._restore_quantization_callbacks()
        self._checkpoint_connector.restore_model()
        self._checkpoint_connector.restore_datamodule()
        if self.state.fn == TrainerFn.FITTING:
            # restore callback states
            self._checkpoint_connector.restore_callbacks()


class NLPDDPStrategy(XLAStrategy):
    """DDP plugin for Pytorch Lightning. Needed to customize DDP for model parallel models.

    This class overrides certain API's from XLAStrategy for NxDT. For more info on 
    XLAStrategy, please see: https://lightning.ai/docs/pytorch/2.5.0/api/lightning.pytorch.strategies.XLAStrategy.html

    This class overrides certain API's from TPUSpawnStrategy for NxDT. For more info on 
    TPUSpawnStrategy, please see: https://lightning.ai/docs/pytorch/1.8.6/api/pytorch_lightning.strategies.TPUSpawnStrategy

    Args:
        no_ddp_communication_hook: Disable DDP communication hook when using AMP-O2
        with FP32 gradient accumulation.
    """

    def __init__(
        self,
        accelerator: Optional["pl.accelerators.Accelerator"] = None,
        parallel_devices: Optional[List[torch.device]] = None,
        cluster_environment: Optional[ClusterEnvironment] = None,
        checkpoint_io: Optional[CheckpointIO] = None,
        precision_plugin: Optional[PrecisionPlugin] = None,
        debug: bool = False,
        sync_module_states: bool = False,
        **_: Any,
    ) -> None:
        if cluster_environment is None:
            cluster_environment = XLAEnvironment()
        super(XLAStrategy, self).__init__(
            accelerator=accelerator,
            parallel_devices=parallel_devices,
            cluster_environment=cluster_environment,
            checkpoint_io=checkpoint_io,
            precision_plugin=precision_plugin,
            start_method="fork",
            sync_module_states=sync_module_states
        )
        self._checkpoint_io: Optional[CheckpointIO]
        self.debug = debug
        self._launched = False
        self._sync_module_states = sync_module_states
        self._init_torch_dist()

    def _configure_launcher(self) -> None:
        self._launcher = _NLPXLALauncher(self)
    
    def _init_torch_dist(self):
        # call PTL init ddp
        if torch.__version__.startswith("2.0"):
            import torch_xla.experimental.pjrt_backend  # noqa
            torch.distributed.init_process_group("xla", init_method="pjrt://")
        else:
            torch.distributed.init_process_group("xla")

        self._launched = True
        
    def setup_distributed(self, global_rank: int = None, world_size: int = None) -> None:
        super().setup_distributed()

    def is_save_type_xser(self):
        return self.lightning_module.config.exp_manager.get("save_xser", False)

    def save_checkpoint(
        self, checkpoint: Dict[str, Any], filepath: _PATH, storage_options: Optional[Any] = None
    ) -> None:
        # exp_manager calls this on_train_batch_end before fwd/bwd/optimizer tracing is finished
        xm.mark_step()
        self.checkpoint_io.save_checkpoint(checkpoint, filepath, self.is_save_type_xser())

    def is_load_type_xser(self):
        return self.lightning_module.config.exp_manager.get("load_xser", False)

    def load_checkpoint(self, checkpoint_path: _PATH) -> Dict[str, Any]:
        return self.checkpoint_io.load_checkpoint(checkpoint_path, self.is_load_type_xser())

    def load_model_state_dict(self, checkpoint: Mapping[str, Any], strict: bool = False) -> None:
        # Release strict state dict matching when using Megatron AMP-O2 to skip matching
        # half-precision module wrapper module.
        # TODO: Refactor this to be more generic.
        model_key = None
        model_attr = None
        if hasattr(self.lightning_module, "model"):
            model_key = "model"
            model_attr = self.lightning_module.model
        elif hasattr(self.lightning_module, "enc_dec_model"):
            model_key = "enc_dec_model"
            model_attr = self.lightning_module.enc_dec_model
        if model_key is not None:
            if isinstance(model_attr, Float16Module):
                new_state_dict = {}
                for key in checkpoint["state_dict"].keys():
                    new_key = key.replace(f"{model_key}.", f"{model_key}.module.", 1)
                    new_state_dict[new_key] = checkpoint["state_dict"][key]
                checkpoint["state_dict"] = new_state_dict

        load_result = self.lightning_module.load_state_dict(checkpoint["state_dict"], strict=False)

        # Print out the unexpected keys
        if load_result.unexpected_keys:
            logging.warning(f"Warning: Unexpected keys in state dictionary: {', '.join(load_result.unexpected_keys)}")

        # Filter out 'inv_freq' from the missing keys - as it is created from scratch
        real_missing_keys = [
            k
            for k in load_result.missing_keys
            if not any(key_pattern in k for key_pattern in self._key_patterns_to_be_ignored())
        ]

        # Print out the real missing keys and throw an exception if there are any
        if real_missing_keys and not get_attribute_from_cfg(self.lightning_module.config, "peft", False):
            logging.error(f"Error: Missing keys when loading state dictionary: {', '.join(real_missing_keys)}")
            raise RuntimeError(f"Missing keys when loading state dictionary: {', '.join(real_missing_keys)}")

    def _key_patterns_to_be_ignored(self):
        """
        This function gives child of NLPDDPStrategy to extend list
        of key patterns to be ignored from missing keys
        """
        return [".rotary_emb.inv_freq"]

    def remove_checkpoint(self, filepath: _PATH) -> None:
        self.checkpoint_io.remove_checkpoint(filepath)

    @property
    def is_distributed(self) -> bool:
        # HOST_WORLD_SIZE is not set outside the xmp.spawn process
        # HOST_WORLD_SIZE only exists in XRT, not PJRT
        import torch_xla.core.xla_env_vars as xenv

        if torch.__version__.startswith("2"):
            return self.world_size != 1

        return (xenv.HOST_WORLD_SIZE in os.environ) and self.world_size != 1

    @property
    def distributed_sampler_kwargs(self):
        app_state = AppState()
        if app_state.model_parallel_size is not None:
            # When using model parallel, data parallel groups are non-trivial and they
            # correspond to the logical GPUs. This means that the GPUs that form a
            # single logical GPU all need to get the same batch of data.
            if parallel_state.get_data_parallel_size() <= 1 and self.trainer.use_distributed_sampler:
                raise ValueError("Data parallel size must be >= 1 if use_distributed_sampler is True")

            distributed_sampler_kwargs = dict(
                num_replicas=parallel_state.get_data_parallel_size(), rank=parallel_state.get_data_parallel_rank()
            )
            return distributed_sampler_kwargs

        else:
            return super(NLPDDPStrategy, self).distributed_sampler_kwargs

    def process_dataloader(self, dataloader):
        return dataloader

    def broadcast(self, obj, src: int = 0):
        return obj

    def teardown(self):
        """This method is called to teardown the training process.

        It is the right place to release memory and free other resources.
        """
        #### Avoid copying to CPU
        self.precision_plugin.teardown()
        assert self.accelerator is not None
        self.accelerator.teardown()
        self.checkpoint_io.teardown()

    # original implementation of this function would go over GRPC and hits message size limit
    # when number of workers is > 128
    # https://github.com/pytorch/xla/issues/1924

    def reduce(
        self,
        output: Union[Tensor, Any],
        group: Optional[Any] = None,
        reduce_op: Optional[Union[torch.distributed.ReduceOp, str]] = "mean",
    ) -> Tensor:
        if not isinstance(output, Tensor):
            output = torch.tensor(output, device=self.root_device)

        invalid_reduce_op = (
            isinstance(reduce_op, torch.distributed.ReduceOp) and reduce_op != torch.distributed.ReduceOp.SUM
        )
        invalid_reduce_op_str = isinstance(reduce_op, str) and reduce_op.lower() not in ("sum", "mean", "avg")
        if invalid_reduce_op or invalid_reduce_op_str:
            raise ValueError(
                "Currently, the XLAStrategy only supports `sum`, `mean`, `avg` for the reduce operation, got:"
                f" {reduce_op}"
            )

        xm.mark_step()
        torch.distributed.all_reduce(
            output, op=torch.distributed.ReduceOp.SUM, group=parallel_state.get_tensor_model_parallel_group()
        )
        xm.mark_step()
        torch.distributed.all_reduce(
            output, op=torch.distributed.ReduceOp.SUM, group=parallel_state.get_pipeline_model_parallel_group()
        )
        xm.mark_step()
        torch.distributed.all_reduce(
            output, op=torch.distributed.ReduceOp.SUM, group=parallel_state.get_data_parallel_group()
        )
        xm.mark_step()

        if isinstance(reduce_op, str) and reduce_op.lower() in ("avg", "mean"):
            output = output / self.world_size

        xm.mark_step()
        return output.cpu()