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
import pytorch_lightning as pl
import torch
import torch.multiprocessing as mp
from lightning_lite.plugins import ClusterEnvironment, XLACheckpointIO
from lightning_lite.plugins.environments import XLAEnvironment
from lightning_lite.strategies.launchers.xla import _rank_teardown
from lightning_lite.utilities.data import _auto_add_worker_init_fn
from lightning_lite.utilities.data import (
    has_iterable_dataset as new_has_iterable_dataset,
)
from lightning_lite.utilities.types import _PATH, Optimizable
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
from pytorch_lightning.loops import (
    OptimizerLoop,
    PredictionLoop,
    TrainingBatchLoop,
    TrainingEpochLoop,
)
from pytorch_lightning.loops.dataloader.evaluation_loop import EvaluationLoop
from pytorch_lightning.loops.fit_loop import FitLoop, _select_data_fetcher
from pytorch_lightning.loops.utilities import _block_parallel_sync_behavior
from pytorch_lightning.plugins import PLUGIN_INPUT
from pytorch_lightning.plugins.io.checkpoint_plugin import CheckpointIO
from pytorch_lightning.plugins.precision import PrecisionPlugin
from pytorch_lightning.profilers import Profiler
from pytorch_lightning.strategies import Strategy, TPUSpawnStrategy
from pytorch_lightning.strategies.launchers.xla import _XLALauncher
from pytorch_lightning.trainer import setup
from pytorch_lightning.trainer.connectors.accelerator_connector import (
    _LITERAL_WARN,
    AcceleratorConnector,
)
from pytorch_lightning.trainer.connectors.callback_connector import CallbackConnector
from pytorch_lightning.trainer.connectors.checkpoint_connector import (
    CheckpointConnector,
)
from pytorch_lightning.trainer.connectors.data_connector import DataConnector
from pytorch_lightning.trainer.connectors.logger_connector import LoggerConnector
from pytorch_lightning.trainer.connectors.logger_connector.result import (
    _ResultCollection,
    _ResultMetric,
    _ResultMetricCollection,
)
from pytorch_lightning.trainer.connectors.signal_connector import SignalConnector
from pytorch_lightning.trainer.states import RunningStage, TrainerFn, TrainerState
from pytorch_lightning.trainer.supporters import CombinedLoader
from pytorch_lightning.trainer.trainer import Trainer
from pytorch_lightning.tuner.tuning import Tuner
from pytorch_lightning.utilities.argparse import _defaults_from_env_vars
from pytorch_lightning.utilities.auto_restart import _add_capture_metadata_collate
from pytorch_lightning.utilities.exceptions import MisconfigurationException
from pytorch_lightning.utilities.imports import _fault_tolerant_training
from pytorch_lightning.utilities.model_helpers import is_overridden
from pytorch_lightning.utilities.rank_zero import rank_zero_warn
from pytorch_lightning.utilities.warnings import PossibleUserWarning
from torch import Tensor
from torch.utils.data import DataLoader
from torch_xla.distributed.zero_redundancy_optimizer import ZeroRedundancyOptimizer
from torchmetrics import Metric
from neuronx_distributed_training.utils import get_vnc_size


def has_len_all_ranks_patched(
    dataloader,
    strategy,
    model,
) -> bool:
    """Checks if a given Dataloader has ``__len__`` method implemented i.e. if it is a finite dataloader or
    infinite dataloader."""
    try:
        local_length = len(dataloader)  # type: ignore [arg-type] # we are checking with duck-typing
        total_length = strategy.reduce(torch.tensor(local_length, device=strategy.root_device), reduce_op="sum")

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
    if has_len and new_has_iterable_dataset(dataloader):  # type: ignore [arg-type]
        rank_zero_warn(
            "Your `IterableDataset` has `__len__` defined."
            " In combination with multi-process data loading (when num_workers > 1),"
            " `__len__` could be inaccurate if each worker is not configured independently"
            " to avoid having duplicate data."
        )
    if os.environ.get("NEURON_EXTRACT_GRAPHS_ONLY", None):
        return True
    return has_len


class TRNPrecisionPlugin(PrecisionPlugin):
    """Precision plugin for TPU integration."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)

    def optimizer_step(  # type: ignore[override]
        self,
        optimizer: Optimizable,
        model: "pl.LightningModule",
        optimizer_idx: int,
        closure: Callable[[], Any],
        **kwargs: Any,
    ) -> Any:
        """Hook to run the optimizer step."""
        if not isinstance(optimizer, ZeroRedundancyOptimizer):
            closure = partial(self._wrap_closure, model, optimizer, optimizer_idx, closure)
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

        value = apply_to_collection(value, (Tensor, Metric), fn)
        if isinstance(value, dict):
            value = _ResultMetricCollection(value)
        self[key] = value

    def update_metrics(self, key: str, value, batch_size: int) -> None:
        def fn(result_metric, v):
            # performance: avoid calling `__call__` to avoid the checks in `torch.nn.Module._call_impl`
            ### NEURON: Do not move metrics to device, results in unnnecessary compiles
            result_metric.forward(v, batch_size)
            result_metric.has_reset = False

        apply_to_collections(self[key], value, _ResultMetric, fn)


class NLPEvaluationLoop(EvaluationLoop):
    # We override this class to make sure we use _NLPResultCollection
    # and avoid transferring results to device
    def __init__(self, verbose: bool = True) -> None:
        super().__init__(verbose)
        self._results = _NLPResultCollection(training=False)

    def teardown(self) -> None:
        if self._data_fetcher is not None:
            self._data_fetcher.teardown()
            self._data_fetcher = None
        self.epoch_loop.teardown()

    def _on_evaluation_start(self, *args: Any, **kwargs: Any) -> None:
        """Runs ``on_{validation/test}_start`` hooks."""
        assert self._results is not None

        hook_name = "on_test_start" if self.trainer.testing else "on_validation_start"
        self.trainer._call_callback_hooks(hook_name, *args, **kwargs)
        self.trainer._call_lightning_module_hook(hook_name, *args, **kwargs)
        self.trainer._call_strategy_hook(hook_name, *args, **kwargs)


class NLPTrainingEpochLoop(TrainingEpochLoop):
    def __init__(self, min_steps: Optional[int] = None, max_steps: int = -1) -> None:
        super().__init__(min_steps, max_steps)
        self.val_loop = NLPEvaluationLoop(verbose=True)
        self._results = _NLPResultCollection(training=True)


class NLPFitLoop(FitLoop):
    # We override this class to make sure results are on CPU on run start
    def on_run_start(self) -> None:
        """Calls the ``on_train_start`` hook."""
        # update the current_epoch in-case of checkpoint reload
        if not self._iteration_based_training():
            self.epoch_progress.current.completed = self.epoch_progress.current.processed

        self.trainer.reset_train_dataloader(self.trainer.lightning_module)
        # reload the evaluation dataloaders too for proper display in the progress bar
        if self.epoch_loop._should_check_val_epoch():
            self.epoch_loop.val_loop._reload_evaluation_dataloaders()

        data_fetcher_cls = _select_data_fetcher(self.trainer)
        self._data_fetcher = data_fetcher_cls(prefetch_batches=self.prefetch_batches)

        self._is_fresh_start_epoch = True
        self._results.cpu()

        self.trainer._call_callback_hooks("on_train_start")
        self.trainer._call_lightning_module_hook("on_train_start")
        self.trainer._call_strategy_hook("on_train_start")


class NLPCheckpointIO(XLACheckpointIO):
    def __init__(self, async_save=False, weight_init_only=False):
        super().__init__()
        self._async_save = async_save
        self._weight_init_only = weight_init_only

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
        batch_loop = fit_loop.epoch_loop.batch_loop
        if pl_module.automatic_optimization:
            batch_loop.optimizer_loop.optim_progress.optimizer.step.total.completed = global_step
        else:
            batch_loop.manual_loop.optim_step_progress.total.completed = global_step

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


class NLPTrainer(Trainer):
    @_defaults_from_env_vars
    def __init__(
        self,
        logger: Union[Logger, Iterable[Logger], bool] = False,  # Change the default to False
        enable_checkpointing: bool = False,  # Change the default to False
        callbacks: Optional[Union[List[Callback], Callback]] = None,
        default_root_dir: Optional[_PATH] = None,
        gradient_clip_val: Optional[Union[int, float]] = None,
        gradient_clip_algorithm: Optional[str] = None,
        num_nodes: int = 1,
        num_processes: Optional[int] = None,  # TODO: Remove in 2.0
        devices: Optional[Union[List[int], str, int]] = None,
        gpus: Optional[Union[List[int], str, int]] = None,  # TODO: Remove in 2.0
        auto_select_gpus: bool = False,
        tpu_cores: Optional[Union[List[int], str, int]] = None,  # TODO: Remove in 2.0
        ipus: Optional[int] = None,  # TODO: Remove in 2.0
        enable_progress_bar: bool = True,
        overfit_batches: Union[int, float] = 0.0,
        track_grad_norm: Union[int, float, str] = -1,
        check_val_every_n_epoch: Optional[int] = 1,
        fast_dev_run: Union[int, bool] = False,
        accumulate_grad_batches: Optional[Union[int, Dict[int, int]]] = None,
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
        log_every_n_steps: int = 50,
        accelerator: Optional[Union[str, Accelerator]] = "tpu",
        strategy: Optional[Union[str, Strategy]] = None,
        sync_batchnorm: bool = False,
        precision: Union[int, str] = 32,
        enable_model_summary: bool = False,  # Change the default to False
        num_sanity_val_steps: int = 2,
        resume_from_checkpoint: Optional[Union[Path, str]] = None,
        profiler: Optional[Union[Profiler, str]] = None,
        benchmark: Optional[bool] = None,
        deterministic: Optional[Union[bool, _LITERAL_WARN]] = None,
        reload_dataloaders_every_n_epochs: int = 0,
        auto_lr_find: Union[bool, str] = False,
        replace_sampler_ddp: bool = False,  # Change the default to False
        detect_anomaly: bool = False,
        auto_scale_batch_size: Union[str, bool] = False,
        plugins: Optional[Union[PLUGIN_INPUT, List[PLUGIN_INPUT]]] = None,
        amp_backend: str = "native",
        amp_level: Optional[str] = None,
        move_metrics_to_cpu: bool = False,
        multiple_trainloader_mode: str = "max_size_cycle",
        inference_mode: bool = True,
        vnc: int = None,
    ) -> None:
        Trainer._log_api_event("init")
        logging.info(f"{self.__class__.__name__}: Initializing trainer with parameters: {locals()}")
        self.state = TrainerState()
        self.vnc = get_vnc_size(vnc)

        if default_root_dir is not None:
            default_root_dir = os.fspath(default_root_dir)

        if accumulate_grad_batches and accumulate_grad_batches > 1:
            raise ValueError(
                "gradient accumulation is handled by setting the global batchsize and micro-batchsize automatically"
            )

        if enable_model_summary:
            raise ValueError("enable_model_summary is not supported.")

        # init connectors
        self._data_connector = NLPDataConnector(self, multiple_trainloader_mode)

        self._accelerator_connector = NLPAcceleratorConnector(
            num_processes=num_processes,
            devices=devices,
            tpu_cores=tpu_cores,
            ipus=ipus,
            accelerator=accelerator,
            strategy=strategy,
            gpus=gpus,
            num_nodes=num_nodes,
            sync_batchnorm=sync_batchnorm,
            benchmark=benchmark,
            replace_sampler_ddp=replace_sampler_ddp,
            deterministic=deterministic,
            auto_select_gpus=auto_select_gpus,
            precision=precision,
            amp_type=amp_backend,
            amp_level=amp_level,
            plugins=plugins,
        )
        self._logger_connector = LoggerConnector(self)
        self._callback_connector = CallbackConnector(self)
        self._resume_from_checkpoint = resume_from_checkpoint
        self._checkpoint_connector = NLPCheckpointConnector(self, resume_from_checkpoint)
        self._signal_connector = SignalConnector(self)
        self.tuner = Tuner(self)

        fit_loop = NLPFitLoop(min_epochs=min_epochs, max_epochs=max_epochs)
        training_epoch_loop = NLPTrainingEpochLoop(min_steps=min_steps, max_steps=max_steps)
        fit_loop.connect(epoch_loop=training_epoch_loop)

        # default .fit() loop
        self.fit_loop = fit_loop

        # default .validate() loop
        self.validate_loop = NLPEvaluationLoop()

        # default .test() loop
        self.test_loop = NLPEvaluationLoop()

        # default .predict() loop
        self.predict_loop = PredictionLoop()

        # set when a checkpoint is loaded via `Trainer.{fit,validate,test,predict}`.
        self._ckpt_path: Optional[str] = None

        # init callbacks
        # Declare attributes to be set in _callback_connector on_trainer_init
        self._callback_connector.on_trainer_init(
            callbacks,
            enable_checkpointing,
            enable_progress_bar,
            default_root_dir,
            enable_model_summary,
            max_time,
            accumulate_grad_batches,
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
        self.gradient_clip_algorithm = None
        self.track_grad_norm: float = float(track_grad_norm)

        self._inference_mode: bool = inference_mode

        self._detect_anomaly: bool = detect_anomaly
        self._setup_on_init()

        # configure tuner
        self.tuner.on_trainer_init(auto_lr_find, auto_scale_batch_size)

        # configure profiler
        setup._init_profiler(self, profiler)

        # init logger flags
        self._loggers: List[Logger]
        self._logger_connector.on_trainer_init(logger, log_every_n_steps, move_metrics_to_cpu)

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

    def is_resuming_from_checkpoint(self):
        return self._resume_from_checkpoint is not None

    def reset_train_dataloader(self, model: Optional["pl.LightningModule"] = None) -> None:
        """Resets the train dataloader and initialises required variables (number of batches, when to validate,
        etc.).

        Args:
            model: The ``LightningModule`` if calling this outside of the trainer scope.
        """
        source = self._data_connector._train_dataloader_source
        pl_module = model or self.lightning_module
        has_step = is_overridden("training_step", pl_module)
        enable_training = self.limit_train_batches > 0
        if not (source.is_defined() and has_step and enable_training):
            return

        self.train_dataloader = self._data_connector._request_dataloader(RunningStage.TRAINING)

        if self.overfit_batches > 0:
            self.train_dataloader = self._data_connector._resolve_overfit_batches(
                self.train_dataloader, mode=RunningStage.TRAINING
            )

        # automatically add samplers
        self.train_dataloader = apply_to_collection(
            self.train_dataloader,
            (DataLoader, CombinedLoader),
            self._data_connector._prepare_dataloader,
            mode=RunningStage.TRAINING,
        )
        loaders = (
            self.train_dataloader.loaders
            if isinstance(self.train_dataloader, CombinedLoader)
            else self.train_dataloader
        )

        # check the workers recursively
        apply_to_collection(loaders, DataLoader, self._data_connector._worker_check, "train_dataloader")

        # add worker_init_fn for correct seeding in worker processes
        apply_to_collection(loaders, DataLoader, _auto_add_worker_init_fn, rank=self.global_rank)

        # add collate_fn to collect metadata for fault tolerant training
        if _fault_tolerant_training():
            apply_to_collection(loaders, DataLoader, _add_capture_metadata_collate)

        # wrap the sequence of train loaders to a CombinedLoader object for computing the num_training_batches
        if not isinstance(self.train_dataloader, CombinedLoader):
            self.train_dataloader = CombinedLoader(loaders, self._data_connector.multiple_trainloader_mode)

        module = model or self.lightning_module or self.datamodule
        orig_train_batches = self.num_training_batches = (
            len(self.train_dataloader)  # type: ignore[arg-type]
            if has_len_all_ranks_patched(self.train_dataloader, self.strategy, module)
            else float("inf")
        )
        if orig_train_batches == 0:
            return

        # store epoch of dataloader reset for reload_dataloaders_every_n_epochs
        self._last_train_dl_reload_epoch = self.current_epoch

        if isinstance(self.limit_train_batches, int):
            self.num_training_batches = min(orig_train_batches, self.limit_train_batches)
        elif self.num_training_batches != float("inf"):
            self.num_training_batches = int(orig_train_batches * self.limit_train_batches)
        elif self.limit_train_batches != 1.0:
            raise MisconfigurationException(
                "When using an `IterableDataset`, `Trainer(limit_train_batches)` must be `1.0` or an int."
                "An int specifies `num_training_batches` to use."
            )

        if isinstance(self.val_check_interval, int):
            self.val_check_batch = self.val_check_interval
            if self.val_check_batch > self.num_training_batches and self.check_val_every_n_epoch is not None:
                raise ValueError(
                    f"`val_check_interval` ({self.val_check_interval}) must be less than or equal "
                    f"to the number of the training batches ({self.num_training_batches}). "
                    "If you want to disable validation set `limit_val_batches` to 0.0 instead."
                    "If you want to validate based on the total training batches, set `check_val_every_n_epoch=None`."
                )
        else:
            if not has_len_all_ranks_patched(self.train_dataloader, self.strategy, module):
                if self.val_check_interval == 1.0:
                    self.val_check_batch = float("inf")
                else:
                    raise MisconfigurationException(
                        "When using an IterableDataset for `train_dataloader`,"
                        " `Trainer(val_check_interval)` must be `1.0` or an int. An int k specifies"
                        " checking validation every k training batches."
                    )
            else:
                self.val_check_batch = int(self.num_training_batches * self.val_check_interval)
                self.val_check_batch = max(1, self.val_check_batch)

        if self.loggers and self.num_training_batches < self.log_every_n_steps:
            rank_zero_warn(
                f"The number of training batches ({self.num_training_batches}) is smaller than the logging interval"
                f" Trainer(log_every_n_steps={self.log_every_n_steps}). Set a lower value for log_every_n_steps if"
                " you want to see logs for the training epoch.",
                category=PossibleUserWarning,
            )

        if (
            self.num_training_batches == 0
            and self.limit_train_batches > 0.0
            and isinstance(self.limit_train_batches, float)
            and orig_train_batches != float("inf")
        ):
            min_percentage = 1.0 / orig_train_batches
            raise MisconfigurationException(
                f"You requested to check {self.limit_train_batches} of the `train_dataloader` but"
                f" {self.limit_train_batches} * {orig_train_batches} < 1. Please increase the"
                f" `limit_train_batches` argument. Try at least"
                f" `limit_train_batches={min_percentage}`"
            )


class NLPDDPStrategy(TPUSpawnStrategy):
    """DDP plugin for Pytorch Lightning. Needed to customize DDP for model parallel models.

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
        no_ddp_communication_hook: bool = True,  # Chamge the default to true
        megatron_amp_o2: bool = False,
        restore_path=None,
        **_: Any,
    ) -> None:
        if cluster_environment is None:
            cluster_environment = XLAEnvironment()
        super(TPUSpawnStrategy, self).__init__(
            accelerator=accelerator,
            parallel_devices=parallel_devices,
            cluster_environment=cluster_environment,
            checkpoint_io=checkpoint_io,
            precision_plugin=precision_plugin,
            start_method="fork",
        )
        self._checkpoint_io: Optional[CheckpointIO]
        self.debug = debug
        self._launched = False
        self.no_ddp_communication_hook = no_ddp_communication_hook
        self.megatron_amp_o2 = megatron_amp_o2
        self.restore_path = restore_path
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

    def setup_distributed(self, global_rank: int = None, world_size: int = None) -> None:
        super().setup_distributed()

    def is_save_type_xser(self):
        return self.lightning_module.config.exp_manager.get("save_xser", False)

    def save_checkpoint(
        self, checkpoint: Dict[str, Any], filepath: _PATH, storage_options: Optional[Any] = None
    ) -> None:
        self.checkpoint_io.save_checkpoint(checkpoint, filepath, self.is_save_type_xser())

    def is_load_type_xser(self):
        return self.lightning_module.config.exp_manager.get("load_xser", False)

    def load_checkpoint(self, checkpoint_path: _PATH) -> Dict[str, Any]:
        return self.checkpoint_io.load_checkpoint(checkpoint_path, self.is_load_type_xser())

    def load_model_state_dict(self, checkpoint: Mapping[str, Any]) -> None:
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
        if real_missing_keys:
            logging.error(f"Error: Missing keys when loading state dictionary: {', '.join(real_missing_keys)}")
            raise RuntimeError(f"Missing keys when loading state dictionary: {', '.join(real_missing_keys)}")

    def _key_patterns_to_be_ignored(self):
        """
        This function gives child of NLPDDPStrategy to extend list
        of key patterns to be ignored from missing keys
        """
        return [".rotary_emb.inv_freq"]

    def remove_checkpoint(self, filepath: _PATH) -> None:
        if not self.restore_path:
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
            distributed_sampler_kwargs = dict(
                num_replicas=app_state.data_parallel_size, rank=app_state.data_parallel_rank
            )
            return distributed_sampler_kwargs

        else:
            return super(NLPDDPStrategy, self).distributed_sampler_kwargs

    def process_dataloader(self, dataloader):
        TPUSpawnStrategy._validate_dataloader(dataloader)
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
                "Currently, the TPUSpawnStrategy only supports `sum`, `mean`, `avg` for the reduce operation, got:"
                f" {reduce_op}"
            )

        import torch_xla.core.xla_model as xm

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