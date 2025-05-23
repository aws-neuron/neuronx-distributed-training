# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
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

import os
import sys
import time
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path
from shutil import copy, move
from typing import Dict, Optional, Union

# import pytorch_lightning
import lightning.pytorch as pl
from nemo.collections.common.callbacks import EMA
from nemo.constants import NEMO_ENV_VARNAME_TESTING
from nemo.utils import exp_manager as nemo_exp_manager
from nemo.utils import logging
from nemo.utils.app_state import AppState
from nemo.utils.env_var_parsing import get_envbool
from nemo.utils.get_rank import is_global_rank_zero
from nemo.utils.lightning_logger_patch import add_filehandlers_to_pl_logger
from nemo.utils.model_utils import uninject_model_parallel_rank
from omegaconf import DictConfig, OmegaConf, open_dict
from pytorch_lightning.callbacks import ModelCheckpoint


@dataclass
class CallbackParams(nemo_exp_manager.CallbackParams):
    every_n_epochs: Optional[int] = 0
    every_n_train_steps: Optional[int] = None
    train_time_interval: Optional[int] = None
    save_on_train_epoch_end: Optional[bool] = False  # Save after training, not after validation
    save_nemo_on_train_end: Optional[bool] = False
    use_master_weights_in_ckpt: Optional[bool] = False
    enable_version_counter: Optional[bool] = False


@dataclass
class ExpManagerConfig(nemo_exp_manager.ExpManagerConfig):
    checkpoint_callback_params: Optional[CallbackParams] = CallbackParams()
    log_parameter_norm: Optional[bool] = True # Logs parameter norm across model parallel ranks
    log_gradient_norm: Optional[bool] = True # Logs gradient norm across model parallel ranks
    enable_recovery_time_instrumentation: Optional[bool] = False # default to not printing the detailing timing for recovery
    save_xser: Optional[bool] = True
    load_xser: Optional[bool] = True
    save_bf16: Optional[bool] = False
    async_checkpointing: Optional[bool] = False # default to not use async checkpointing
    resume_from_checkpoint: Optional[str] = None # manually set the checkpoint file to load from
    ckpt_ptl_version: Optional[str] = None # PTL version used of checkpoint


class TimingCallback(nemo_exp_manager.TimingCallback):
    """
    Logs execution time of train/val/test steps
    """

    def _on_batch_end(self, name, pl_module):
        self.timer.stop(name)

        def _log(pl_module, _current_fx_name):
            pl_module._current_fx_name = _current_fx_name
            pl_module.log(name, self.timer[name], on_step=True, on_epoch=False)

        import torch_xla.core.xla_model as xm

        xm.add_step_closure(_log, args=(pl_module, pl_module._current_fx_name))


def exp_manager(trainer: "pl.Trainer", cfg: Optional[Union[DictConfig, Dict]] = None) -> Optional[Path]:
    """
    exp_manager is a helper function used to manage folders for experiments. It follows the pytorch lightning paradigm
    of exp_dir/model_or_experiment_name/version. If the lightning trainer has a logger, exp_manager will get exp_dir,
    name, and version from the logger. Otherwise it will use the exp_dir and name arguments to create the logging
    directory. exp_manager also allows for explicit folder creation via explicit_log_dir.

    The version can be a datetime string or an integer. Datestime version can be disabled if use_datetime_version is set
    to False. It optionally creates TensorBoardLogger, WandBLogger, DLLogger, MLFlowLogger, ClearMLLogger,
    ModelCheckpoint objects from pytorch lightning.
    It copies sys.argv, and git information if available to the logging directory. It creates a log file for each
    process to log their output into.

    exp_manager additionally has a resume feature (resume_if_exists) which can be used to continuing training from
    the constructed log_dir. When you need to continue the training repeatedly (like on a cluster which you need
    multiple consecutive jobs), you need to avoid creating the version folders. Therefore from v1.0.0, when
    resume_if_exists is set to True, creating the version folders is ignored.

    Args:
        trainer (lightning.pytorch.Trainer): The lightning trainer.
        cfg (DictConfig, dict): Can have the following keys:

            - explicit_log_dir (str, Path): Can be used to override exp_dir/name/version folder creation. Defaults to
                None, which will use exp_dir, name, and version to construct the logging directory.
            - exp_dir (str, Path): The base directory to create the logging directory. Defaults to None, which logs to
                ./nemo_experiments.
            - name (str): The name of the experiment. Defaults to None which turns into "default" via name = name or
                "default".
            - version (str): The version of the experiment. Defaults to None which uses either a datetime string or
                lightning's TensorboardLogger system of using version_{int}.
            - use_datetime_version (bool): Whether to use a datetime string for version. Defaults to True.
            - resume_if_exists (bool): Whether this experiment is resuming from a previous run. If True, it sets
                trainer._checkpoint_connector._ckpt_path so that the trainer should auto-resume. exp_manager will move files
                under log_dir to log_dir/run_{int}. Defaults to False. From v1.0.0, when resume_if_exists is True,
                we would not create version folders to make it easier to find the log folder for next runs.
            - resume_past_end (bool): exp_manager errors out if resume_if_exists is True and a checkpoint matching
                ``*end.ckpt`` indicating a previous training run fully completed. This behaviour can be disabled, in which
                case the ``*end.ckpt`` will be loaded by setting resume_past_end to True. Defaults to False.
            - resume_ignore_no_checkpoint (bool): exp_manager errors out if resume_if_exists is True and no checkpoint
                could be found. This behaviour can be disabled, in which case exp_manager will print a message and
                continue without restoring, by setting resume_ignore_no_checkpoint to True. Defaults to False.
            - resume_from_checkpoint (str): Can be used to specify a path to a specific checkpoint file to load from. This will
                override any checkpoint found when resume_if_exists is True. Defaults to None.
            - create_tensorboard_logger (bool): Whether to create a tensorboard logger and attach it to the pytorch
                lightning trainer. Defaults to True.
            - summary_writer_kwargs (dict): A dictionary of kwargs that can be passed to lightning's TensorboardLogger
                class. Note that log_dir is passed by exp_manager and cannot exist in this dict. Defaults to None.
            - create_wandb_logger (bool): Whether to create a Weights and Baises logger and attach it to the pytorch
                lightning trainer. Defaults to False.
            - wandb_logger_kwargs (dict): A dictionary of kwargs that can be passed to lightning's WandBLogger
                class. Note that name and project are required parameters if create_wandb_logger is True.
                Defaults to None.
            - create_mlflow_logger (bool): Whether to create an MLFlow logger and attach it to the pytorch lightning
                training. Defaults to False
            - mlflow_logger_kwargs (dict): optional parameters for the MLFlow logger
            - create_dllogger_logger (bool): Whether to create an DLLogger logger and attach it to the pytorch lightning
                training. Defaults to False
            - dllogger_logger_kwargs (dict): optional parameters for the DLLogger logger
            - create_clearml_logger (bool): Whether to create an ClearML logger and attach it to the pytorch lightning
                training. Defaults to False
            - clearml_logger_kwargs (dict): optional parameters for the ClearML logger
            - create_checkpoint_callback (bool): Whether to create a ModelCheckpoint callback and attach it to the
                pytorch lightning trainer. The ModelCheckpoint saves the top 3 models with the best "val_loss", the most
                recent checkpoint under ``*last.ckpt``, and the final checkpoint after training completes under ``*end.ckpt``.
                Defaults to True.
            - create_early_stopping_callback (bool): Flag to decide if early stopping should be used to stop training. Default is False.
                See EarlyStoppingParams dataclass above.
            - create_preemption_callback (bool): Flag to decide whether to enable preemption callback to save checkpoints and exit training
                immediately upon preemption. Default is True.
            - create_straggler_detection_callback (bool): Use straggler detection callback. Default is False.
            - create_fault_tolerance_callback (bool): Use fault tolerance callback. Default is False.
            - files_to_copy (list): A list of files to copy to the experiment logging directory. Defaults to None which
                copies no files.
            - log_local_rank_0_only (bool): Whether to only create log files for local rank 0. Defaults to False.
                Set this to True if you are using DDP with many GPUs and do not want many log files in your exp dir.
            - log_global_rank_0_only (bool): Whether to only create log files for global rank 0. Defaults to False.
                Set this to True if you are using DDP with many GPUs and do not want many log files in your exp dir.
            - max_time (str): The maximum wall clock time *per run*. This is intended to be used on clusters where you want
                a checkpoint to be saved after this specified time and be able to resume from that checkpoint. Defaults to None.
            - seconds_to_sleep (float): seconds to sleep non rank 0 processes for. Used to give enough time for rank 0 to initialize
            - train_time_interval (timedelta): pass an object of timedelta to save the model every timedelta. Defaults to None.
                (use _target_ with hydra to achieve this)

    returns:
        log_dir (Path): The final logging directory where logging files are saved. Usually the concatenation of
            exp_dir, name, and version.
    """

    # Add rank information to logger
    # Note: trainer.global_rank and trainer.is_global_zero are not set until trainer.fit, so have to hack around it
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    global_rank = trainer.node_rank * trainer.num_devices + local_rank
    logging.rank = global_rank

    if cfg is None:
        logging.error("exp_manager did not receive a cfg argument. It will be disabled.")
        return
    if trainer.fast_dev_run:
        logging.info("Trainer was called with fast_dev_run. exp_manager will return without any functionality.")
        return

    # Ensure passed cfg is compliant with ExpManagerConfig
    schema = OmegaConf.structured(ExpManagerConfig)
    if isinstance(cfg, dict):
        cfg = OmegaConf.create(cfg)
    elif not isinstance(cfg, DictConfig):
        raise ValueError(f"cfg was type: {type(cfg)}. Expected either a dict or a DictConfig")
    cfg = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
    cfg = OmegaConf.merge(schema, cfg)

    nemo_exp_manager.error_checks(
        trainer, cfg
    )  # Ensures that trainer options are compliant with NeMo and exp_manager arguments

    log_dir, exp_dir, name, version = nemo_exp_manager.get_log_dir(
        trainer=trainer,
        exp_dir=cfg.exp_dir,
        name=cfg.name,
        version=cfg.version,
        explicit_log_dir=cfg.explicit_log_dir,
        use_datetime_version=cfg.use_datetime_version,
        resume_if_exists=cfg.resume_if_exists,
    )

    if cfg.resume_if_exists:
        # Check for existing checkpoints in `dirpath` if it's specified, use <log_dir>/checkpoints otherwise
        if cfg.checkpoint_callback_params.dirpath:
            check_resume(
                trainer,
                log_dir,
                cfg.resume_past_end,
                cfg.resume_ignore_no_checkpoint,
                cfg.checkpoint_callback_params.dirpath,
            )
        else:
            check_resume(trainer, log_dir, cfg.resume_past_end, cfg.resume_ignore_no_checkpoint)

    checkpoint_name = name
    # If name returned from get_log_dir is "", use cfg.name for checkpointing
    if checkpoint_name is None or checkpoint_name == "":
        checkpoint_name = cfg.name or "default"

    # Set mlflow name if it's not set, before the main name is erased
    if cfg.create_mlflow_logger and (not cfg.mlflow_logger_kwargs.get("experiment_name", None)):
        cfg.mlflow_logger_kwargs.experiment_name = cfg.name
        logging.warning(
            "mlflow logger specified but no experiment name set. Using the same as Tensorboard: %s",
            cfg.mlflow_logger_kwargs.experiment_name,
        )

    cfg.name = name  # Used for configure_loggers so that the log_dir is properly set even if name is ""
    cfg.version = version

    # update app_state with log_dir, exp_dir, etc
    app_state = AppState()
    app_state.log_dir = log_dir
    app_state.exp_dir = exp_dir
    app_state.name = name
    app_state.version = version
    app_state.checkpoint_name = checkpoint_name
    app_state.create_checkpoint_callback = cfg.create_checkpoint_callback
    app_state.checkpoint_callback_params = cfg.checkpoint_callback_params

    # Create the logging directory if it does not exist
    os.makedirs(log_dir, exist_ok=True)  # Cannot limit creation to global zero as all ranks write to own log file
    logging.info(f"Experiments will be logged at {log_dir}")
    trainer._default_root_dir = log_dir

    if cfg.log_local_rank_0_only is True and cfg.log_global_rank_0_only is True:
        raise ValueError(
            "Cannot set both log_local_rank_0_only and log_global_rank_0_only to True. Please set either one or neither."
        )

    # This is set if the env var NEMO_TESTING is set to True.
    nemo_testing = get_envbool(NEMO_ENV_VARNAME_TESTING, False)

    # Handle logging to file
    log_file = log_dir / f"nemo_log_globalrank-{global_rank}_localrank-{local_rank}.txt"
    if cfg.log_local_rank_0_only is True and not nemo_testing:
        if local_rank == 0:
            logging.add_file_handler(log_file)
    elif cfg.log_global_rank_0_only is True and not nemo_testing:
        if global_rank == 0:
            logging.add_file_handler(log_file)
    else:
        # Logs on all ranks.
        logging.add_file_handler(log_file)

    # For some reason, LearningRateLogger requires trainer to have a logger. Safer to create logger on all ranks
    # not just global rank 0.
    if cfg.create_tensorboard_logger or cfg.create_wandb_logger or cfg.create_mlflow_logger:
        nemo_exp_manager.configure_loggers(
            trainer=trainer,
            exp_dir=exp_dir,
            log_dir=log_dir,
            name=cfg.name,
            version=cfg.version,
            checkpoint_callback_params=app_state.checkpoint_callback_params,
            create_tensorboard_logger=cfg.create_tensorboard_logger,
            summary_writer_kwargs=cfg.summary_writer_kwargs,
            create_wandb_logger=cfg.create_wandb_logger,
            wandb_kwargs=cfg.wandb_logger_kwargs,
            create_mlflow_logger=cfg.create_mlflow_logger,
            mlflow_kwargs=cfg.mlflow_logger_kwargs,
            create_dllogger_logger=False,
            dllogger_kwargs=None,
            create_clearml_logger=False,
            clearml_kwargs=None,
            create_neptune_logger=False,
            neptune_kwargs=None,
        )

    # add loggers timing callbacks
    if cfg.log_step_timing:
        timing_callback = TimingCallback(timer_kwargs=cfg.step_timing_kwargs or {})
        trainer.callbacks.insert(0, timing_callback)

    if cfg.ema.enable:
        ema_callback = EMA(
            decay=cfg.ema.decay,
            apply_ema_every_n_steps=cfg.ema.apply_ema_every_n_steps,
            start_step=cfg.ema.start_step,
            evaluate_ema_weights_instead=cfg.ema.evaluate_ema_weights_instead,
        )
        trainer.callbacks.append(ema_callback)

    if cfg.create_checkpoint_callback:
        configure_checkpointing(trainer, log_dir, checkpoint_name, cfg.resume_if_exists, cfg.checkpoint_callback_params)

    if cfg.disable_validation_on_resume:
        # extend training loop to skip initial validation when resuming from checkpoint
        nemo_exp_manager.configure_no_restart_validation_training_loop(trainer)

    if is_global_rank_zero():
        # Move files_to_copy to folder and add git information if present
        if cfg.files_to_copy:
            for _file in cfg.files_to_copy:
                copy(Path(_file), log_dir)

        # Create files for cmd args and git info
        with open(log_dir / "cmd-args.log", "w", encoding="utf-8") as _file:
            _file.write(" ".join(sys.argv))

        # Add err_file logging to global_rank zero
        logging.add_err_file_handler(log_dir / "nemo_error_log.txt")

        # Add lightning file logging to global_rank zero
        add_filehandlers_to_pl_logger(log_dir / "lightning_logs.txt", log_dir / "nemo_error_log.txt")

    return log_dir


def check_resume(
    trainer: "pl.Trainer",
    log_dir: str,
    resume_past_end: bool = False,
    resume_ignore_no_checkpoint: bool = False,
    dirpath: str = None,
):
    """Checks that resume=True was used correctly with the arguments pass to exp_manager. Sets
    trainer._checkpoint_connector.resume_from_checkpoint_fit_path as necessary.

    Returns:
        log_dir (Path): The log_dir
        exp_dir (str): The base exp_dir without name nor version
        name (str): The name of the experiment
        version (str): The version of the experiment

    Raises:
        NotFoundError: If resume is True, resume_ignore_no_checkpoint is False, and checkpoints could not be found.
        ValueError: If resume is True, and there were more than 1 checkpoint could found.
    """

    if not log_dir:
        raise ValueError(f"Resuming requires the log_dir {log_dir} to be passed to exp_manager")

    # Use <log_dir>/checkpoints/ unless `dirpath` is set
    checkpoint_dir = Path(dirpath) if dirpath else Path(Path(log_dir) / "checkpoints")

    if not checkpoint_dir.exists():
        if resume_ignore_no_checkpoint:
            logging.warning(
                f"There was no checkpoint folder at checkpoint_dir :{checkpoint_dir}. Training from scratch."
            )
            return
        else:
            raise nemo_exp_manager.NotFoundError(
                f"There was no checkpoint folder at checkpoint_dir :{checkpoint_dir}. Cannot resume."
            )
    all_checkpoints = list(checkpoint_dir.rglob("*.ckpt"))
    if len(all_checkpoints) == 0:
        if resume_ignore_no_checkpoint:
            logging.warning(f"There were no checkpoints found in {checkpoint_dir}. Training from scratch.")
            return
        else:
            raise FileNotFoundError(f"There were no checkpoints found in {checkpoint_dir}. Cannot resume.")

    all_checkpoints.sort(key=os.path.getmtime, reverse=True)
    checkpoint = all_checkpoints[0]
    if "mp_rank" in str(checkpoint) or "tp_rank" in str(checkpoint):
        checkpoint = uninject_model_parallel_rank(checkpoint)

    logging.info(f"Resuming from {checkpoint}")

    trainer.ckpt_path = str(checkpoint)

    if is_global_rank_zero():
        # Check to see if any files exist that need to be moved
        files_to_move = []
        for child in Path(log_dir).iterdir():
            if child.is_file():
                files_to_move.append(child)

        if len(files_to_move) > 0:
            # Move old files to a new folder
            other_run_dirs = Path(log_dir).glob("run_*")
            run_count = 0
            for fold in other_run_dirs:
                if fold.is_dir():
                    run_count += 1
            new_run_dir = Path(Path(log_dir) / f"run_{run_count}")
            new_run_dir.mkdir()
            for _file in files_to_move:
                move(str(_file), str(new_run_dir))


class NeMoModelCheckpoint(nemo_exp_manager.NeMoModelCheckpoint):
    """Light wrapper around Lightning's ModelCheckpoint to force a saved checkpoint on train_end"""

    def __init__(
        self,
        always_save_nemo: bool = False,
        save_nemo_on_train_end: bool = True,
        save_best_model: bool = False,
        postfix: str = ".nemo",
        n_resume: bool = False,
        model_parallel_size: int = None,
        use_master_weights_in_ckpt: bool = False,
        async_save: bool = False,
        save_last_n_optim_states: int = -1,
        enable_version_counter: bool = False,
        **kwargs,
    ):
        # Parse and store "extended" parameters: save_best model and postfix.
        self.always_save_nemo = always_save_nemo
        self.save_nemo_on_train_end = save_nemo_on_train_end
        self.save_best_model = save_best_model
        if self.save_best_model and not self.save_nemo_on_train_end:
            logging.warning(
                (
                    "Found save_best_model is True and save_nemo_on_train_end is False. "
                    "Set save_nemo_on_train_end to True to automatically save the best model."
                )
            )
        self.postfix = postfix
        self.previous_best_path = ""
        self.model_parallel_size = model_parallel_size
        self.async_save = async_save
        self.save_last_n_optim_states = save_last_n_optim_states
        self.use_master_weights_in_ckpt = use_master_weights_in_ckpt

        # `prefix` is deprecated
        if "prefix" in kwargs:
            self.prefix = kwargs.pop("prefix")
        else:
            self.prefix = ""

        # Solve type mismatch between NeMo and PTL: https://github.com/NVIDIA/NeMo/pull/6108#issuecomment-1448998342
        train_time_interval = kwargs.pop("train_time_interval", None)
        if train_time_interval is not None:
            train_time_interval = timedelta(seconds=train_time_interval)

        # Call the parent class constructor with the remaining kwargs.
        super(nemo_exp_manager.NeMoModelCheckpoint, self).__init__(train_time_interval=train_time_interval, **kwargs)
        self._enable_version_counter = enable_version_counter

        if self.save_top_k != -1 and n_resume:
            logging.debug("Checking previous runs")
            self.nemo_topk_check_previous_run()

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        """Save checkpoint on train batch end if we meet the criteria for `every_n_train_steps`"""
        if self._should_skip_saving_checkpoint(trainer):
            return
        skip_batch = self._every_n_train_steps < 1 or (trainer.global_step % self._every_n_train_steps != 0)

        train_time_interval = self._train_time_interval
        skip_time = True
        now = time.monotonic()
        if train_time_interval:
            prev_time_check = self._last_time_checked
            if isinstance(prev_time_check, float):
                decision = True if (now - prev_time_check) < train_time_interval.total_seconds() else False
                # in case we have time differences across ranks
                # have all ranks agree to avoid possible hangs
                decision = trainer.strategy.reduce_boolean_decision(decision)
            skip_time = prev_time_check is None or decision
        if skip_batch and skip_time:
            return
        if not skip_time:
            self._last_time_checked = now

        monitor_candidates = self._monitor_candidates(trainer)
        self._save_topk_checkpoint(trainer, monitor_candidates)
        self._save_last_checkpoint(trainer, monitor_candidates)

    def on_train_end(self, trainer, pl_module):
        if trainer.fast_dev_run:
            return None

        should_save_last_checkpoint = False
        if not os.environ.get("NEURON_EXTRACT_GRAPHS_ONLY"):
            should_save_last_checkpoint = True
        if should_save_last_checkpoint:
            monitor_candidates = self._monitor_candidates(trainer)
            super()._save_last_checkpoint(trainer, monitor_candidates)
        # Call parent on_train_end() to save the -last checkpoint
        super().on_train_end(trainer, pl_module)

        # Load the best model and then re-save it
        if self.save_best_model:
            # wait for all processes
            trainer.strategy.barrier("SaveBestCheckpointConnector.resume_end")
            if self.best_model_path == "":
                logging.warning(
                    f"{self} was told to save the best checkpoint at the end of training, but no saved checkpoints "
                    "were found. Saving latest model instead."
                )
            else:
                self.best_model_path = trainer.strategy.broadcast(self.best_model_path)
                trainer._checkpoint_connector.restore(self.best_model_path)

        if self.save_nemo_on_train_end:
            pl_module.save_to(save_path=os.path.join(self.dirpath, self.prefix + self.postfix))


def configure_checkpointing(
    trainer: "pl.Trainer",
    log_dir: Path,
    name: str,
    resume: bool,
    params: "DictConfig",
):
    """Adds ModelCheckpoint to trainer. Raises CheckpointMisconfigurationError if trainer already has a ModelCheckpoint
    callback
    """
    for callback in trainer.callbacks:
        if isinstance(callback, ModelCheckpoint):
            raise nemo_exp_manager.CheckpointMisconfigurationError(
                "The pytorch lightning trainer that was passed to exp_manager contained a ModelCheckpoint "
                "and create_checkpoint_callback was set to True. Please either set create_checkpoint_callback "
                "to False, or remove ModelCheckpoint from the lightning trainer"
            )
    # Create the callback and attach it to trainer
    if "filepath" in params:
        if params.filepath is not None:
            logging.warning("filepath is deprecated. Please switch to dirpath and filename instead")
            if params.dirpath is None:
                params.dirpath = Path(params.filepath).parent
            if params.filename is None:
                params.filename = Path(params.filepath).name
        with open_dict(params):
            del params["filepath"]
    if params.dirpath is None:
        params.dirpath = Path(log_dir / "checkpoints")
    if params.filename is None:
        params.filename = f"{name}--{{{params.monitor}:.4f}}-{{epoch}}"
    if params.prefix is None:
        params.prefix = name
    NeMoModelCheckpoint.CHECKPOINT_NAME_LAST = params.filename + "-last"

    logging.debug(params.dirpath)
    logging.debug(params.filename)
    logging.debug(params.prefix)

    if "val" in params.monitor:
        if (
            trainer.max_epochs is not None
            and trainer.max_epochs != -1
            and trainer.max_epochs < trainer.check_val_every_n_epoch
        ):
            logging.error(
                "The checkpoint callback was told to monitor a validation value but trainer.max_epochs("
                f"{trainer.max_epochs}) was less than trainer.check_val_every_n_epoch({trainer.check_val_every_n_epoch}"
                f"). It is very likely this run will fail with ModelCheckpoint(monitor='{params.monitor}') not found "
                "in the returned metrics. Please ensure that validation is run within trainer.max_epochs."
            )
        elif trainer.max_steps is not None:
            logging.warning(
                "The checkpoint callback was told to monitor a validation value and trainer's max_steps was set to "
                f"{trainer.max_steps}. Please ensure that max_steps will run for at least "
                f"{trainer.check_val_every_n_epoch} epochs to ensure that checkpointing will not error out."
            )

    checkpoint_callback = NeMoModelCheckpoint(n_resume=resume, **params)
    checkpoint_callback.last_model_path = trainer.ckpt_path or ""
    if "mp_rank" in checkpoint_callback.last_model_path or "tp_rank" in checkpoint_callback.last_model_path:
        checkpoint_callback.last_model_path = uninject_model_parallel_rank(checkpoint_callback.last_model_path)
    trainer.callbacks.append(checkpoint_callback)
