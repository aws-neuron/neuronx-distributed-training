import hydra
import os

def set_env_variable(env_var_name, value, append=False, overwrite=False):
    if append and os.environ.get(env_var_name, None):
        os.environ[env_var_name] = f"{os.environ[env_var_name]} {value}"
        return
    if overwrite:
        os.environ[env_var_name] = value
    elif os.environ.get(env_var_name, None) is None:
        os.environ[env_var_name] = value


def process_config(cfg):
    if cfg.model.fusions.get("softmax", None):
        set_env_variable("NEURON_FUSE_SOFTMAX", "1")
    if cfg.neuron_experimental_compress_rg is True:
        set_env_variable("NEURON_EXPERIMENTAL_COMPRESS_RG", "1")
    set_env_variable("NEURON_RT_ASYNC_EXEC_MAX_INFLIGHT_REQUESTS", str(cfg.aync_exec_max_inflight_requests))
    set_env_variable("BUCKET_CAP_MB", str(cfg.bucket_size_collectives))
    set_env_variable("NEURON_COMPILE_CACHE_URL", cfg.compiler_cache_url)
    set_env_variable("NEURON_CC_FLAGS", cfg.compiler_flags, append=True)
    set_env_variable("NEURON_RT_EXEC_TIMEOUT", str(cfg.neuron_rt_exec_timeout))

    # if doing compile then change the some config vars
    # elif training and have TRAIN_ITERS set (used in tests) set max_steps as TRAIN_ITERS, rest reamins as in '.yaml' file
    # else keep everything the same as in '.yaml' file
    # NOTE - Specifying TRAIN_ITERS as env variable is used for test purposes only, recommended approach is to update variable in yaml cfg/pass it to script
    if os.environ.get("COMPILE") == "1":
        cfg.trainer.max_steps = 4
        cfg.exp_manager.create_tensorboard_logger = False
        cfg.exp_manager.create_checkpoint_callback = False
    elif os.environ.get("COMPILE") == "0" and os.environ.get("TRAIN_ITERS") is not None:
        cfg.trainer.max_steps = int(os.environ.get("TRAIN_ITERS"))

    if cfg.precision.get("type") == "bf16SR":
        set_env_variable("XLA_USE_BF16", "1")
        set_env_variable("XLA_DOWNCAST_BF16", "0")
        set_env_variable("NEURON_RT_STOCHASTIC_ROUNDING_EN", "1")
        set_env_variable("NEURON_CC_FLAGS", "--enable-mixed-precision-accumulation", append=True)
    elif cfg.precision.get("type") == "mixed_precision":
        set_env_variable("XLA_USE_BF16", "0")
        set_env_variable("XLA_DOWNCAST_BF16", "1")
        set_env_variable("NEURON_RT_STOCHASTIC_ROUNDING_EN", "0")
        set_env_variable("NEURON_CC_FLAGS", "--enable-mixed-precision-accumulation", append=True)
    elif cfg.precision.get("type") == "mixed_precision_SR":
        set_env_variable("XLA_USE_BF16", "0")
        set_env_variable("XLA_DOWNCAST_BF16", "1")
        set_env_variable("NEURON_RT_STOCHASTIC_ROUNDING_EN", "1")
        set_env_variable("NEURON_CC_FLAGS", "--enable-mixed-precision-accumulation", append=True)  
    elif cfg.precision.get("type") == "fp32":
        set_env_variable("XLA_USE_BF16", "0")
        set_env_variable("XLA_DOWNCAST_BF16", "0")
        set_env_variable("NEURON_CC_FLAGS", "--auto-cast none", append=True) 
    elif cfg.precision.get("type") == "manual":
        set_env_variable("XLA_USE_BF16", cfg.precision.get("xla_use_bf16"))
        set_env_variable("XLA_DOWNCAST_BF16", cfg.precision.get("xla_downcast_bf16"))
        set_env_variable("NEURON_RT_STOCHASTIC_ROUNDING_EN", cfg.precision.get("neuron_rt_stochastic_rounding_en"))

@hydra.main(config_path="conf",config_name="megatron_gpt_config", version_base="1.2")
def main(cfg) -> None:
    process_config(cfg)
    from training import train
    train(cfg)

if __name__ == "__main__":
    main()
