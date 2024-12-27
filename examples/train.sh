#!/usr/bin/env bash

source ./train_setup.sh

# For SFT: use the config file megatron_llama2_7B_SFT_config or megatron_llama3_8B_SFT_config
: ${CONF_FILE:=megatron_gpt_config} 

if [ "$COMPILE" = "1" ]; then
    echo "compiling only run"
    MAYBE_COMPILE="neuron_parallel_compile"
fi


$MAYBE_COMPILE torchrun $DISTRIBUTED_ARGS training_orchestrator.py  \
    --config-path=conf \
    --config-name=$CONF_FILE \
    trainer.devices=$PROCESSES_PER_NODE \
    trainer.num_nodes=$NTASKS \
    exp_manager.explicit_log_dir=$EXPLICIT_LOGDIR   2>&1  | tee -a $LOG_PATH/log