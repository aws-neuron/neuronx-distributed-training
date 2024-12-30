#!/usr/bin/env bash

source ./train_setup.sh

: ${CONF_FILE:=megatron_gpt_config}

CONF_FILE_PATH="./conf/${CONF_FILE}.yaml"

if [ -f "$CONF_FILE_PATH" ]; then
    devices=$(grep 'devices:' "$CONF_FILE_PATH" | awk '{print $2}')
    export PROCESSES_PER_NODE=$devices
else
    echo "Error: YAML file '$CONF_FILE_PATH' not found!"
fi

export DISTRIBUTED_ARGS="--nproc_per_node $PROCESSES_PER_NODE --nnodes $NTASKS --node_rank $NODEID --master_addr $MASTER_ADDR --master_port $MASTER_PORT"
echo $DISTRIBUTED_ARGS

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