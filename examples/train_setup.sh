#!/usr/bin/env bash

set -o pipefail
set -e

ulimit -n 65535

if [ -v SLURM_NNODES ]
then
    # SLURM runs
    export FI_EFA_USE_DEVICE_RDMA=1
    export FI_PROVIDER=efa
    export FI_EFA_FORK_SAFE=1
    sudo sysctl -w net.ipv4.ip_local_reserved_ports=41000
    if which lctl >/dev/null 2>&1; then
        sudo lctl set_param 'osc.*.max_dirty_mb=64' # Cap max space each connection to FSx reserves so we avoid OODs
    fi
    IPS=""
    for h in $(scontrol show hostname); do
        IPS="$IPS $(nslookup $h  | awk '/^Address: / { print $2 }')";
    done
    HOSTS=(${IPS//\ / })
    NODEID=$SLURM_NODEID
    NTASKS=$SLURM_NTASKS

    export NEMO_EXPM_VERSION=$SLURM_JOB_ID
    export EXPLICIT_LOGDIR=null
    : ${SLURM_RESTART_COUNT:=0}
    LOG_PATH=logs/$SLURM_JOB_ID/$SLURM_RESTART_COUNT/$NODEID/
    mkdir -p $LOG_PATH
    # Make sure to install latest runtime
    # ./setup.sh   2>&1  | tee  $LOG_PATH/setup.log
elif [ -v OMPI_COMM_WORLD_RANK ]
then
    # MPI runs on EKS
    export CCOM_SOCKET_IFNAME=eth0
    NODELIST=`/nodelist_helper.py`
    HOSTS=(${NODELIST//\ / })
    NODEID=$OMPI_COMM_WORLD_RANK
    NTASKS=$OMPI_COMM_WORLD_SIZE
    export EXPLICIT_LOGDIR=/shared/nemo_experiments/$POD_UID
    LOG_PATH=$EXPLICIT_LOGDIR/$NODEID/
    mkdir -p $LOG_PATH
    export FI_EFA_USE_DEVICE_RDMA=1
    export FI_PROVIDER=efa
    export FI_EFA_FORK_SAFE=1
else
    # Single-node, non-SLURM, non-MPI runs
    HOSTS=(localhost)
    NODEID=0
    NTASKS=1
    export NEMO_EXPM_VERSION=$(date "+%Y-%m-%d_%H-%M-%S")
    export EXPLICIT_LOGDIR=null
    LOG_PATH=./nemo_experiments/logs
    mkdir -p $LOG_PATH
fi

# Note: This is a temp fix till we handle functionalization
# In PT2.1, functionalization is needed to close 3% convergence gap compared to PT1.13 for ZeRO1
export XLA_DISABLE_FUNCTIONALIZATION=0

export HYDRA_FULL_ERROR=1
export MASTER_ADDR=${HOSTS[0]}
export MASTER_PORT=41000
export MALLOC_ARENA_MAX=128
export CREATE_TB_LOGGER=True
export CHECKPOINT_CALLBACK=True