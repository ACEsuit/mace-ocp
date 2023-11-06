#!/bin/sh
PATH=/checkpoint/abhshkdz/local/bin:/private/home/abhshkdz/.local/bin:/public/apps/anaconda3/2020.11/condabin:/public/apps/anaconda3/2020.11/bin:/public/apps/gcc/7.1.0/bin:/public/apps/cuda/11.6/bin:/usr/local/cuda/bin:/opt/bin:/usr/local/cuda/bin:/opt/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/usr/games:/usr/local/games:/snap/bin:/public/slurm/20.11.5/bin:/public/slurm/20.11.5/sbin

################################################################################

ROOT_DIR="/private/home/abhshkdz/projects/mace-ocp"

# name your run; it'll show up on wandb by this name
IDENTIFIER="mace_0127"

# yaml config
CONFIG_YML=configs/s2ef/2M/mace/base_0127.yml

# directory where slurm logs will get saved
LOG_DIR=/checkpoint/abhshkdz/ocp_oct1_logs

# conda env
ENV=/private/home/abhshkdz/.conda/envs/ocp-mace-2022dec20

# running locally?
LOCAL_RUN=false

# slurm?
PARTITION=ocp,learnaccel
NODES=2

# initializing from a checkpoint?
# CHECKPOINT_PATH=

MODE=train

# launching a sweep?
# SWEEP_YML=experimental/abhshkdz/gnoc_v2/configs/2M/101103_no_loss_high_fmax_sweep.yml

################################################################################

PYTHON=$ENV/bin/python

_setArgs(){
    while [ "$1" != "" ]; do
        case $1 in
            "--local")
                IDENTIFIER="debug"
                LOCAL_RUN=true
                ;;
        esac
        shift
    done
}

_setArgs $*

################################################################################

LAUNCH_ARGS="--config-yml $CONFIG_YML \
    --identifier $IDENTIFIER \
    --logdir $LOG_DIR \
    --seed 1 \
    --distributed \
    --mode $MODE"

# LAUNCH_ARGS="$LAUNCH_ARGS --amp"

# with pretrained checkpoint
if [ ${CHECKPOINT_PATH+x} ] ; then
    LAUNCH_ARGS="$LAUNCH_ARGS --checkpoint $CHECKPOINT_PATH"
fi

# with sweep params
if [ ${SWEEP_YML+x} ] ; then
    LAUNCH_ARGS="$LAUNCH_ARGS --sweep-yml $SWEEP_YML"
fi

if [ "$LOCAL_RUN" = true ] ; then
    # local run
    LAUNCH_ARGS="$LAUNCH_ARGS --debug"
    $PYTHON -u -m torch.distributed.launch --nproc_per_node=1 main.py $LAUNCH_ARGS
else
    # submit to slurm
    LAUNCH_ARGS="$LAUNCH_ARGS --submit \
        --slurm-mem 480 \
        --slurm-partition $PARTITION \
        --num-nodes $NODES \
        --num-gpus 8"
    $PYTHON $ROOT_DIR/main.py $LAUNCH_ARGS
fi
