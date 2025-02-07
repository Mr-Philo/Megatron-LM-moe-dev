#!/bin/bash

# Test for simple MoE structure under Megatron-LM

# this is for preventing NCCL error: missing such environment variable will cause training loss to be NaN
NCCL_LIBRARY=/usr/lib/x86_64-linux-gnu/libnccl.so
export LD_PRELOAD="/usr/local/lib/libmsamp_dist.so:${NCCL_LIBRARY}:${LD_PRELOAD}"

export CUDA_DEVICE_MAX_CONNECTIONS=2

# ------------------------------------------------------------------------------------------------------------
# User-defined variables

MICRO_BATCH_SIZE=4
# GRAD_ACCUMULATION_STEPS=8
GLOBAL_BATCH_SIZE=512
SEED=42

TENSOR_PARALLEL_SIZE=1
PIPELINE_PARALLEL_SIZE=1

# training tokens
# tokens: 1T(1e12)
SEQ_LEN=2048
# training_samples = train_tokens / seq_len = 1e12 / 2048 = 488281250
TRAIN_SAMPLES=488281250
# lr_decay_samples = train_samples * lr_decay_fraction (0.9) = 439453125
LR_DECAY_SAMPLES=439453125
# lr_warmup_samples = train_samples * lr_warmup_fraction (0.05) = 24414062
LR_WARMUP_SAMPLES=24414062


# Tokenizer path
HF_TOKENIZER_PATH="/mnt/ruizhe/hf_model/Meta-Llama-3-8B"

DATA_BASE_DIR="/mnt/ruizhe/dataset/dclm-baseline-top-100B-subset-recon-tokenized/"
DATA_CACHE_PATH="/mnt/ruizhe/dataset/dclm-baseline-top-100B-subset-recon-tokenized/index_cache/"
echo "DATA_BASE_DIR: $DATA_BASE_DIR"
echo "DATA_CACHE_PATH: $DATA_CACHE_PATH"

# wandb init
# WANDB_API_KEY="7d145ddb51919baf27f77f979e23035f6424ddc8"
# wandb login $WANDB_API_KEY

# multi-node environment variables
# ------------------------------------------------------------------------------------------------------------
# if AZ_BATCHAI_GPU_COUNT has been setted, then use it as GPU_PER_NODE_COUNT,
# as A100 use AZ_BATCHAI_GPU_COUNT, while H100 use GPU_PER_NODE_COUNT :)
if [ ! -z ${AZ_BATCHAI_GPU_COUNT+x} ]; then
    GPU_PER_NODE_COUNT=${AZ_BATCHAI_GPU_COUNT}
fi

# singularity
export GPUS_PER_NODE=${GPU_PER_NODE_COUNT:=1}
export NUM_NODES=${AZUREML_NODE_COUNT:=1}
export NODE_RANK=${NODE_RANK:=0}
export MASTER_ADDR=${MASTER_ADDR:=localhost}
export MASTER_PORT=${MASTER_PORT:=1828}
WORLD_SIZE=$(($GPUS_PER_NODE*$NUM_NODES))

# GLOBAL_BATCH_SIZE=$(($MICRO_BATCH_SIZE*$WORLD_SIZE*$GRAD_ACCUMULATION_STEPS))     # abandoned
echo "GPUS_PER_NODE: ${GPUS_PER_NODE}"
echo "MASTER_ADDR: ${MASTER_ADDR}"
echo "MASTER_PORT: ${MASTER_PORT}"
echo "NUM_NODES: ${NUM_NODES}"
echo "NODE_RANK: ${NODE_RANK}"
echo "WORLD_SIZE: ${WORLD_SIZE}"
echo "MICRO_BATCH_SIZE: ${MICRO_BATCH_SIZE}"
echo "GLOBAL_BATCH_SIZE: ${GLOBAL_BATCH_SIZE}"


# Distributed arguments
DISTRIBUTED_ARGS=(
    --nproc_per_node $GPUS_PER_NODE 
    --nnodes $NUM_NODES 
    --node_rank $NODE_RANK
    --master_addr $MASTER_ADDR 
    --master_port $MASTER_PORT
)

# GPT model arguments
GPT_MODEL_ARGS=(
    --num-layers 24
    --hidden-size 2048 
    --num-attention-heads 16 
    --seq-length $SEQ_LEN 
    --max-position-embeddings $SEQ_LEN 
    --transformer-impl "local"
)

# LLAMA model arguments
LLAMA_ARGS=(
    --normalization RMSNorm
    --use-rotary-position-embeddings
    --swiglu
    # --untie-embeddings-and-output-weights     # 如果enable，则不共享权重，则需要手动将lm_head设置为高精度
    --disable-bias-linear
    --attention-dropout 0
    --hidden-dropout 0
)

# Training arguments
TRAINING_ARGS=(
    --micro-batch-size $MICRO_BATCH_SIZE 
    --global-batch-size $GLOBAL_BATCH_SIZE
    # --rampup-batch-size 16 16 5859375
    #! iter-based training---------------------------------
    # --train-iters 500000
    # --lr-decay-iters 430000
    # --lr-warmup-fraction 0.01
    #! sample-based training-------------------------------
    --train-samples $TRAIN_SAMPLES
    --lr-decay-samples $LR_DECAY_SAMPLES
    --lr-warmup-samples $LR_WARMUP_SAMPLES
    #! -----------------------------------------------------
    --weight-decay 0.1 
    --adam-beta1 0.9 
    --adam-beta2 0.95 
    --init-method-std 0.006 
    --clip-grad 1.0 
    --bf16      # todo: fp16
    --use-flash-attn 
    --lr 6.0e-4
    --lr-decay-style cosine 
    --min-lr 6.0e-5
    --use-distributed-optimizer
    --no-gradient-accumulation-fusion
)

# Model parallel arguments
MODEL_PARALLEL_ARGS=(
    --tensor-model-parallel-size $TENSOR_PARALLEL_SIZE
    --pipeline-model-parallel-size $PIPELINE_PARALLEL_SIZE 
)

# Data arguments
DATA_ARGS=(
    #! GPT2 BPE Tokenizer -----------------------------------------------
    # --data-path $DATA_PATH 
    # --vocab-file $VOCAB_FILE 
    # --merge-file $MERGE_FILE
    # --data-impl mmap 
    #! Huggingface Tokenizer --------------------------------------------
    --data-path $DATA_BASE_DIR
    --use-dir-data-input
    --tokenizer-model $HF_TOKENIZER_PATH
    --tokenizer-type HuggingFaceTokenizer
    #! ------------------------------------------------------------------
    --split 949,50,1
    --seed $SEED
    --data-cache-path $DATA_CACHE_PATH
    --distributed-timeout-minutes 240
    --num-workers 16
    # --num-dataset-builder-threads 16      # this option is for higher version Megatron
    # --dataloader-type "cyclic"              # disable it will use default 'single' dataloader
)

# Evaluation and logging arguments
# 1 iteration = (global_batch_size * seq_len) tokens
EVAL_AND_LOGGING_ARGS=(
    --log-interval 10
    --save-interval 5000                       # local
    --eval-interval 500 
    # --save $CHECKPOINT_PATH                   # enable this line and next line for cluster training
    # --load $CHECKPOINT_PATH 
    --eval-iters 5
    --disable-valid-and-test                   # for alta cluster. valid or test will lead to NCCL timeout
    --wandb-project "megatron-fp4-running"
    --wandb-enable-system-info
)

# -------------------------------------------------------------------------------------------

CHECKPOINT_PATH="/mnt/ruizhe/checkpoints/fp4_llama/LLaMA_1B3_DCLM_scheduler_2T_bf16_bs512_lr6e-4"
LOG_PATH=$CHECKPOINT_PATH"/log_node"$NODE_RANK".log"        # distinguish different nodes
mkdir -p "$CHECKPOINT_PATH"
touch "$LOG_PATH"
torchrun ${DISTRIBUTED_ARGS[@]} pretrain_gpt.py \
    ${GPT_MODEL_ARGS[@]} \
    ${LLAMA_ARGS[@]} \
    ${TRAINING_ARGS[@]} \
    ${MODEL_PARALLEL_ARGS[@]} \
    ${DATA_ARGS[@]} \
    ${EVAL_AND_LOGGING_ARGS[@]} \
    --save $CHECKPOINT_PATH \
    --load $CHECKPOINT_PATH \
    2>&1 | tee $LOG_PATH

