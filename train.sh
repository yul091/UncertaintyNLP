#!/bin/bash

# EPOCHS=20
# TRAIN_BATCH=32
# EVAL_BATCH=128
# LR=0.005
# COEFF=0.1 # metric loss weight
# # COEFF=0 # metric loss weight
# CROSS=1.0 # ensemble loss weight
# # CROSS=0 # ensemble loss weight
# ALPHA=0.5 # margin between positive and negative pairs
# EPSILON=1.0 # radius of the sphare for generating transformed inputs
# LAYERS=2 # number of layers of RNN model
# HIDDEN=256 # hidden size of RNN model
# DROPOUT=0.2 # dropout rate of RNN model
# DATASET='20news'  # dataset name
# # MODEL='bert'
# MODEL='rnn'
# N_SAMPLES=200 # number of samples to generate transformed test inputs

# CUDA_VISIBLE_DEVICES=1 python main.py \
# --epochs=$EPOCHS --lr=$LR \
# --n_samples=$N_SAMPLES --alpha=$ALPHA --epsilon=$EPSILON \
# --train_batch_size=$TRAIN_BATCH \
# --eval_batch_size=$EVAL_BATCH \
# --dataset=$DATASET --model=$MODEL \
# --num_layers=$LAYERS --hidden_size=$HIDDEN --dropout=$DROPOUT \
# --coeff=$COEFF --cross_rate=$CROSS \
# --ensemble 
# # --from_scratch



EPOCHS=20
TRAIN_BATCH=32
EVAL_BATCH=128
# LR=0.003
LR=5e-5
# COEFF=0.1 # metric loss weight
COEFF=0 # metric loss weight
# CROSS=0.1 # ensemble loss weight
CROSS=0 # ensemble loss weight
ALPHA=0.5 # margin between positive and negative pairs
EPSILON=1.0 # radius of the sphare for generating transformed inputs
LAYERS=2 # number of layers of RNN model
HIDDEN=256 # hidden size of RNN model
DROPOUT=0.2 # dropout rate of RNN model
DATASET='20news'  # dataset name
MODEL='bert'
N_SAMPLES=200 # number of samples to generate transformed test inputs

CUDA_VISIBLE_DEVICES=0 python main.py \
--epochs=$EPOCHS --lr=$LR \
--n_samples=$N_SAMPLES --alpha=$ALPHA --epsilon=$EPSILON \
--train_batch_size=$TRAIN_BATCH \
--eval_batch_size=$EVAL_BATCH \
--dataset=$DATASET --model=$MODEL \
--num_layers=$LAYERS --hidden_size=$HIDDEN --dropout=$DROPOUT \
--coeff=$COEFF --cross_rate=$CROSS 
# --from_scratch
# --ensemble \ 
