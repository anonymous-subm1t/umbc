#!/bin/bash

python trainer.py \
    --dataset $DATASET \
    --set-size $SET_SIZE \
    --slot-type $SLOT_TYPE \
    --mode $MODE \
    --gpu $GPU \
    --model $MODEL \
    --run $RUN \
    --universal $UNIVERSAL \
    --n-parallel $N_PARALLEL \
    --universal-k $UNIVERSAL_K \
    --heads $HEADS \
    --fixed $FIXED \
    --ln-after $LN_AFTER \
    --attn-act $ATTN_ACT \
    --slot-residual $SLOT_RESIDUAL \
    --slot-drop $SLOT_DROP
