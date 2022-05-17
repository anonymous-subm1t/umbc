#!/bin/bash

python trainer.py \
    --set-size $SET_SIZE \
    --slot-type $SLOT_TYPE \
    --mode $MODE \
    --gpu $GPU \
    --model $MODEL \
    --h-dim $H_DIM \
    --n-parallel $N_PARALLEL \
    --slot-drop $SLOT_DROP \
    --run $RUN \
    --universal $UNIVERSAL \
    --universal-k $UNIVERSAL_K \
    --heads $HEADS \
