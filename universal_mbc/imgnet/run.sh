#!/bin/bash

python trainer.py \
    --mode $MODE \
    --gpu $GPU \
    --clusters $CLUSTERS \
    --run $RUN \
    --slot-drop $SLOT_DROP \
    --model $MODEL \
    --n-parallel $N_PARALLEL \
    --universal $UNIVERSAL \
    --universal-k $UNIVERSAL_K \
    --heads $HEADS \
    --slot-type $SLOT_TYPE \
