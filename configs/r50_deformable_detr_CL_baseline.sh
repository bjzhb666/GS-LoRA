#!/usr/bin/env bash

set -x

# EXP_DIR=exps/r50_deformable_detr
PY_ARGS=${@:1}

python -u main_cl_baseline.py \
    ${PY_ARGS}