#!/usr/bin/env bash
set -euo pipefail
SIF=/scratch/u6g/zhouyihe.u6g/sglang.sif
apptainer exec --nv -B /scratch/u6g/zhouyihe.u6g:/mnt/home "$SIF" bash -lc 
