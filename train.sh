#!/usr/bin/env sh
LOG=./log-'data +%Y-%m-%d-%H-%S'.log
SOLVER=./solver.prototxt
caffe train --solver=$SOLVER 2>&1 | tee $LOG

