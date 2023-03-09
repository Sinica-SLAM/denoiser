#!/bin/bash

RUN_DIR="utputs/static_nyenhtt4"
DSET=PLACEHOLDER


python3 train_nytt.py \
  hydra.run.dir=$RUN_DIR \
  epochs=500 \
  dset=$DSET \
  demucs.causal=1 \
  demucs.hidden=48 \
  bandmask=0.2 \
  demucs.resample=4 \
  remix=1 \
  shift=8000 \
  shift_same=True \
  stft_loss=False \
  stft_sc_factor=0.1 stft_mag_factor=0.1 \
  loss=l1 \
  segment=4.5 \
  stride=0.5 \
  ddp=4 $@
