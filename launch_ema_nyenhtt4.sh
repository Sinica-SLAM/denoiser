#!/bin/bash

RUN_DIR="outputs/ema_nyenhtt4"
TEACHER_CHECKPOINT=PLACEHOLDER # baseline or initial teacher model ckpt
SAVE_CHECKPOINT="${RUN_DIR}/checkpoints/checkpoint1.th"
S_CHECKPOINT="${RUN_DIR}/checkpoint.th"
NOISY_DIR=PLACEHOLDER # origin noisy speech folder path (in out case is valentini trainset)
ENHANCE_DIR=PLACEHOLDER # your training data folder path (related to dset)
DSET=PLACEHOLDER 

mkdir -p $RUN_DIR/checkpoints

for i in {1..35}
do
  python3 train_nytt.py \
    hydra.run.dir=$RUN_DIR \
    epochs=1 \
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

  python3 ./average_model.py $TEACHER_CHECKPOINT $S_CHECKPOINT $SAVE_CHECKPOINT -w 0.995
    
  python3 -m denoiser.enhance_noise --model_path=$SAVE_CHECKPOINT --noisy_dir=$NOISY_DIR --out_dir=$ENHANCE_DIR

  rm $ENHANCE_DIR/noise/*.wav
  rm $ENHANCE_DIR/clean/*.wav

  mv $ENHANCE_DIR/*_noise.wav $ENHANCE_DIR/noise
  mv $ENHANCE_DIR/*_enhanced.wav $ENHANCE_DIR/clean

  rename 's/_noise//' $ENHANCE_DIR/noise/*.wav
  rename 's/_enhanced//' $ENHANCE_DIR/clean/*.wav

  TEACHER_CHECKPOINT=$SAVE_CHECKPOINT
  SAVE_CHECKPOINT="${RUN_DIR}/checkpoints/checkpoint$((i+1)).th"
  rm $S_CHECKPOINT

done

