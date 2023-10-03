#!/bin/bash

# Continue even if a command fails
set +e

# Loop through each option
for ds_stg in 0 2 3; do
  for n_gpu in 1 2 3; do
    for gc in True False; do
      for seq_len in 64 512 2048; do
        for model_sz in 7 13 34; do
          # Conditional batch sizes based on sequence length
          if [ $seq_len -e 64 ]; then
            bs_list=(1 8 64 200)
          elif [ $seq_len -e 512 ]; then
            bs_list=(1 4 8 32)
          elif [ $seq_len -e 2048 ]; then
            bs_list=(1 4 8)
          fi
          for bs in "${bs_list[@]}"; do
            # Execute the command
            echo "Running with ds_stg=$ds_stg n_gpu=$n_gpu gc=$gc seq_len=$seq_len bs=$bs model_sz=$model_sz"
            python run_bench.py --ds_stg $ds_stg --n_gpu $n_gpu --gc $gc --seq_len $seq_len --bs $bs --model_sz $model_sz
            # sync peak gpu mem during run with wandb
            python sync_mem.py
          done
        done
      done
    done
  done
done
