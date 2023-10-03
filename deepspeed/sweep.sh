#!/bin/bash

# Continue even if a command fails
set +e

# Loop through each option
for ds_stg in 0 3; do
  for n_gpu in 1 2 3; do
    for gc in True False; do
      for seq_len in 64 256 512 1024 2048; do

        # Conditional batch sizes based on sequence length
        if [ $seq_len -le 256 ]; then
          bs_list=(1 4 8 16 32 64 100 128 200 256)
        else
          bs_list=(1 3 4 6 8 16 32)
        fi

        for bs in "${bs_list[@]}"; do
          # Execute the command
          echo "Running with ds_stg=$ds_stg n_gpu=$n_gpu gc=$gc seq_len=$seq_len bs=$bs"
          python run_bench.py --ds_stg $ds_stg --n_gpu $n_gpu --gc $gc --seq_len $seq_len --bs $bs
          # sync gpu mem usage stats to wandb runs
          python sync_mem.py
        done
      done
    done
  done
done
