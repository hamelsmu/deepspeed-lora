#!/bin/bash

set +e

for ds_stg in 0 2 3; do
  for n_gpu in 1 2 3; do
    for gc in True False; do
      for seq_len in 64 512 2048; do
        for model_sz in 7 13 34; do

          # Special exceptions
          if [ "$ds_stg" -gt 0 ] && [ "$n_gpu" -eq 1 ]; then continue; fi

          bs_list=()
          if [ "$seq_len" -eq 64 ]; then bs_list=(1 8 64 200 400)
          elif [ "$seq_len" -eq 512 ]; then bs_list=(1 4 8 32 64)
          elif [ "$seq_len" -eq 2048 ]; then bs_list=(1 4 8 16)
          fi

          for bs in "${bs_list[@]}"; do
            echo "Running with ds_stg=$ds_stg n_gpu=$n_gpu gc=$gc seq_len=$seq_len bs=$bs model_sz=$model_sz"
            python run_bench.py --ds_stg $ds_stg --n_gpu $n_gpu --gc $gc --seq_len $seq_len --bs $bs --model_sz $model_sz --qaunt4 True
            # Check if the last command (run_bench.py) was successful
            if [ $? -ne 0 ]; then
              echo "Error encountered for bs=$bs. Skipping higher batch sizes."
              break  # Break out of the bs loop
            fi
            python sync_mem.py
          done
        done
      done
    done
  done
done
