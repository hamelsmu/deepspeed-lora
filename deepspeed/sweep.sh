#!/bin/bash

set +e

for ds_stg in 0 2 3; do
  for n_gpu in 1 2 3; do
    for gc in True False; do
      for seq_len in 64 512 2048; do
        for model_sz in 7 13 34; do

          # Handle large model
          if [ "$model_sz" -eq 34 ]; then
            if [ "$ds_stg" -lt 2 ] || [ "$n_gpu" -lt 2 ] || [ "$seq_len" -ge 2048 ]; then
              continue
            fi
            if [ "$gc" = "False" ] && ([ "$n_gpu" -ne 3 ] || [ "$ds_stg" -lt 3 ]); then
              continue
            fi
          fi

          # Special exceptions
          if [ "$ds_stg" -gt 0 ] && [ "$n_gpu" -eq 1 ]; then continue; fi
          if [ "$gc" = "False" ] && [ "$n_gpu" -eq 1 ] && [ "$model_sz" -ne 7 ]; then continue; fi

          bs_list=()
          if [ "$seq_len" -eq 64 ]; then bs_list=(1 8 64 200)
          elif [ "$seq_len" -eq 512 ]; then bs_list=(1 4 8 32)
          elif [ "$seq_len" -eq 2048 ]; then bs_list=(1 4 8)
          fi

          for bs in "${bs_list[@]}"; do
            if [ "$bs" -ge 32 ] && ([ "$model_sz" -eq 34 ] || [ "$seq_len" -eq 2048 ]); then continue; fi
            echo "Running with ds_stg=$ds_stg n_gpu=$n_gpu gc=$gc seq_len=$seq_len bs=$bs model_sz=$model_sz"
            python run_bench.py --ds_stg $ds_stg --n_gpu $n_gpu --gc $gc --seq_len $seq_len --bs $bs --model_sz $model_sz
            python sync_mem.py
          done
        done
      done
    done
  done
done
