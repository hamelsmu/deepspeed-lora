#!/bin/bash

# Continue even if a command fails
set +e

for ds_stg in 0 2 3; do
  for n_gpu in 1 2 3; do
    for gc in True False; do
      for seq_len in 64 512 2048; do
        for model_sz in 7 13 34; do
          if [ "$model_sz" -eq 34 ] && [ "$ds_stg" -le 2 ]; then continue; fi
          if [ "$n_gpu" -eq 1 ] && [ "$ds_stg" -gt 0 ]; then continue; fi
          if [ "$model_sz" -eq 34 ] && [ "$n_gpu" -lt 2 ]; then continue; fi
          if [ "$model_sz" -eq 34 ] && [ "$seq_len" -ge 2048 ]; then continue; fi
          if [ "$gc" = "False" ] && [ "$n_gpu" -eq 1 ] && [ "$model_sz" -ne 7 ]; then continue; fi
          if [ "$gc" = "False" ] && [ "$n_gpu" -eq 2 ] && [ "$model_sz" -eq 34 ]; then continue; fi
          
          if [ "$seq_len" -eq 64 ]; then bs_list=(1 8 64 200)
          elif [ "$seq_len" -eq 512 ]; then bs_list=(1 4 8 32)
          elif [ "$seq_len" -eq 2048 ]; then bs_list=(1 4 8)
          fi

          for bs in "${bs_list[@]}"; do
            if [ "$model_sz" -eq 34 ] && [ "$bs" -ge 32 ]; then continue; fi
            echo "Running with ds_stg=$ds_stg n_gpu=$n_gpu gc=$gc seq_len=$seq_len bs=$bs model_sz=$model_sz"
            python run_bench.py --ds_stg $ds_stg --n_gpu $n_gpu --gc $gc --seq_len $seq_len --bs $bs --model_sz $model_sz
            python sync_mem.py
          done
        done
      done
    done
  done
done
