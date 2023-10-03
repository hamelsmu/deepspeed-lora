# AUTOGENERATED! DO NOT EDIT! File to edit: run_bench.ipynb.

# %% auto 0
__all__ = ['gen_cli']

# %% run_bench.ipynb 1
from fastcore.script import call_parse, Param, store_true
from random import randint
import os, subprocess

# %% run_bench.ipynb 2
@call_parse
def gen_cli(debug:Param('Print command instead of running it', store_true),
            ds_stg:Param('The deepspeed stage', int, choices=[0,1,2,3])=3,
            n_gpu:Param('number of GPUs', int, choices=[1,2,3])=1,
            gc:Param('Toggle gradient checkpointing', choices=['True', 'False'])='True',  
            seq_len:Param('Sequence length', int, choices=[64, 256, 512, 1024, 2048])=256,
            bs:Param('Batch size', int, choices=[1,3,4,6,8,16,32, 64,100, 128, 200, 256])=1,
            model_sz:Param('Model size in Billions', int, choices=[3, 7, 13, 34])=7,
            n_epochs:Param('# of epochs', int) = 1,
           ):
    "Generate Training CLI Command"

    model_id = {3:'pankajmathur/orca_mini_3b', 7:'NousResearch/Llama-2-7b-hf', 13: 'NousResearch/Llama-2-13b-hf', 34: 'NousResearch/CodeLlama-34b-hf'}[model_sz]
    nr = randint(10000000,99999999)
    env_values = [('WANDB_ENTITY', 'hamelsmu'), ('WANDB_PROJECT', 'deepspeed-data'),
                  ('WANDB_RUN_ID', f'z{ds_stg}-n_gpu{n_gpu}-gc{gc}-seq_len{seq_len}-bs{bs}-model_sz{model_sz}-{nr}')]
    env_str = ''
    for v in env_values:
        env_str  += f'{v[0]}={v[1]} '
    
    cmd = f"""torchrun --nproc_per_node {n_gpu} run_lora.py \
  --model_id {model_id} \
  --dataset_path data_{seq_len} \
  --output_dir {model_id}-fa \
  --num_train_epochs {n_epochs} \
  --per_device_train_batch_size {bs} \
  --learning_rate 4e-3 \
  --gradient_checkpointing {gc} \
  --bf16 True \
  --tf32 True \
  --lr_scheduler_type constant_with_warmup \
  --logging_steps 25 \
  --report_to wandb \
  --deepspeed z{ds_stg}.json"""

    full_cmd = env_str + ' ' + cmd

    if debug:
        print(full_cmd)
    else:
        env_vars = os.environ.copy()
        for v in env_values:
            env_vars[v[0]] = v[1]
        print(f"running command:\n{full_cmd}")
        return subprocess.run(cmd.split(), env=env_vars)
