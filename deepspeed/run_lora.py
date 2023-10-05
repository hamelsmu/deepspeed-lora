from dataclasses import dataclass, field
from typing import cast

import os
import subprocess
import deepspeed
from fastcore.utils import Path
from typing import Optional
import torch
import wandb

from transformers import HfArgumentParser, TrainingArguments, Trainer, TrainerCallback
from utils.peft_utils import SaveDeepSpeedPeftModelCallback, create_and_prepare_model
from datasets import load_from_disk


import json
import torch


wandb_user=os.getenv('WANDB_ENTITY')
wandb_proj=os.getenv('WANDB_PROJECT')
wandb_run_id=os.getenv('WANDB_RUN_ID')
wandb_run_path = f"{wandb_user}/{wandb_proj}/{wandb_run_id}"


def serialize_gpu_mem(run_id, run_path, proc_id):
    Path(f"mem_data/{run_id}").mkdir(parents=True, exist_ok=True)
    filename = f"mem_data/{run_id}/{proc_id}_gpu_mem.json"
    with torch.cuda.device(proc_id):
        max_mem_gb = torch.cuda.max_memory_allocated() / (1024 ** 3)
        memory_stats = {'run_id': run_id, 'run_path': run_path, 'gpu': proc_id, 'mem': max_mem_gb}
    with open(filename, 'w') as f: json.dump(memory_stats, f)


# Define and parse arguments.
@dataclass
class ScriptArguments:
    """
    Additional arguments for training, which are not part of TrainingArguments.
    """
    model_id: str = field(
      metadata={
            "help": "The model that you want to train from the Hugging Face hub. E.g. gpt2, gpt2-xl, bert, etc."
        },
    )
    dataset_path: Optional[str] = field(
        default="timdettmers/openassistant-guanaco",
        metadata={"help": "The preference dataset to use."},
    )
    quant4: Optional[bool] = field(default=False)
    lora_alpha: Optional[int] = field(default=16)
    lora_dropout: Optional[float] = field(default=0.1)
    lora_r: Optional[int] = field(default=64)
    use_flash_attn: Optional[bool] = field(
        default=False,
        metadata={"help": "Enables Flash attention for training."},
    )
    merge_adapters: bool = field(
        metadata={"help": "Wether to merge weights for LoRA."},
        default=False,
    )


def training_function(script_args:ScriptArguments, training_args:TrainingArguments):

    # Load processed dataset from disk
    dataset = load_from_disk(script_args.dataset_path)
    
    # Load and create peft model
    model, peft_config, tokenizer = create_and_prepare_model(script_args.model_id,training_args, script_args)
    model.config.use_cache = False

    # Create trainer and add callbacks
    trainer = Trainer(model=model, args=training_args, train_dataset=dataset)
    trainer.model.print_trainable_parameters()
    trainer.add_callback(SaveDeepSpeedPeftModelCallback(trainer, save_steps=training_args.save_steps))
    
    # Start training
    trainer.train()
    serialize_gpu_mem(wandb_run_id, wandb_run_path, trainer.args.process_index)

    # Save model on main process
    trainer.accelerator.wait_for_everyone()
    state_dict = trainer.accelerator.get_state_dict(trainer.deepspeed)
    unwrapped_model = trainer.accelerator.unwrap_model(trainer.deepspeed)
    if trainer.accelerator.is_main_process:
        unwrapped_model.save_pretrained(training_args.output_dir, state_dict=state_dict)
    trainer.accelerator.wait_for_everyone()



def main():
    parser = HfArgumentParser([ScriptArguments,TrainingArguments])
    script_args, training_args = parser.parse_args_into_dataclasses()
    script_args = cast(ScriptArguments, script_args)
    training_args = cast(TrainingArguments, training_args)
    training_function(script_args, training_args)


if __name__ == "__main__":
    main()
