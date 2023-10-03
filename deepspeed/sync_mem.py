from run_lora import read_and_sync_to_wandb, Path
for d in Path('mem_data').ls(): read_and_sync_to_wandb('hamelsmu', 'deepspeed-data', d.name)
