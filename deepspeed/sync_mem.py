from fastcore.utils import Path
import wandb
import json

def read_and_sync_to_wandb(wandb_user, wandb_proj, run_id):
    dpth = Path(f"mem_data/{run_id}")
    api = wandb.Api()
    run_path = f"{wandb_user}/{wandb_proj}/{run_id}"
    run = api.run(run_path)
    for f in dpth.ls():
        if f.name.endswith(".json"):
            with open(f, 'r') as f:
                memory_stats = json.load(f)
                gpu = memory_stats['gpu']
                mem = memory_stats['mem']
                run.summary[f"gpu_{gpu}_mem"] =  mem
    run.summary.update()


for d in Path('mem_data').ls(): 
    if d.name.startswith('z') and d.is_dir():
        print(f"syncing {d.name} to wandb")
        read_and_sync_to_wandb('hamelsmu', 'deepspeed-data', d.name)
