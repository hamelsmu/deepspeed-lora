from fastcore.utils import Path
import wandb
from wandb import CommError
import json, os, time

def read_and_sync_to_wandb(wandb_user, wandb_proj, run_id):
    dpth = Path(f"mem_data/{run_id}")
    api = wandb.Api()

    run_path = f"{wandb_user}/{wandb_proj}/{run_id}"
    try:
        run = api.run(run_path)
    except CommError:
        print(f"Run {run_path} does not exist. Skipping...")
        return
    for f in dpth.ls():
        if f.name.endswith(".json"):
            with open(f, 'r') as f:
                memory_stats = json.load(f)
                gpu = memory_stats['gpu']
                mem = memory_stats['mem']
                run.summary[f"gpu_{gpu}_mem"] =  mem
    run.summary.update()

if __name__ == '__main__':
    try:
        with open("last_run_timestamp.txt", "r") as f:
            last_run_timestamp = float(f.read())
    except FileNotFoundError:
        last_run_timestamp = 0.0
    
    for d in Path('mem_data').ls(): 
        if d.name.startswith('z') and d.is_dir():
            if any(os.path.getmtime(f) > last_run_timestamp for f in d.ls()):
                print(f"syncing {d.name} to wandb")
                read_and_sync_to_wandb('hamelsmu', 'deepspeed-data', d.name)

    current_timestamp = time.time()
    with open("last_run_timestamp.txt", "w") as f:
        f.write(str(current_timestamp))
