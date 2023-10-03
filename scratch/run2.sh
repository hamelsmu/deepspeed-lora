WANDB_ENTITY=hamelsmu WANDB_PROJECT=deepspeed-bench WANDB_LOG_MODEL=all WANDB_RUN_ID=gctest—z0-chkptF-3gpu-bs10-13b-v7  \
    torchrun --nproc_per_node 3 --master_port=29506 run_lora.py \
  --model_id "meta-llama/Llama-2-13b-hf" \
  --dataset_path dolly-processed-tiny-truncated \
  --output_dir "meta-llama/Llama-2-13b-hf-fa" \
  --num_train_epochs 3 \
  --per_device_train_batch_size 10 \
  --learning_rate 4e-3 \
  --gradient_checkpointing False \
  --gradient_accumulation_steps 2 \
  --bf16 True \
  --tf32 True \
  --lr_scheduler_type "constant_with_warmup" \
  --logging_steps 25 \
  --report_to "wandb" \
  --deepspeed z0.json

WANDB_ENTITY=hamelsmu WANDB_PROJECT=deepspeed-bench WANDB_LOG_MODEL=all WANDB_RUN_ID=gctest—z1-chkptF-3gpu-bs10-13b-v7  \
    torchrun --nproc_per_node 3 --master_port=29506 run_lora.py \
  --model_id "meta-llama/Llama-2-13b-hf" \
  --dataset_path dolly-processed-tiny-truncated \
  --output_dir "meta-llama/Llama-2-13b-hf-fa" \
  --num_train_epochs 3 \
  --per_device_train_batch_size 10 \
  --learning_rate 4e-3 \
  --gradient_checkpointing False \
  --gradient_accumulation_steps 2 \
  --bf16 True \
  --tf32 True \
  --lr_scheduler_type "constant_with_warmup" \
  --logging_steps 25 \
  --report_to "wandb" \
  --deepspeed z1.json

WANDB_ENTITY=hamelsmu WANDB_PROJECT=deepspeed-bench WANDB_LOG_MODEL=all WANDB_RUN_ID=gctest—z2-chkptF-3gpu-bs10-13b-v7  \
    torchrun --nproc_per_node 3 --master_port=29506 run_lora.py \
  --model_id "meta-llama/Llama-2-13b-hf" \
  --dataset_path dolly-processed-tiny-truncated \
  --output_dir "meta-llama/Llama-2-13b-hf-fa" \
  --num_train_epochs 3 \
  --per_device_train_batch_size 10 \
  --learning_rate 4e-3 \
  --gradient_checkpointing False \
  --gradient_accumulation_steps 2 \
  --bf16 True \
  --tf32 True \
  --lr_scheduler_type "constant_with_warmup" \
  --logging_steps 25 \
  --report_to "wandb" \
  --deepspeed z2.json

WANDB_ENTITY=hamelsmu WANDB_PROJECT=deepspeed-bench WANDB_LOG_MODEL=all WANDB_RUN_ID=gctest—z3-chkptF-3gpu-bs10-13b-v7  \
    torchrun --nproc_per_node 3 --master_port=29506 run_lora.py \
  --model_id "meta-llama/Llama-2-13b-hf" \
  --dataset_path dolly-processed-tiny-truncated \
  --output_dir "meta-llama/Llama-2-13b-hf-fa" \
  --num_train_epochs 3 \
  --per_device_train_batch_size 10 \
  --learning_rate 4e-3 \
  --gradient_checkpointing False \
  --gradient_accumulation_steps 2 \
  --bf16 True \
  --tf32 True \
  --lr_scheduler_type "constant_with_warmup" \
  --logging_steps 25 \
  --report_to "wandb" \
  --deepspeed z3.json


  WANDB_ENTITY=hamelsmu WANDB_PROJECT=deepspeed-bench WANDB_LOG_MODEL=all WANDB_RUN_ID=gctest—z0-chkptT-3gpu-bs10-13b-v7  \
    torchrun --nproc_per_node 3 --master_port=29506 run_lora.py \
  --model_id "meta-llama/Llama-2-13b-hf" \
  --dataset_path dolly-processed-tiny-truncated \
  --output_dir "meta-llama/Llama-2-13b-hf-fa" \
  --num_train_epochs 3 \
  --per_device_train_batch_size 10 \
  --learning_rate 4e-3 \
  --gradient_checkpointing True \
  --gradient_accumulation_steps 2 \
  --bf16 True \
  --tf32 True \
  --lr_scheduler_type "constant_with_warmup" \
  --logging_steps 25 \
  --report_to "wandb" \
  --deepspeed z0.json

WANDB_ENTITY=hamelsmu WANDB_PROJECT=deepspeed-bench WANDB_LOG_MODEL=all WANDB_RUN_ID=gctest—z1-chkptT-3gpu-bs10-13b-v7  \
    torchrun --nproc_per_node 3 --master_port=29506 run_lora.py \
  --model_id "meta-llama/Llama-2-13b-hf" \
  --dataset_path dolly-processed-tiny-truncated \
  --output_dir "meta-llama/Llama-2-13b-hf-fa" \
  --num_train_epochs 3 \
  --per_device_train_batch_size 10 \
  --learning_rate 4e-3 \
  --gradient_checkpointing True \
  --gradient_accumulation_steps 2 \
  --bf16 True \
  --tf32 True \
  --lr_scheduler_type "constant_with_warmup" \
  --logging_steps 25 \
  --report_to "wandb" \
  --deepspeed z1.json

WANDB_ENTITY=hamelsmu WANDB_PROJECT=deepspeed-bench WANDB_LOG_MODEL=all WANDB_RUN_ID=gctest—z2-chkptT-3gpu-bs10-13b-v7  \
    torchrun --nproc_per_node 3 --master_port=29506 run_lora.py \
  --model_id "meta-llama/Llama-2-13b-hf" \
  --dataset_path dolly-processed-tiny-truncated \
  --output_dir "meta-llama/Llama-2-13b-hf-fa" \
  --num_train_epochs 3 \
  --per_device_train_batch_size 10 \
  --learning_rate 4e-3 \
  --gradient_checkpointing True \
  --gradient_accumulation_steps 2 \
  --bf16 True \
  --tf32 True \
  --lr_scheduler_type "constant_with_warmup" \
  --logging_steps 25 \
  --report_to "wandb" \
  --deepspeed z2.json

WANDB_ENTITY=hamelsmu WANDB_PROJECT=deepspeed-bench WANDB_LOG_MODEL=all WANDB_RUN_ID=gctest—z3-chkptT-3gpu-bs10-13b-v7  \
    torchrun --nproc_per_node 3 --master_port=29506 run_lora.py \
  --model_id "meta-llama/Llama-2-13b-hf" \
  --dataset_path dolly-processed-tiny-truncated \
  --output_dir "meta-llama/Llama-2-13b-hf-fa" \
  --num_train_epochs 3 \
  --per_device_train_batch_size 10 \
  --learning_rate 4e-3 \
  --gradient_checkpointing True \
  --gradient_accumulation_steps 2 \
  --bf16 True \
  --tf32 True \
  --lr_scheduler_type "constant_with_warmup" \
  --logging_steps 25 \
  --report_to "wandb" \
  --deepspeed z3.json