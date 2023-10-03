WANDB_ENTITY=hamelsmu WANDB_PROJECT=deepspeed-bench WANDB_LOG_MODEL=all WANDB_RUN_ID=memtest—z3-chkptF-3gpu-bs5-13b-v6  \
    torchrun --nproc_per_node 3 --master_port=29506 run_lora.py \
  --model_id "meta-llama/Llama-2-13b-hf" \
  --dataset_path dolly-processed-tiny-truncated \
  --output_dir "meta-llama/Llama-2-13b-hf-fa" \
  --num_train_epochs 1 \
  --per_device_train_batch_size 5 \
  --learning_rate 4e-3 \
  --gradient_checkpointing False \
  --gradient_accumulation_steps 2 \
  --bf16 True \
  --tf32 True \
  --lr_scheduler_type "constant_with_warmup" \
  --logging_steps 25 \
  --report_to "wandb" \
  --deepspeed z3.json

CUDA_VISIBLE_DEVICES=0,1 WANDB_ENTITY=hamelsmu WANDB_PROJECT=deepspeed-bench WANDB_LOG_MODEL=all WANDB_RUN_ID=memtest—z3-chkptF-2gpu-bs5-13b-v6  \
    torchrun --nproc_per_node 2 --master_port=29506 run_lora.py \
  --model_id "meta-llama/Llama-2-13b-hf" \
  --dataset_path dolly-processed-tiny-truncated \
  --output_dir "meta-llama/Llama-2-13b-hf-fa" \
  --num_train_epochs 1 \
  --per_device_train_batch_size 5 \
  --learning_rate 4e-3 \
  --gradient_checkpointing False \
  --gradient_accumulation_steps 2 \
  --bf16 True \
  --tf32 True \
  --lr_scheduler_type "constant_with_warmup" \
  --logging_steps 25 \
  --report_to "wandb" \
  --deepspeed z3.json

CUDA_VISIBLE_DEVICES=0 WANDB_ENTITY=hamelsmu WANDB_PROJECT=deepspeed-bench WANDB_LOG_MODEL=all WANDB_RUN_ID=memtest—z3-chkptF-1gpu-bs5-13b-v6  \
    torchrun --nproc_per_node 2 --master_port=29506 run_lora.py \
  --model_id "meta-llama/Llama-2-13b-hf" \
  --dataset_path dolly-processed-tiny-truncated \
  --output_dir "meta-llama/Llama-2-13b-hf-fa" \
  --num_train_epochs 1 \
  --per_device_train_batch_size 5 \
  --learning_rate 4e-3 \
  --gradient_checkpointing False \
  --gradient_accumulation_steps 2 \
  --bf16 True \
  --tf32 True \
  --lr_scheduler_type "constant_with_warmup" \
  --logging_steps 25 \
  --report_to "wandb" \
  --deepspeed z3.json

CUDA_VISIBLE_DEVICES=0,1 WANDB_ENTITY=hamelsmu WANDB_PROJECT=deepspeed-bench WANDB_LOG_MODEL=all WANDB_RUN_ID=memtest—z3-chkptF-2gpu-bs2-34b-v6  \
    torchrun --nproc_per_node 2 --master_port=29506 run_lora.py \
  --model_id "codellama/CodeLlama-34b-Python-hf" \
  --dataset_path dolly-processed-tiny-truncated \
  --output_dir "codellama/CodeLlama-34b-Python-hf-fa" \
  --num_train_epochs 1 \
  --per_device_train_batch_size 2 \
  --learning_rate 4e-3 \
  --gradient_checkpointing False \
  --gradient_accumulation_steps 2 \
  --bf16 True \
  --tf32 True \
  --lr_scheduler_type "constant_with_warmup" \
  --logging_steps 25 \
  --report_to "wandb" \
  --deepspeed z3.json

WANDB_ENTITY=hamelsmu WANDB_PROJECT=deepspeed-bench WANDB_LOG_MODEL=all WANDB_RUN_ID=memtest—z3-chkptF-3gpu-bs2-34b-v6  \
    torchrun --nproc_per_node 3 --master_port=29506 run_lora.py \
  --model_id "codellama/CodeLlama-34b-Python-hf" \
  --dataset_path dolly-processed-tiny-truncated \
  --output_dir "codellama/CodeLlama-34b-Python-hf-fa" \
  --num_train_epochs 1 \
  --per_device_train_batch_size 2 \
  --learning_rate 4e-3 \
  --gradient_checkpointing False \
  --gradient_accumulation_steps 2 \
  --bf16 True \
  --tf32 True \
  --lr_scheduler_type "constant_with_warmup" \
  --logging_steps 25 \
  --report_to "wandb" \
  --deepspeed z3.json


  ## BS = 1

  WANDB_ENTITY=hamelsmu WANDB_PROJECT=deepspeed-bench WANDB_LOG_MODEL=all WANDB_RUN_ID=memtest—z3-chkptF-3gpu-bs1-13b-v6  \
    torchrun --nproc_per_node 3 --master_port=29506 run_lora.py \
  --model_id "meta-llama/Llama-2-13b-hf" \
  --dataset_path dolly-processed-tiny-truncated \
  --output_dir "meta-llama/Llama-2-13b-hf-fa" \
  --num_train_epochs 1 \
  --per_device_train_batch_size 1 \
  --learning_rate 4e-3 \
  --gradient_checkpointing False \
  --gradient_accumulation_steps 2 \
  --bf16 True \
  --tf32 True \
  --lr_scheduler_type "constant_with_warmup" \
  --logging_steps 25 \
  --report_to "wandb" \
  --deepspeed z3.json

CUDA_VISIBLE_DEVICES=0,1 WANDB_ENTITY=hamelsmu WANDB_PROJECT=deepspeed-bench WANDB_LOG_MODEL=all WANDB_RUN_ID=memtest—z3-chkptF-2gpu-bs1-13b-v6  \
    torchrun --nproc_per_node 2 --master_port=29506 run_lora.py \
  --model_id "meta-llama/Llama-2-13b-hf" \
  --dataset_path dolly-processed-tiny-truncated \
  --output_dir "meta-llama/Llama-2-13b-hf-fa" \
  --num_train_epochs 1 \
  --per_device_train_batch_size 1 \
  --learning_rate 4e-3 \
  --gradient_checkpointing False \
  --gradient_accumulation_steps 2 \
  --bf16 True \
  --tf32 True \
  --lr_scheduler_type "constant_with_warmup" \
  --logging_steps 25 \
  --report_to "wandb" \
  --deepspeed z3.json

CUDA_VISIBLE_DEVICES=0 WANDB_ENTITY=hamelsmu WANDB_PROJECT=deepspeed-bench WANDB_LOG_MODEL=all WANDB_RUN_ID=memtest—z3-chkptF-1gpu-bs1-13b-v6  \
    torchrun --nproc_per_node 2 --master_port=29506 run_lora.py \
  --model_id "meta-llama/Llama-2-13b-hf" \
  --dataset_path dolly-processed-tiny-truncated \
  --output_dir "meta-llama/Llama-2-13b-hf-fa" \
  --num_train_epochs 1 \
  --per_device_train_batch_size 1 \
  --learning_rate 4e-3 \
  --gradient_checkpointing False \
  --gradient_accumulation_steps 2 \
  --bf16 True \
  --tf32 True \
  --lr_scheduler_type "constant_with_warmup" \
  --logging_steps 25 \
  --report_to "wandb" \
  --deepspeed z3.json

CUDA_VISIBLE_DEVICES=0,1 WANDB_ENTITY=hamelsmu WANDB_PROJECT=deepspeed-bench WANDB_LOG_MODEL=all WANDB_RUN_ID=memtest—z3-chkptF-2gpu-bs1-34b-v6  \
    torchrun --nproc_per_node 2 --master_port=29506 run_lora.py \
  --model_id "codellama/CodeLlama-34b-Python-hf" \
  --dataset_path dolly-processed-tiny-truncated \
  --output_dir "codellama/CodeLlama-34b-Python-hf-fa" \
  --num_train_epochs 1 \
  --per_device_train_batch_size 1 \
  --learning_rate 4e-3 \
  --gradient_checkpointing False \
  --gradient_accumulation_steps 2 \
  --bf16 True \
  --tf32 True \
  --lr_scheduler_type "constant_with_warmup" \
  --logging_steps 25 \
  --report_to "wandb" \
  --deepspeed z3.json

WANDB_ENTITY=hamelsmu WANDB_PROJECT=deepspeed-bench WANDB_LOG_MODEL=all WANDB_RUN_ID=memtest—z3-chkptF-3gpu-bs1-34b-v6  \
    torchrun --nproc_per_node 3 --master_port=29506 run_lora.py \
  --model_id "codellama/CodeLlama-34b-Python-hf" \
  --dataset_path dolly-processed-tiny-truncated \
  --output_dir "codellama/CodeLlama-34b-Python-hf-fa" \
  --num_train_epochs 1 \
  --per_device_train_batch_size 1 \
  --learning_rate 4e-3 \
  --gradient_checkpointing False \
  --gradient_accumulation_steps 2 \
  --bf16 True \
  --tf32 True \
  --lr_scheduler_type "constant_with_warmup" \
  --logging_steps 25 \
  --report_to "wandb" \
  --deepspeed z3.json
  