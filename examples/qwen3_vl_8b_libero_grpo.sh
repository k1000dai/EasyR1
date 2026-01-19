#!/bin/bash

set -x

MODEL_PATH=Qwen/Qwen3-VL-8B-Instruct  # replace it with your local file path

python3 -m verl.trainer.main \
	config=examples/config.yaml \
	data.train_files=k1000dai/libero-fasttoken-v3@train \
	data.val_files=k1000dai/libero-fasttoken-v3@val \
	data.min_pixels=65530 \
	worker.actor.model.model_path=${MODEL_PATH} \
	worker.actor.fsdp.torch_dtype=bf16 \
worker.actor.optim.lr=2e-6 \
worker.actor.optim.strategy=adamw_bf16 \
worker.actor.optim.lr_warmup_ratio=0.01 \
worker.rollout.tensor_parallel_size=1 \
worker.rollout.n=8 \
worker.rollout.gpu_memory_utilization=0.85 \
worker.actor.offload.offload_params=false \
worker.actor.offload.offload_optimizer=false \
worker.ref.fsdp.enable_cpu_offload=false \
trainer.experiment_name=qwen3_vl_8b_libero_pickandplace_grpo_kddi \
trainer.n_gpus_per_node=8
