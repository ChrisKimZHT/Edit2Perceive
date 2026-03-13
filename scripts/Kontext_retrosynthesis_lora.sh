export TORCH_NCCL_TIMEOUT=1800000

accelerate launch --config_file configs/accelerate_config.yaml scripts/train.py \
	--dataset_base_path "./uspto/processed/train" \
	--dataset_metadata_path "./uspto/processed/train/data.csv" \
	--data_file_keys "kontext_images,image" \
	--model_paths "./FLUX.1-Kontext-dev" \
	--learning_rate "1e-5" \
	--num_epochs "5" \
	--remove_prefix_in_ckpt "pipe.dit." \
	--trainable_models "dit" \
	--extra_inputs "kontext_images" \
	--use_gradient_checkpointing \
	--default_caption "Predict reactant structures from the product structure" \
	--batch_size "4" \
	--output_path "ckpts/kontext_retrosynthesis/bs4_lora" \
	--save_steps "500" \
	--eval_steps "1000000000" \
	--lora_base_model "dit" \
	--lora_target_modules "a_to_qkv,b_to_qkv,ff_a.0,ff_a.2,ff_b.0,ff_b.2,a_to_out,b_to_out,proj_out,norm.linear,norm1_a.linear,norm1_b.linear,to_qkv_mlp" \
	--lora_rank 64 \
	--align_to_opensource_format \
	--resume \
	--height 768 \
	--width 768 \
	--adamw8bit \
	--extra_loss "cycle_consistency_retrosynthesis" \
	--extra_loss_start_epoch 0
