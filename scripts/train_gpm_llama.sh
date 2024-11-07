set -x

read -r -d '' training_commands <<EOF
openrlhf.cli.train_gpm \
   --save_path ./checkpoint/Llama-3.2-3B-Instruct-GPM \
   --save_steps -1 \
   --logging_steps 1 \
   --eval_steps -1 \
   --train_batch_size 128 \
   --micro_train_batch_size 8 \
   --pretrain meta-llama/Llama-3.2-3B-Instruct \
   --bf16 \
   --max_epochs 1 \
   --max_len 8192 \
   --zero_stage 3 \
   --learning_rate 2e-6 \
   --l2 1e-3 \
   --general_preference_tau 1 \
   --dataset_probs 1 \
   --group_size 1 \
   --value_head_dim 2 \
   --add_pretrain_loss \
   --ptx_loss_coef 0.01 \
   --is_general_preference \
   --train_split_ratio 1.0 \
   --save_best_model 2 \
   --dataset Skywork/Skywork-Reward-Preference-80K-v0.2 \
   --apply_chat_template \
   --chosen_key chosen \
   --rejected_key rejected \
   --flash_attn \
   --load_checkpoint \
   --gradient_checkpointing \
   --use_wandb True \
   --wandb_project deeprlhf-rm
EOF
     # --use_wandb [WANDB_TOKENS] or True (use wandb login command)
     # --packing_samples


if [[ ${1} != "slurm" ]]; then
    deepspeed --module $training_commands
fi
