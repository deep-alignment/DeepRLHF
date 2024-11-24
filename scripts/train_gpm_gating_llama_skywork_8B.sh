set -x

read -r -d '' training_commands <<EOF
openrlhf.cli.train_gpm \
   --save_path ./checkpoint/Skywork-Reward-Llama-3.1-8B-v0.2-GPM \
   --save_steps -1 \
   --logging_steps 1 \
   --eval_steps -1 \
   --train_batch_size 128 \
   --micro_train_batch_size 8 \
   --pretrain Skywork/Skywork-Reward-Llama-3.1-8B-v0.2 \
   --bf16 \
   --max_epochs 2 \
   --max_len 4096 \
   --zero_stage 3 \
   --learning_rate 2e-6 \
   --l2 1e-3 \
   --general_preference_tau 0.1 \
   --dataset_probs 1 \
   --group_size 1 \
   --value_head_dim 64 \
   --is_general_preference \
   --is_preference_embedding_normalized \
   --return_prompt_length \
   --add_prompt_head \
   --is_using_nonlinear_value_head \
   --is_using_nonlinear_prompt_gate \
   --train_split_ratio 0.97 \
   --dataset Skywork/Skywork-Reward-Preference-80K-v0.2 \
   --apply_chat_template \
   --chosen_key chosen \
   --rejected_key rejected \
   --flash_attn \
   --packing_samples \
   --gradient_checkpointing \
   --use_wandb True \
   --wandb_project deeprlhf-rm \
   --job_id ${SLURM_JOB_ID:-"local"} 
EOF
     # --use_wandb [WANDB_TOKENS] or True (use wandb login command)
     # --packing_samples
     # --load_checkpoint
     # --is_preference_embedding_normalized \
     # --return_prompt_length \
     # --add_prompt_head \
     # --add_pretrain_loss \
     # --ptx_loss_coef 0.00 \
     # --is_using_nonlinear_value_head \
     # --is_using_nonlinear_prompt_gate \

if [[ ${1} != "slurm" ]]; then
    deepspeed --module $training_commands
fi
