set -x

read -r -d '' training_commands <<EOF
openrlhf.cli.train_rm \
   --save_path ./checkpoint/Skywork-Reward-Llama-3.1-8B-v0.2 \
   --save_steps -1 \
   --logging_steps 1 \
   --eval_steps -1 \
   --train_batch_size 128 \
   --micro_train_batch_size 8 \
   --pretrain  Skywork/Skywork-Reward-Llama-3.1-8B-v0.2 \
   --bf16 \
   --max_epochs 2 \
   --max_len 4096 \
   --zero_stage 3 \
   --learning_rate 2e-6 \
   --l2 1e-3 \
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


if [[ ${1} != "slurm" ]]; then
    deepspeed --module $training_commands
fi
