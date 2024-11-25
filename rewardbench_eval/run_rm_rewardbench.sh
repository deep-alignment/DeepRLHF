MODEL_NAME="Llama-3.1-8B-Instruct-GPM-dim8"
export CUDA_VISIBLE_DEVICES=0
python run_rm_rewardbench.py \
--model "../checkpoint/${MODEL_NAME}" \
--tokenizer 'meta-llama/Llama-3.1-8B-Instruct' \
--chat_template raw \
--is_custom_model \
--do_not_save \
--enable_beaker_save \
--model_name "${MODEL_NAME}" \
--batch_size 32 \
--value_head_dim 8 \
--max_length 4096 \
--is_general_preference \
--is_preference_embedding_normalized \
--add_prompt_head \
--is_using_nonlinear_value_head \
--is_using_nonlinear_prompt_gate

# Optional flags:
# --bf16 \
# --flash_attn \
# MODEL_NAME="Skywork-Reward-Llama-3.1-8B-v0.2-GPM-dim2"
# MODEL_NAME="Llama-3.1-8B-Instruct-GPM-dim2"

