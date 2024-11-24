export CUDA_VISIBLE_DEVICES=0
python run_rm_rewardbench.py \
--model general-preference/GPM-Gemma-2B \
--chat_template raw \
--is_custom_model \
--do_not_save \
--model_name "model" \
--batch_size 64 \
--value_head_dim 8 \
--max_length 4096 \
--is_general_preference \
--add_prompt_head 

# --bf16 \
# --flash_attn \