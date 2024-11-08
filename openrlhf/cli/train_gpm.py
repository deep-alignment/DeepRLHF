import argparse
import math
import os
from datetime import datetime
from transformers.trainer import get_scheduler

from openrlhf.datasets import GeneralRewardDataset
from openrlhf.models import get_general_preference_model
from openrlhf.trainer import GeneralPreferenceModelTrainer
from openrlhf.utils import blending_datasets, get_strategy, get_tokenizer

def train(args):
    # configure strategy
    strategy = get_strategy(args)
    strategy.setup_distributed()

    # Configure model - keep GPM specific parameters
    model = get_general_preference_model(
        args.pretrain,
        use_flash_attention_2=args.flash_attn,
        bf16=args.bf16,
        load_in_4bit=args.load_in_4bit,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        target_modules=args.target_modules,
        lora_dropout=args.lora_dropout,
        ds_config=strategy.get_ds_train_config(is_actor=False),
        init_value_head=True,
        is_general_preference=args.is_general_preference,
        value_head_dim=args.value_head_dim,
        init_prompt_head=True,
        add_prompt_head=args.add_prompt_head,
        is_preference_embedding_normalized=args.is_preference_embedding_normalized,
    )

    # configure tokenizer
    tokenizer = get_tokenizer(args.pretrain, model, "left", strategy, use_fast=not args.disable_fast_tokenizer)
    tokenizer.truncation_side = "right"
    
    strategy.print(model)

    # configure optimizer
    optim = strategy.create_optimizer(
        model, lr=args.learning_rate, betas=args.adam_betas, weight_decay=args.l2
    )

    # Prepare datasets - align with RM pattern but keep GPM features
    train_data, eval_data = blending_datasets(
        args.dataset,
        args.dataset_probs,
        strategy,
        args.seed,
        max_count=args.max_samples,
        stopping_strategy="all_exhausted",
        train_split=args.train_split,
        eval_split=args.eval_split,
        train_split_ratio=args.train_split_ratio,
    )

    train_dataset = GeneralRewardDataset(
        train_data.select(range(min(args.max_samples, len(train_data)))),
        tokenizer,
        args.max_len,
        strategy,
        input_template=args.input_template,
        is_custom=args.is_custom_dataset,
        return_prompt_length=args.return_prompt_length,
        multiple_of=args.ring_attn_size,
    )

    eval_dataset = GeneralRewardDataset(
        eval_data.select(range(min(args.max_samples, len(eval_data)))),
        tokenizer,
        args.max_len,
        strategy,
        input_template=args.input_template,
        is_custom=args.is_custom_dataset,
        return_prompt_length=args.return_prompt_length,
        multiple_of=args.ring_attn_size,
    ) if len(eval_data) > 0 else None

    # Create dataloaders only for non-empty datasets
    train_dataloader = strategy.setup_dataloader(
        train_dataset,
        args.micro_train_batch_size,
        True,
        True,
        train_dataset.packing_collate_fn if args.packing_samples else train_dataset.collate_fn,  # Choose collate function based on packing_samples
        group_size=args.group_size,
    )

    eval_dataloader = strategy.setup_dataloader(
        eval_dataset,
        args.micro_train_batch_size,
        True,
        False,
        eval_dataset.packing_collate_fn if args.packing_samples else eval_dataset.collate_fn,
    ) if eval_dataset is not None else None

    # scheduler
    num_update_steps_per_epoch = len(train_dataset) // args.train_batch_size
    max_steps = math.ceil(args.max_epochs * num_update_steps_per_epoch)

    scheduler = get_scheduler(
        "cosine_with_min_lr",
        optim,
        num_warmup_steps=math.ceil(max_steps * 0.03),
        num_training_steps=max_steps,
        scheduler_specific_kwargs={"min_lr": args.learning_rate * 0.1},
    )

    # gradient_checkpointing
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": args.gradient_checkpointing_use_reentrant}
        )

    # strategy prepare
    (model, optim, scheduler) = strategy.prepare((model, optim, scheduler))

    # load checkpoint
    consumed_samples = 0
    if args.load_checkpoint and os.path.exists(args.ckpt_path):
        _, states = strategy.load_ckpt(model, args.ckpt_path)
        consumed_samples = states["consumed_samples"]
        strategy.print(f"Loaded the checkpoint: {args.ckpt_path}, consumed_samples: {consumed_samples}")

    os.makedirs(args.save_path, exist_ok=True)

    # batch_size here is micro_batch_size * 2
    # we use merged chosen + rejected response forward
    trainer = GeneralPreferenceModelTrainer(
        model=model,
        strategy=strategy,
        optim=optim,
        tokenizer=tokenizer,
        train_dataloader=train_dataloader,
        eval_dataloader=eval_dataloader,
        scheduler=scheduler,
        max_epochs=args.max_epochs,
        is_general_preference=args.is_general_preference,
        tau=args.general_preference_tau,
        value_head_dim=args.value_head_dim,
    )

    # Only assign eval_dataloader if it exists and has data
    if eval_dataloader is not None and len(eval_dataset) > 0:
        trainer.eval_dataloader = eval_dataloader
    else:
        strategy.print("Skipping evaluation due to empty eval_dataloader.")

    # Proceed with training
    trainer.fit(args, consumed_samples, num_update_steps_per_epoch)

    # save model checkpoint after fitting on only rank0
    strategy.save_model(model, tokenizer, args.save_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrain", type=str, default="google/gemma-2b")
    parser.add_argument("--dataset", type=str, default="../data/test_data/test_data.jsonl")
    parser.add_argument("--dataset_probs", type=str, default="1.0", help="sampling probs for datasets")
    parser.add_argument("--save_path", type=str, default="../results/saved_model")
    parser.add_argument("--save_steps", type=int, default=-1)
    parser.add_argument("--logging_steps", type=int, default=1)
    parser.add_argument("--eval_steps", type=int, default=-1)
    parser.add_argument("--ckpt_path", type=str, default="../results/saved_model/checkpoint")
    parser.add_argument("--max_ckpt_num", type=int, default=3)
    parser.add_argument("--max_ckpt_mem", type=int, default=1000)  # 1000GB
    parser.add_argument("--max_epochs", type=int, default=1)
    parser.add_argument("--micro_train_batch_size", type=int, default=8)
    parser.add_argument("--accumulated_gradient", type=int, default=1)
    parser.add_argument("--max_samples", type=int, default=1000000)
    parser.add_argument("--load_checkpoint", action="store_true", default=False)
    parser.add_argument("--max_norm", type=float, default=1.0)
    parser.add_argument("--max_len", type=int, default=512)
    parser.add_argument("--l2", type=float, default=0.0)
    parser.add_argument("--gradient_checkpointing", action="store_true", default=False)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--local_rank", type=int, default=-1, help="local_rank for deepspeed")
    parser.add_argument("--zero_stage", type=int, default=2)
    parser.add_argument("--bf16", action="store_true", default=False)
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--zpg", type=int, default=1, help="ZeRO++ max partition size")
    parser.add_argument("--adam_offload", action="store_true", default=False)
    parser.add_argument("--flash_attn", action="store_true", default=False)
    parser.add_argument("--compute_fp32_loss", action="store_true", default=False)
    parser.add_argument("--margin_loss", action="store_true", default=False)
    parser.add_argument("--grad_accum_dtype", type=str, default=None)
    parser.add_argument("--disable_trace_cache", action="store_true", default=False)
    parser.add_argument("--load_in_4bit", action="store_true", default=False)
    parser.add_argument("--lora_rank", type=int, default=0)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--lora_dropout", type=float, default=0)
    parser.add_argument("--target_modules", type=str, nargs="*", default="all-linear")
    parser.add_argument("--input_template", type=str, default="Human:\n{}\nAssistant:\n")
    parser.add_argument("--gradient_checkpointing_use_reentrant", action="store_true")
    parser.add_argument("--disable_fast_tokenizer", action="store_true", default=False)
    
    # custom arguments added 
    parser.add_argument("--is_custom_dataset", action="store_true", default=False, help="Whether to use custom cyclic dataset. Default to False.")
    parser.add_argument("--is_general_preference", action="store_true", default=False, help="Whether to use General Preference model. Default to False (Bradley Terry model by default).")
    parser.add_argument("--general_preference_tau", type=float, default=0.1, help="Hyperparameter tau used in general preference loss.")
    parser.add_argument("--group_size", type=int, default=1, help="Number of data to group together during shuffling.")
    parser.add_argument("--value_head_dim", type=int, default=2, help="Dimension of the value head in the general preference model. Ignored by the Bradley Terry model. Should be even.")
    parser.add_argument("--save_best_model", type=int, default=None, help="Save the top N models with the lowest evaluation loss.")
    parser.add_argument("--add_pretrain_loss", action="store_true", default=False, help="Include the pretraining loss of chosen inputs in the total loss calculation.")
    parser.add_argument("--ptx_loss_coef", type=float, default=0.01, help="coefficient for pretraining loss included in the total loss.")
    parser.add_argument("--train_split_ratio", type=float, default=0.97, help="Ratio of the dataset to use for training. (1-train_split_ratio) for validation. Should not exceed 1.")
    parser.add_argument("--train_split", type=str, default="train", help="train split of the HF dataset")
    parser.add_argument("--eval_split", type=str, default="test", help="test split of the dataset")
    parser.add_argument("--reward_scaler_beta", type=float, default=2.0, help="A constant that controls the scaling of the reward difference.")
    parser.add_argument("--reward_margin", type=float, default=1.0, help="Chosen response exceeds rejected reward by at least reward_margin. A hyperparameter for DPORefFree Loss.")
    parser.add_argument("--regression_target_margin", type=float, default=10.0, help="Target regression margin. A hyperparameter for Regression Loss.")
    parser.add_argument("--return_prompt_length", action="store_true", default=False, help="Return the prompt length in the dataset collator if set. Default to False. Should set to be True when 'add_prompt_head' is True.")
    parser.add_argument("--add_prompt_head", action="store_true", default=False, help="Add a prompt head to the model if set. Default to False.")
    parser.add_argument("--is_preference_embedding_normalized", action="store_true", default=False, help="Whether to normalize preference embeddings")  # Updated argument name and help text
    # wandb pamameters
    parser.add_argument("--use_wandb", type=str, default=None)
    parser.add_argument("--wandb_org", type=str, default=None)
    parser.add_argument("--wandb_group", type=str, default=None)
    parser.add_argument("--wandb_project", type=str, default="train_rm_general_preference")
    parser.add_argument(
        "--wandb_run_name",
        type=str,
        default="gpm_%s" % datetime.now().strftime("%m%dT%H:%M"),
    )

    # Add args.train_batch_size and args.adam_betas
    parser.add_argument(
        "--train_batch_size", type=int, default=128, help="Global training batch size"
    )
    parser.add_argument(
        "--adam_betas",
        type=float,
        nargs=2,
        default=(0.9, 0.95),
        help="Betas for Adam optimizer",
    )

    # Add the missing arguments
    parser.add_argument("--apply_chat_template", action="store_true", default=False, help="Use HF tokenizer chat template")
    parser.add_argument("--chosen_key", type=str, default="chosen")
    parser.add_argument("--rejected_key", type=str, default="rejected")
    parser.add_argument("--prompt_key", type=str, default=None)
    parser.add_argument("--job_id", type=str, default="", help="Job ID for wandb run name")

    # Add packing samples argument
    parser.add_argument("--packing_samples", action="store_true", default=False, help="Enable sample packing using Flash Attention2")
    parser.add_argument("--ring_attn_size", type=int, default=1, help="Ring attention group size")
    parser.add_argument("--ring_head_stride", type=int, default=1, help="Ring attention head stride")

    args = parser.parse_args()

    if args.packing_samples and not args.flash_attn:
        print("[Warning] Please --flash_attn to accelerate when --packing_samples is enabled.")
        args.flash_attn = True

    if args.ring_attn_size > 1:
        assert args.packing_samples, "packing_samples must be enabled when using ring attention"

    train(args)
