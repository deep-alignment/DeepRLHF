from abc import ABC
import os 
import shutil
import loralib as lora
import torch
from torch import nn
from torch.optim import Optimizer
from torch.utils.data import DistributedSampler
from openrlhf.utils.group_distributed_sampler import GroupDistributedSampler
from tqdm import tqdm
from openrlhf.models import PairWiseLoss, GeneralPreferenceLoss, HighDimGeneralPreferenceLoss, SFTMeanLoss, SFTSumLoss, DPORefFreeLoss, SFTVanillaLoss
from openrlhf.models import GeneralPreferenceLearnableTauLoss, GeneralPreferenceLearnableTauRegressionLoss, GeneralPreferenceRegressionLoss
from openrlhf.models import PairWiseLearnableTauLoss, PairWiseLearnableTauRegressionLoss, PairWiseRegressionLoss, HighDimGeneralPreferenceRegressionMoELoss
from openrlhf.models import HighDimGeneralPreferenceRegressionLoss, HighDimGeneralPreferenceMoELoss

class GeneralPreferenceModelTrainer(ABC):
    """
        Trainer for training a reward model.

    Args:
        model (torch.nn.Module): The model to be trained.
        strategy (Strategy): The training strategy to apply.
        optim (Optimizer): The optimizer to use during training.
        train_dataloader (DataLoader): The dataloader for the training dataset.
        eval_dataloader (DataLoader): The dataloader for the evaluation dataset.
        scheduler (Scheduler): The learning rate scheduler for dynamic adjustments during training.
        tokenizer (Tokenizer): The tokenizer for processing input text data.
        max_norm (float, defaults to 0.5): Maximum gradient norm for gradient clipping.
        max_epochs (int, defaults to 2): Maximum number of training epochs.
        is_general_preference (bool, defaults to False): Whether the model is a General Preference model.
        tau (float, defaults to 0.1): Hyperparameter tau used in the calculation of General Preference loss.
        value_head_dim (int, defaults to 2): Dimension of the value head in the General Preference model. Ignored by the Bradley Terry model.

    """

    def __init__(
        self,
        model,
        strategy,
        optim: Optimizer,
        train_dataloader,
        eval_dataloader,
        scheduler,
        tokenizer,
        max_epochs: int = 2,
        is_general_preference: bool = False,
        tau: float = 0.1,
        value_head_dim: int = 2,
        packing_samples=False,
        
    ) -> None:
        super().__init__()
        self.strategy = strategy
        self.epochs = max_epochs
        self.model = model
        self.train_dataloader = train_dataloader
        self.eval_dataloader = eval_dataloader
        self.scheduler = scheduler  # Add scheduler here
        self.optimizer = optim
        self.tokenizer = tokenizer
        self.args = strategy.args
        self.is_general_preference = is_general_preference

        # Add packing samples flag
        self.packing_samples = packing_samples

        if is_general_preference:
            if value_head_dim == 2 and not self.args.add_prompt_head:
                self.loss_fn = GeneralPreferenceLoss(tau)
                self.strategy.print("GeneralPreference Loss")
                # self.loss_fn = GeneralPreferenceRegressionLoss(tau, self.args.regression_target_margin)
                # self.loss_fn = GeneralPreferenceLearnableTauLoss()
                # self.loss_fn = GeneralPreferenceLearnableTauRegressionLoss(target_margin=self.args.regression_target_margin)
            else:
                assert value_head_dim % 2 == 0, "Dimension of value head for general preference model can not be odd!"
                if self.args.add_prompt_head:
                    self.loss_fn = HighDimGeneralPreferenceMoELoss(model=self.model, value_head_dim=value_head_dim, softmax_tau=tau)
                    # self.loss_fn = HighDimGeneralPreferenceRegressionMoELoss(model=self.model, value_head_dim=value_head_dim, target_margin=self.args.regression_target_margin, softmax_tau=tau)
                else:
                    self.loss_fn = HighDimGeneralPreferenceLoss(tau, value_head_dim)
                    # strategy.print("Loss for high-dimensional value head General Preference model.")
                    # self.loss_fn = HighDimGeneralPreferenceRegressionLoss(tau=tau, target_margin=self.args.regression_target_margin, value_head_dim=value_head_dim)    
        else:
            self.loss_fn = PairWiseLoss(tau)
            self.strategy.print("LogSigmoid Loss")
            # self.loss_fn = PairWiseRegressionLoss(tau, self.args.regression_target_margin)
            # self.loss_fn = PairWiseLearnableTauLoss()
            # self.loss_fn = PairWiseLearnableTauRegressionLoss(target_margin=self.args.regression_target_margin)

        # self.ptx_loss_fn = SFTVanillaLoss()
        # self.ptx_loss_fn = SFTMeanLoss(self.args.reward_scaler_beta)
        self.ptx_loss_fn = SFTSumLoss(self.args.reward_scaler_beta)
        # self.ptx_loss_fn = DPORefFreeLoss(self.args.reward_scaler_beta, self.args.reward_margin)
        

        self.margin_loss = self.strategy.args.margin_loss
        self.compute_fp32_loss = self.strategy.args.compute_fp32_loss
        self.packing_samples = strategy.args.packing_samples

        self._wandb = None
        if self.strategy.args.use_wandb and self.strategy.is_rank_0():
            import wandb

            self._wandb = wandb
            if not wandb.api.api_key:
                wandb.login(key=strategy.args.use_wandb)
            wandb.init(
                entity=strategy.args.wandb_org,
                project=strategy.args.wandb_project,
                group=strategy.args.wandb_group,
                name="GPM" + \
                     ("_norm" if strategy.args.is_preference_embedding_normalized else "") + \
                     ("_gating" if strategy.args.add_prompt_head else "") + \
                     f"_vdim{strategy.args.value_head_dim}" + \
                     ("_ptx" if strategy.args.add_pretrain_loss else "") + \
                     (f"{strategy.args.ptx_loss_coef}" if strategy.args.add_pretrain_loss else "") + \
                     "_M_" + str(strategy.args.pretrain) + "_D_" + \
                     str(strategy.args.dataset) + "_mbs" + \
                     str(strategy.args.micro_train_batch_size) + "_" + \
                     str(strategy.args.max_epochs) + "epoch_jobid_" + \
                     str(strategy.args.job_id) + "_" + str(strategy.args.wandb_run_name),
                config=strategy.args.__dict__,
                reinit=True,
            )

            wandb.define_metric("train/global_step")
            wandb.define_metric("train/*", step_metric="train/global_step", step_sync=True)
            wandb.define_metric("eval/global_step")
            wandb.define_metric("eval/*", step_metric="eval/global_step", step_sync=True)

        self.add_pretrain_loss = strategy.args.add_pretrain_loss
        if self.add_pretrain_loss:
            if strategy.args.ptx_loss_coef > 0:
                self.ptx_loss_fn = SFTSumLoss(strategy.args.reward_scaler_beta)
            else:
                self.add_pretrain_loss = False

    def fit(self, args, consumed_samples=0, num_update_steps_per_epoch=None):
        # get eval and save steps
        if args.eval_steps == -1:
            args.eval_steps = num_update_steps_per_epoch  # Evaluate once per epoch
        if args.save_steps == -1:
            args.save_steps = float("inf")  # do not save ckpt

        # Restore step and start_epoch
        step = consumed_samples // args.train_batch_size * self.strategy.accumulated_gradient + 1
        start_epoch = consumed_samples // args.train_batch_size // num_update_steps_per_epoch
        consumed_samples = consumed_samples % (num_update_steps_per_epoch * args.train_batch_size)

        epoch_bar = tqdm(range(start_epoch, self.epochs), desc="Train epoch", disable=not self.strategy.is_rank_0())
        for epoch in range(start_epoch, self.epochs):
            if isinstance(self.train_dataloader.sampler, DistributedSampler):
                self.train_dataloader.sampler.set_epoch(
                    epoch, consumed_samples=0 if epoch > start_epoch else consumed_samples
                )

            #  train
            step_bar = tqdm(
                range(self.train_dataloader.__len__()),
                desc="Train step of epoch %d" % epoch,
                disable=not self.strategy.is_rank_0(),
            )

            self.model.train()
            acc_mean = 0
            prob_mean = 0
            loss_mean = 0
            relative_score_mean = 0  # Add new tracking metric

            for data in self.train_dataloader:
                # Process input data based on packing mode
                if not self.packing_samples:
                    chosen_ids, c_mask, reject_ids, r_mask, margin, chosen_response_len = data
                    chosen_ids = chosen_ids.squeeze(1).to(torch.cuda.current_device())
                    c_mask = c_mask.squeeze(1).to(torch.cuda.current_device())
                    reject_ids = reject_ids.squeeze(1).to(torch.cuda.current_device())
                    r_mask = r_mask.squeeze(1).to(torch.cuda.current_device())
                    margin = torch.tensor(margin).to(torch.cuda.current_device()) if margin else None
                    chosen_response_len = torch.tensor(chosen_response_len).view(-1, 1).to(torch.cuda.current_device())
                else:
                    packed_input_ids, packed_attention_masks, packed_seq_lens, margin, chosen_response_lens = data
                    packed_input_ids = packed_input_ids.to(torch.cuda.current_device())
                    packed_attention_masks = packed_attention_masks.to(torch.cuda.current_device())
                    margin = torch.tensor(margin).to(torch.cuda.current_device()) if margin else None
                    chosen_response_len = torch.tensor(chosen_response_lens).view(-1, 1).to(torch.cuda.current_device())

                # Forward pass
                return_output = True if isinstance(self.loss_fn, (HighDimGeneralPreferenceRegressionMoELoss, HighDimGeneralPreferenceMoELoss)) else False
                if not self.packing_samples:
                    chosen_reward, reject_reward, outputs = self.concatenated_forward(
                        self.model, chosen_ids, c_mask, reject_ids, r_mask, return_output
                    )
                else:
                    chosen_reward, reject_reward, outputs = self.concatenated_forward(
                        self.model, packed_input_ids, packed_attention_masks, packed_seq_lens, None, return_output
                    )

                # Extract prompt hidden states for MoE models if needed
                prompt_hidden_state = None
                if isinstance(self.loss_fn, (HighDimGeneralPreferenceRegressionMoELoss, HighDimGeneralPreferenceMoELoss)):
                    batch_size = len(packed_seq_lens) // 2 if self.packing_samples else chosen_ids.shape[0]
                    chosen_last_hidden_states = outputs["last_hidden_state"][:batch_size, :, :]
                    prompt_end_index = chosen_last_hidden_states.size(1) - chosen_response_len - 1
                    prompt_end_index_expanded = prompt_end_index.unsqueeze(-1).expand(-1, 1, chosen_last_hidden_states.size(-1))
                    prompt_end_index_expanded = prompt_end_index_expanded.long()

                    if chosen_last_hidden_states.size(0) == 1:
                        chosen_last_hidden_states = chosen_last_hidden_states.expand(prompt_end_index_expanded.size(0), -1, -1)
                    prompt_hidden_state = torch.gather(chosen_last_hidden_states, 1, prompt_end_index_expanded).squeeze(1)

                # Compute loss and probabilities
                if self.compute_fp32_loss:
                    chosen_reward = chosen_reward.float()
                    reject_reward = reject_reward.float()
                    
                if isinstance(self.loss_fn, (HighDimGeneralPreferenceRegressionMoELoss, HighDimGeneralPreferenceMoELoss)):
                    preference_loss, relative_score = self.loss_fn(chosen_reward, reject_reward, prompt_hidden_state.to(torch.cuda.current_device()), margin)
                else:
                    preference_loss, relative_score = self.loss_fn(chosen_reward, reject_reward, margin)

                # Calculate probabilities from relative scores
                probs = torch.sigmoid(relative_score)

                # Calculate metrics
                acc = (probs > 0.5).float().mean().item()
                acc_mean = acc_mean * 0.9 + 0.1 * acc
                prob_mean = prob_mean * 0.9 + 0.1 * probs.mean().item()
                loss_mean = loss_mean * 0.9 + 0.1 * preference_loss.item()
                relative_score_mean = relative_score_mean * 0.9 + 0.1 * relative_score.mean().item()

                # Add pretrain loss if enabled
                if args.add_pretrain_loss:
                    if not self.packing_samples:
                        ptx_output = self.model.forward(chosen_ids, attention_mask=c_mask)
                        ptx_label = torch.where(
                            c_mask.bool(),
                            chosen_ids,
                            self.ptx_loss_fn.IGNORE_INDEX,
                        ).to(torch.cuda.current_device())
                        ptx_log_probs = ptx_output["logits"]
                        chosen_reward_ptx_loss = self.ptx_loss_fn(ptx_log_probs, ptx_label, c_mask.bool())
                    else:
                        # Extract chosen sequences for pretraining loss
                        num_sequences = len(packed_seq_lens)
                        num_chosen = num_sequences // 2
                        chosen_seq_lens = packed_seq_lens[:num_chosen]
                        chosen_total_len = sum(chosen_seq_lens)
                        chosen_ids = packed_input_ids[:, :chosen_total_len]
                        chosen_attn_mask = packed_attention_masks[:, :chosen_total_len]
                        
                        ptx_output = self.model.forward(chosen_ids, attention_mask=chosen_attn_mask)
                        ptx_label = torch.where(
                            chosen_attn_mask.bool(),
                            chosen_ids,
                            self.ptx_loss_fn.IGNORE_INDEX,
                        ).to(torch.cuda.current_device())
                        ptx_log_probs = ptx_output["logits"]
                        chosen_reward_ptx_loss = self.ptx_loss_fn(ptx_log_probs, ptx_label, chosen_attn_mask.bool())

                    loss = (1 - args.ptx_loss_coef) * preference_loss + chosen_reward_ptx_loss * args.ptx_loss_coef
                else:
                    loss = preference_loss

                # Backward pass and optimization
                self.strategy.backward(loss, self.model, self.optimizer)
                self.strategy.optimizer_step(self.optimizer, self.model, self.scheduler)

                # Update logs
                logs_dict = {
                    "loss": preference_loss.item(),
                    "loss_mean": loss_mean,
                    "lr": self.scheduler.get_last_lr()[0],
                    "acc": acc,
                    "acc_mean": acc_mean,
                    "probs": probs.mean().item(),
                    "prob_mean": prob_mean,
                    "relative_score": relative_score.mean().item(),
                    "relative_score_mean": relative_score_mean,
                }

                # logs/checkpoints/evaluation
                if step % self.strategy.accumulated_gradient == 0:
                    global_step = step // self.strategy.accumulated_gradient
                    client_states = {"consumed_samples": global_step * args.train_batch_size}
                    self.save_logs_and_checkpoints(args, global_step, step_bar, logs_dict, client_states)

                step += 1
                step_bar.update()

            epoch_bar.update()

        if self._wandb is not None and self.strategy.is_rank_0():
            self._wandb.finish()

    def save_logs_and_checkpoints(self, args, global_step, step_bar, logs_dict={}, client_states={}):
        if global_step % args.logging_steps == 0:
            # step bar
            logs_dict = self.strategy.all_reduce(logs_dict)
            step_bar.set_postfix(logs_dict)
            # wandb
            if self._wandb is not None and self.strategy.is_rank_0():
                logs = {"train/%s" % k: v for k, v in {**logs_dict, "global_step": global_step}.items()}
                self._wandb.log(logs)

        # eval
        if global_step % args.eval_steps == 0:
            self.evaluate(self.eval_dataloader, global_step)

        # save ckpt
        if global_step % args.save_steps == 0:
            tag = f"global_step{global_step}"
            self.strategy.save_ckpt(
                self.model, args.ckpt_path, tag, args.max_ckpt_num, args.max_ckpt_mem, client_states
            )

    def evaluate(self, eval_dataloader, steps=0):
        if not eval_dataloader or len(eval_dataloader) == 0:
            self.strategy.print("Warning: Skipping evaluation due to empty eval_dataloader")
            return

        step_bar = tqdm(
            range(eval_dataloader.__len__()),
            desc="Eval stage of steps %d" % steps,
            disable=not self.strategy.is_rank_0(),
        )
        
        self.model.eval()
        with torch.no_grad():
            acc = 0
            rewards = []
            loss_sum = 0
            prob_sum = 0
            relative_score_sum = 0  # Add tracking for relative scores

            for data in self.eval_dataloader:
                # Process input data based on packing mode
                if not self.packing_samples:
                    chosen_ids, c_mask, reject_ids, r_mask, margin, chosen_response_len = data
                    chosen_ids = chosen_ids.squeeze(1).to(torch.cuda.current_device())
                    c_mask = c_mask.squeeze(1).to(torch.cuda.current_device())
                    reject_ids = reject_ids.squeeze(1).to(torch.cuda.current_device())
                    r_mask = r_mask.squeeze(1).to(torch.cuda.current_device())
                    margin = torch.tensor(margin).to(torch.cuda.current_device()) if margin else None
                    chosen_response_len = torch.tensor(chosen_response_len).view(-1, 1).to(torch.cuda.current_device())
                else:
                    packed_input_ids, packed_attention_masks, packed_seq_lens, margin, chosen_response_lens = data
                    packed_input_ids = packed_input_ids.to(torch.cuda.current_device())
                    packed_attention_masks = packed_attention_masks.to(torch.cuda.current_device())
                    margin = torch.tensor(margin).to(torch.cuda.current_device()) if margin else None
                    chosen_response_len = torch.tensor(chosen_response_lens).view(-1, 1).to(torch.cuda.current_device())

                # Forward pass
                return_output = True if isinstance(self.loss_fn, (HighDimGeneralPreferenceRegressionMoELoss, HighDimGeneralPreferenceMoELoss)) else False
                if not self.packing_samples:
                    chosen_reward, reject_reward, outputs = self.concatenated_forward(
                        self.model, chosen_ids, c_mask, reject_ids, r_mask, return_output
                    )
                else:
                    chosen_reward, reject_reward, outputs = self.concatenated_forward(
                        self.model, packed_input_ids, packed_attention_masks, packed_seq_lens, None, return_output
                    )
            
                # Extract prompt hidden states for MoE models if needed
                prompt_hidden_state = None
                if isinstance(self.loss_fn, (HighDimGeneralPreferenceRegressionMoELoss, HighDimGeneralPreferenceMoELoss)):
                    batch_size = len(packed_seq_lens) // 2 if self.packing_samples else chosen_ids.shape[0]
                    chosen_last_hidden_states = outputs["last_hidden_state"][:batch_size, :, :]
                    prompt_end_index = chosen_last_hidden_states.size(1) - chosen_response_len - 1
                    prompt_end_index_expanded = prompt_end_index.unsqueeze(-1).expand(-1, 1, chosen_last_hidden_states.size(-1))
                    prompt_end_index_expanded = prompt_end_index_expanded.long()

                    if chosen_last_hidden_states.size(0) == 1:
                        chosen_last_hidden_states = chosen_last_hidden_states.expand(prompt_end_index_expanded.size(0), -1, -1)
                    prompt_hidden_state = torch.gather(chosen_last_hidden_states, 1, prompt_end_index_expanded).squeeze(1)

                # Compute loss and probabilities
                if self.compute_fp32_loss:
                    chosen_reward = chosen_reward.float()
                    reject_reward = reject_reward.float()
                    
                if isinstance(self.loss_fn, (HighDimGeneralPreferenceRegressionMoELoss, HighDimGeneralPreferenceMoELoss)):
                    preference_loss, relative_score = self.loss_fn(chosen_reward, reject_reward, prompt_hidden_state.to(torch.cuda.current_device()), margin)
                else:
                    preference_loss, relative_score = self.loss_fn(chosen_reward, reject_reward, margin)

                # Calculate probabilities from relative scores
                probs = torch.sigmoid(relative_score)

                # Collect statistics
                rewards += [chosen_reward.flatten(), reject_reward.flatten()]
                acc += (probs > 0.5).float().mean().item()
                loss_sum += preference_loss.item()
                prob_sum += probs.mean().item()
                relative_score_sum += relative_score.mean().item()  # Calculate relative score

                step_bar.update()

            acc_mean = acc / eval_dataloader.__len__()
            loss_mean = loss_sum / eval_dataloader.__len__() 
            prob_mean = prob_sum / eval_dataloader.__len__()
            relative_score_mean = relative_score_sum / eval_dataloader.__len__()  # Calculate mean relative score

            # Calculate reward statistics
            rewards = torch.cat(rewards).float()
            rewards = self.strategy.all_gather(rewards)
            reward_mean = torch.mean(rewards)
            reward_std = torch.std(rewards).clamp(min=1e-8)

            # Update model config with reward statistics
            self.strategy.print("Set reward mean std")
            unwrap_model = self.strategy._unwrap_model(self.model)
            unwrap_model.config.mean = reward_mean.item()
            unwrap_model.config.std = reward_std.item()

            bar_dict = {
                "eval_loss": loss_mean,
                "acc_mean": acc_mean,
                "prob_mean": prob_mean,
                "reward_mean": reward_mean.item(),
                "reward_std": reward_std.item(),
                "relative_score_mean": relative_score_mean,
            }
            logs = self.strategy.all_reduce(bar_dict)
            step_bar.set_postfix(logs)

            # Print reward distribution histogram
            histgram = torch.histogram(rewards.cpu(), bins=10, range=(-10, 10), density=True) * 2
            self.strategy.print("histgram")
            self.strategy.print(histgram)

            if self.strategy.is_rank_0():
                if self._wandb is not None:
                    logs = {"eval/%s" % k: v for k, v in {**logs, "global_step": steps}.items()}
                    self._wandb.log(logs)
                elif self._tensorboard is not None:
                    for k, v in logs.items():
                        self._tensorboard.add_scalar(f"eval/{k}", v, steps)

        self.model.train()  # reset model state
        # torch.cuda.empty_cache()
        # if self.strategy.is_rank_0():
            # return loss_mean

    def concatenated_forward(self, model, chosen_ids, c_mask, reject_ids, r_mask, return_output: bool = False):
        """Run the given model on concatenated inputs or packed samples"""
        if not self.packing_samples:
            # Original concatenated handling
            input_ids, att_masks = self.concatenated_inputs(chosen_ids, c_mask, reject_ids, r_mask)
            all_values, outputs = model.custom_forward(input_ids, attention_mask=att_masks, return_output=return_output)
            chosen_rewards = all_values[: chosen_ids.shape[0]]
            rejected_rewards = all_values[chosen_ids.shape[0] :]
        else:
            # For packed samples
            all_values, outputs = model.custom_forward(
                chosen_ids,  # Contains packed input ids 
                attention_mask=c_mask,
                return_output=return_output,
                ring_attn_group=self.strategy.ring_attn_group if hasattr(self.strategy, 'ring_attn_group') else None,
                packed_seq_lens=reject_ids  # Contains packed sequence lengths
            )
            
            # Split rewards between chosen and rejected
            num_sequences = len(reject_ids)  # reject_ids contains packed sequence lengths
            num_chosen = num_sequences // 2
            chosen_rewards = all_values[:num_chosen] 
            rejected_rewards = all_values[num_chosen:num_sequences]

        return chosen_rewards, rejected_rewards, outputs

    def concatenated_inputs(self, chosen_ids, c_mask, reject_ids, r_mask):
        """Concatenate the chosen and rejected inputs into a single tensor.

        Args:
            batch: A batch of data. Must contain the keys 'chosen_input_ids' and 'rejected_input_ids', which are tensors of shape (batch_size, sequence_length).

        Returns:
            A dictionary containing the concatenated inputs under the key 'concatenated_input_ids'.
        """

        def pad_to_length(tensor, length, pad_value, dim=-1):
            if tensor.size(dim) >= length:
                return tensor
            else:
                pad_size = list(tensor.shape)
                pad_size[dim] = length - tensor.size(dim)
                # left pad
                return torch.cat(
                    [pad_value * torch.ones(*pad_size, dtype=tensor.dtype, device=tensor.device), tensor], dim=dim
                )

        max_length = max(chosen_ids.shape[1], reject_ids.shape[1])
        inputs_ids = torch.cat(
            (
                pad_to_length(chosen_ids, max_length, self.tokenizer.pad_token_id),
                pad_to_length(reject_ids, max_length, self.tokenizer.pad_token_id),
            ),
            dim=0,
        )
        max_length = max(c_mask.shape[1], r_mask.shape[1])
        att_masks = torch.cat((pad_to_length(c_mask, max_length, 0), pad_to_length(r_mask, max_length, 0)), dim=0)
        return inputs_ids, att_masks