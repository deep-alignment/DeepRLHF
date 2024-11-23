# Adapted from https://github.com/OpenRLHF/OpenRLHF/blob/main/openrlhf/models/model.py

from typing import Optional
import deepspeed
import torch
import torch.nn as nn
from peft import LoraConfig, get_peft_model
from peft.tuners.lora import LoraLayer
from transformers import AutoConfig, AutoModel, BitsAndBytesConfig, AutoModelForCausalLM
from transformers.integrations.deepspeed import HfDeepSpeedConfig
from openrlhf.utils.logging_utils import init_logger
import torch.nn.functional as F

from .ring_attn_utils import convert_ring_attn_params
from .utils import reset_position_ids
import math

logger = init_logger(__name__)

# Construct reward model with a value head for sequence classification. (model also with a lm head) 
def get_general_preference_model(
    model_name_or_path: str,
    *,
    bf16=True,
    load_in_4bit=False,
    lora_rank=0,
    lora_alpha=16,
    target_modules=None,
    lora_dropout=0,
    use_flash_attention_2=False,
    ds_config: dict = None,
    init_value_head: bool = False,
    init_prompt_head: bool = False,
    add_prompt_head: bool = False,
    is_general_preference: bool = False,
    value_head_dim: int = 2,
    is_preference_embedding_normalized: bool = False,  # Renamed argument
    is_using_nonlinear_value_head: bool = False,  # Added parameter
    is_using_nonlinear_prompt_gate: bool = False,  # Added new parameter
    **kwargs,
) -> nn.Module:
    """Get reward model with a value head(linear layer) and a lm head.

    Args:
        model_name_or_path (str): Path to pretrained model.
        bf16 (bool, optional): Whether enable bfloat16. Defaults to True.
        use_flash_attention_2 (bool, optional): Whether use Flash Attention 2.0. Defaults to False.
        ds_config (dict, optional): Deepspeed config, used to automatically splitting the model onto multiple gpus during from_pretrained when ZeRO-3 enabled. Defaults to None.
        init_value_head (bool, optional): Whether to initialize the value head weights. Defaults to False.
        is_general_preference (bool, optional): Whether to use General Preference model. Defaults to False (Bradley Terry model by default).
        value_head_dim (int, optional): Dimension of value head for General Prefernce model. Ignored by the Bradley Terry model. Defaults to 2.

    Returns:
        nn.Module: pretrained transformer model.
    """

    config = AutoConfig.from_pretrained(model_name_or_path, trust_remote_code=True)
    config._attn_implementation = "flash_attention_2" if use_flash_attention_2 else "eager"
    base_class = AutoModel._model_mapping[type(config)]
    base_causal_class = AutoModelForCausalLM._model_mapping.get(type(config), None)
    cls_class = _get_general_preference_model(base_causal_class, base_class, is_general_preference, add_prompt_head, value_head_dim, is_preference_embedding_normalized=is_preference_embedding_normalized, is_using_nonlinear_value_head=is_using_nonlinear_value_head, is_using_nonlinear_prompt_gate=is_using_nonlinear_prompt_gate)  # Updated argument name
    # Note: dschf is defined in function scope to avoid global effects
    # https://huggingface.co/docs/transformers/main_classes/deepspeed#nontrainer-deepspeed-integration
    
    if ds_config is not None and ds_config["zero_optimization"]["stage"] == 3:
        dschf = HfDeepSpeedConfig(ds_config)
        ds_config = None
    else:
        dschf = None

    if load_in_4bit:
        assert bf16, "we only support bnb_4bit_compute_dtype = bf16"
        nf4_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
    else:
        nf4_config = None

    model = cls_class.from_pretrained(
        model_name_or_path,
        config=config,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16 if bf16 else "auto",
        quantization_config=nf4_config,
        **kwargs,
    )
    # LoRA
    if lora_rank > 0:
        model.enable_input_require_grads()
        lora_config = LoraConfig(
            r=lora_rank,
            lora_alpha=lora_alpha,
            target_modules=target_modules,
            lora_dropout=lora_dropout,
            bias="none",
        )
        model = get_peft_model(model, lora_config)
        if load_in_4bit:
            for name, module in model.named_modules():
                if isinstance(module, LoraLayer):
                    module.to(torch.bfloat16)
                if "norm" in name:
                    module.to(torch.float32)
                if "value_head" in name or "embed_tokens" in name:
                    if hasattr(module, "weight"):
                        module.to(torch.bfloat16)

    if init_value_head:
        if dschf is not None:
            logger.info("Initialize value_head for ZeRO-3 reward model training.")
            with deepspeed.zero.GatheredParameters([model.value_head.weight], modifier_rank=0):
                if torch.distributed.get_rank() == 0:
                    model.value_head.weight.data.normal_(mean=0.0, std=1 / (config.hidden_size + 1))
        else:
            model.value_head.weight.data.normal_(mean=0.0, std=1 / (config.hidden_size + 1))
            
    if init_prompt_head and add_prompt_head:
        if dschf is not None:
            logger.info("Initialize prompt_head for ZeRO-3 reward model training.")
            with deepspeed.zero.GatheredParameters([model.prompt_head.weight], modifier_rank=0):
                if torch.distributed.get_rank() == 0:
                    model.prompt_head.weight.data.normal_(mean=0.0, std=1 / (config.hidden_size + 1))
        else:
            model.prompt_head.weight.data.normal_(mean=0.0, std=1 / (config.hidden_size + 1))
        
    return model

def _get_general_preference_model(base_causal_model, base_llm_model, is_general_preference: bool=False, add_prompt_head: bool=False, value_head_dim: int=2, is_preference_embedding_normalized: bool = False, is_using_nonlinear_value_head: bool = False, is_using_nonlinear_prompt_gate: bool = False):  # Renamed argument
    class ValueMLP(nn.Module):
        def __init__(self, input_size, output_size):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(input_size, 1024),
                nn.ReLU(),
                nn.Linear(1024, 1024),
                nn.ReLU(),
                nn.Linear(1024, 1024),
                nn.ReLU(),
                nn.Linear(1024, output_size)
            )

        def forward(self, x):
            return self.net(x)

    class PromptMLP(nn.Module):
        def __init__(self, input_size, output_size):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(input_size, 1024),
                nn.ReLU(),
                nn.Linear(1024, 1024),
                nn.ReLU(),
                nn.Linear(1024, 1024),
                nn.ReLU(),
                nn.Linear(1024, output_size)
            )

        def forward(self, x):
            return self.net(x)

    class CustomRewardModel(base_causal_model):
        supports_gradient_checkpointing = True

        def __init__(self, config: AutoConfig):
            super().__init__(config)
            setattr(self, self.base_model_prefix, base_llm_model(config))
            if not is_general_preference:
                self.value_head = nn.Linear(config.hidden_size, 1, bias=False)
            else:
                if is_using_nonlinear_value_head:
                    self.value_head = ValueMLP(config.hidden_size, value_head_dim)
                else:
                    self.value_head = nn.Linear(config.hidden_size, value_head_dim, bias=False)
                
                if add_prompt_head:
                    if is_using_nonlinear_prompt_gate:  
                        self.prompt_head = PromptMLP(config.hidden_size, value_head_dim // 2)
                    else:
                        self.prompt_head = nn.Linear(config.hidden_size, value_head_dim // 2, bias=False)
        
            self.is_general_preference = is_general_preference    
            self.is_preference_embedding_normalized = is_preference_embedding_normalized  # Renamed attribute
            
            self.post_init()

        def custom_forward(self, input_ids, attention_mask, return_output=False, ring_attn_group=None, packed_seq_lens=None):
            """Custom forward with support for ring attention and packed sequences"""
            
            # Set position ids based on attention mask (for both packed and unpacked)
            if packed_seq_lens is None:
                # Regular processing
                position_ids = attention_mask.long().cumsum(-1) - 1
                position_ids.masked_fill_(attention_mask == 0, 1)
            else:
                # For packed sequences, reset position IDs at each sequence boundary
                position_ids = reset_position_ids(attention_mask)
                
                if ring_attn_group is not None:
                    input_ids, attention_mask, position_ids = convert_ring_attn_params(
                        input_ids, attention_mask, packed_seq_lens, ring_attn_group
                    )

            outputs = getattr(self, self.base_model_prefix)(
                input_ids, attention_mask=attention_mask, position_ids=position_ids
            )
            
            last_hidden_states = outputs["last_hidden_state"]
            values = self.value_head(last_hidden_states)

            if packed_seq_lens is not None:
                # For packed sequences, get rewards at sequence boundaries
                packed_seq_lens = torch.tensor(packed_seq_lens, device=values.device)
                eos_indices = packed_seq_lens.cumsum(dim=0) - 1
                reward = values.squeeze(0).gather(dim=0, index=eos_indices.unsqueeze(-1).expand(-1, values.size(-1)))
            else:
                # Regular processing for unpacked sequences
                eos_indices = attention_mask.size(1) - 1 - attention_mask.long().fliplr().argmax(dim=1, keepdim=True)
                reward = values.gather(dim=1, index=eos_indices.unsqueeze(-1).expand(-1, -1, values.size(-1))).squeeze(1)

            if self.is_preference_embedding_normalized:
                reward = F.normalize(reward, p=2, dim=-1)
                
            return (reward, outputs) if return_output else (reward, None)
        
        def create_skew_symmetric_block_matrix(self, dim, device, dtype, prompt_hidden_states=None):
            """
            Create a batch of skew-symmetric block matrices where each matrix is data-dependent on
            the corresponding prompt_hidden_states.
            
            Args:
            - dim: Dimension of the square matrix (must be even)
            - device: Device for the output tensor
            - dtype: Data type for the output tensor
            - prompt_hidden_states: Tensor of shape [batch_size, hidden_dim] or None
            
            Returns:
            - batch_R_matrices: Tensor of shape [batch_size, dim, dim], with skew-symmetric block entries
            """
            if not hasattr(self, 'prompt_head'):
                raise AttributeError("prompt_head is not defined. Ensure 'add_prompt_head' is set to True during initialization.")
                
            if prompt_hidden_states is None:
                # Create default skew-symmetric matrix if no prompt states provided
                batch_size = 1
                batch_R_matrices = torch.zeros((batch_size, dim, dim), device=device, dtype=dtype)
                for i in range(0, dim, 2):
                    batch_R_matrices[:, i, i + 1] = -1.0
                    batch_R_matrices[:, i + 1, i] = 1.0
                return batch_R_matrices
                
            batch_size = prompt_hidden_states.shape[0]
            hidden_dim = prompt_hidden_states.shape[1]
            
            # Ensure dim is even
            assert dim % 2 == 0, "dim must be even for skew-symmetric block generation"

            # Generate block diagonal entries
            block_values = self.prompt_head(prompt_hidden_states).view(batch_size, dim // 2)
            block_values = torch.softmax(block_values / math.sqrt(hidden_dim), dim=-1)
            block_values = block_values * block_values.shape[-1]

            # Create batch of matrices
            batch_R_matrices = torch.zeros((batch_size, dim, dim), device=device, dtype=dtype)
            
            # Fill block diagonal entries
            for i in range(0, dim, 2):
                batch_R_matrices[:, i, i + 1] = -block_values[:, i // 2]
                batch_R_matrices[:, i + 1, i] = block_values[:, i // 2]
                    
            return batch_R_matrices
                
    return CustomRewardModel