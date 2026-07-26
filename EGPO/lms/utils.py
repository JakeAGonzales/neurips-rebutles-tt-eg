import glob
import os
import pdb
import pickle
import time
import torch
import torch.distributed as dist

from collections.abc import Mapping, Sequence
from datasets import Dataset, load_dataset
from peft import PeftModel, PeftConfig
from tqdm import tqdm
from transformers import (
    TrainerCallback,
    AutoTokenizer,
    AutoModelForCausalLM,
)
from trl.trainer.judges import DEFAULT_PAIRWISE_SYSTEM_PROMPT
from typing import List, Dict, Any, Optional

IGNORE_LABEL = -100

PAIRWISE_SYSTEM_PROMPT_FOR_CLASSIFICATION = \
    DEFAULT_PAIRWISE_SYSTEM_PROMPT.split("## Task")[0]

def pref_model_query(prompt, response0, response1, task="classification"):
    if task == "classification":
        template = PAIRWISE_SYSTEM_PROMPT_FOR_CLASSIFICATION
    elif task == "generation":
        template = DEFAULT_PAIRWISE_SYSTEM_PROMPT
    else:
        raise ValueError(f"Invalid task: {task}")
    query = template.format(
        prompt=prompt,
        response0=response0,
        response1=response1
    )

    return query

ALPACA_SYSTEM_PROMPT = \
"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{instruction}

{formatted_input}
### Response:
"""

ALPACA_INPUT_PROMPT = \
"""
### Input:
{input}
"""

def alpaca_query(instruction, input="", **kwargs):
    def format_input(input):
        if input == "":
            return ""
        else:
            return ALPACA_INPUT_PROMPT.format(input=input)
    
    query = ALPACA_SYSTEM_PROMPT.format(
        instruction=instruction,
        formatted_input=format_input(input)
    )
    return query

if "EXP_ROOT" not in os.environ:
    print("Please set the environment variable EXP_ROOT to the directory" \
          " where you want to save the model. Now saved to current directory.")
    EXP_ROOT = ""
else:
    EXP_ROOT = os.environ["EXP_ROOT"]

if "DATA_ROOT" not in os.environ:
    print("Please set the environment variable DATA_ROOT to the directory" \
          " where you want to save the dataset. Now using Huggingface default" \
          " cache.")
    DATA_ROOT = None
else:
    DATA_ROOT = os.environ["DATA_ROOT"]

if "MODEL_ROOT" not in os.environ:
    print("Please set the environment variable MODEL_ROOT to the directory" \
          " where you want to save the model. Now using Huuggingface default" \
          " cache.")
    MODEL_ROOT = None
else:
    MODEL_ROOT = os.environ["MODEL_ROOT"]

def tokenize_pref_dataset(dataset_name, tokenizer, tokenized_path, max_seq_length):
    dataset = load_dataset(dataset_name, cache_dir=DATA_ROOT)
    train_dataset = dataset["train"]

    def preprocess_dataset(dataset):
        inputs, labels = [], []
        for c, r in tqdm(zip(dataset["chosen"], dataset["rejected"]),
                            total=len(dataset["chosen"])):
            for i in range(2):
                if c[i]["role"] == "user":
                    prompt = c[i]["content"]
                if c[i]["role"] == "assistant":
                    chosen = c[i]["content"]
                
                if r[i]["role"] == "assistant":
                    rejected = r[i]["content"]
            
            inputs.append(pref_model_query(prompt,
                                            chosen,
                                            rejected,
                                            task="classification"))
            labels.append(0)

            inputs.append(pref_model_query(prompt,
                                            rejected,
                                            chosen,
                                            task="classification"))
            labels.append(1)

        dataset = Dataset.from_dict({"inputs": inputs, "labels": labels})

        print("Example data:")
        print(dataset[0])

        dataset = dataset.map(
            lambda x: tokenizer(x["inputs"],
                                truncation=True,
                                padding="max_length",
                                max_length=max_seq_length),
            batched=True,
            batch_size=5000,
            num_proc=16,
        )

        return dataset

    train_dataset = preprocess_dataset(train_dataset)
    train_dataset.save_to_disk(tokenized_path)

def tokenize_dataset(dataset_name, tokenizer, tokenized_path, max_seq_length):
    dataset = load_dataset(dataset_name, cache_dir=DATA_ROOT)
    train_dataset = dataset["train"]

    def preprocess_dataset(dataset):
        def build_msgs(messages=None, instruction="", input="", output=""):
            if messages is not None:
                msgs = messages
            else:
                msgs = [
                    # {"role": "system", "content": instruction},
                    {"role": "user", "content": instruction + input},
                ]
                if output != "":
                    msgs.append({"role": "assistant", "content": output})
            
            return msgs

        sft_msgs, nlhf_msgs = [], []
        for i in tqdm(range(len(dataset))):
            sft_msgs.append(build_msgs(**dataset[i]))
            nlhf_msgs.append(build_msgs(**{k: v
                                           for k, v in dataset[i].items()
                                               if k != "output"}))
        
        dataset = Dataset.from_dict({"sft_messages": sft_msgs,
                                     "nlhf_messages": nlhf_msgs})

        print("Example data:")
        print(dataset[0])

        # https://github.com/allenai/open-instruct/blob/main/open_instruct/finetune.py
        def encode(example):
            """
            This function encodes a single example into a format that can be used for sft training.
            Here, we assume each example has a 'messages' field. Each message in it is a dict with 'role' and 'content' fields.
            We use the `apply_chat_template` function from the tokenizer to tokenize the messages and prepare the input and label tensors.
            """
            ret = {}
            for prefix in ["sft", "nlhf"]:
                messages = example[f"{prefix}_messages"]
                if len(messages) == 0:
                    raise ValueError("messages field is empty.")
                input_ids = tokenizer.apply_chat_template(
                    conversation=messages,
                    tokenize=True,
                    return_tensors="pt",
                    padding=False,
                    truncation=True,
                    max_length=max_seq_length,
                    add_generation_prompt=False,
                )
                labels = input_ids.clone()
                # mask the non-assistant part for avoiding loss
                for message_idx, message in enumerate(messages):
                    if message["role"] != "assistant":
                        # we calculate the start index of this non-assistant message
                        if message_idx == 0:
                            message_start_idx = 0
                        else:
                            message_start_idx = tokenizer.apply_chat_template(
                                conversation=messages[:message_idx],  # here marks the end of the previous messages
                                tokenize=True,
                                return_tensors="pt",
                                padding=False,
                                truncation=True,
                                max_length=max_seq_length,
                                add_generation_prompt=False,
                            ).shape[1]
                        # next, we calculate the end index of this non-assistant message
                        if message_idx < len(messages) - 1 and messages[message_idx + 1]["role"] == "assistant":
                            # for intermediate messages that follow with an assistant message, we need to
                            # set `add_generation_prompt=True` to avoid the assistant generation prefix being included in the loss
                            # (e.g., `<|assistant|>`)
                            message_end_idx = tokenizer.apply_chat_template(
                                conversation=messages[: message_idx + 1],
                                tokenize=True,
                                return_tensors="pt",
                                padding=False,
                                truncation=True,
                                max_length=max_seq_length,
                                add_generation_prompt=True,
                            ).shape[1]
                        else:
                            # for the last message or the message that doesn't follow with an assistant message,
                            # we don't need to add the assistant generation prefix
                            message_end_idx = tokenizer.apply_chat_template(
                                conversation=messages[: message_idx + 1],
                                tokenize=True,
                                return_tensors="pt",
                                padding=False,
                                truncation=True,
                                max_length=max_seq_length,
                                add_generation_prompt=False,
                            ).shape[1]
                        # set the label to -100 for the non-assistant part
                        labels[:, message_start_idx:message_end_idx] = -100
                        if max_seq_length and message_end_idx >= max_seq_length:
                            break
                attention_mask = torch.ones_like(input_ids)
                ret.update({
                    f"{prefix}_input_ids": input_ids.flatten(),
                    f"{prefix}_labels": labels.flatten(),
                    f"{prefix}_attention_mask": attention_mask.flatten(),
                })
            
            return ret

        dataset = dataset.map(encode, num_proc=16)
        return dataset

    train_dataset = preprocess_dataset(train_dataset)
    train_dataset.save_to_disk(tokenized_path)

class SFTDataCollator:
    """
    Robust data-collator for supervised fine-tuning (SFT) of causal-LMs.

    * Pads input_ids, attention_mask, AND labels.
    * Masks label padding with ignore_index.
    * Works even if labels are shorter than input_ids.

    Args
    ----
    tokenizer : PreTrainedTokenizerBase
    pad_to_multiple_of : Optional[int]
        If given, final sequence length is padded up to the next multiple
        of this value (e.g. 8 or 16 for A100 tensor-core efficiency).
    ignore_index : int
        Value in labels that CrossEntropyLoss will ignore.  Default IGNORE_LABEL.
    """
    def __init__(
        self,
        tokenizer,
        truncate_length: Optional[int] = None,
        ignore_index: int = IGNORE_LABEL,
    ):
        self.tokenizer = tokenizer
        self.ignore_index = ignore_index
        self.truncate_length = truncate_length

        # make sure the tokenizer actually *has* a pad token
        if self.tokenizer.pad_token is None:
            # for most causal LMs you reuse EOS
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        batch = self.tokenizer.pad(
            {k: [f[k] for f in features] for k in ("input_ids", "attention_mask")},
            padding=True,
            return_tensors="pt",
        )
        max_len = batch["input_ids"].size(1)

        padded_labels = []
        pad_id = self.tokenizer.pad_token_id
        for f in features:
            lbl = torch.tensor(f["labels"], dtype=torch.long)
            if lbl.numel() < max_len:
                pad_len = max_len - lbl.numel()
                lbl = torch.nn.functional.pad(lbl, (0, pad_len), value=pad_id)
            padded_labels.append(lbl)

        labels = torch.stack(padded_labels)
        labels[labels == pad_id] = self.ignore_index
        batch["labels"] = labels

        if self.truncate_length is not None:
            for key in ["input_ids", "attention_mask", "labels"]:
                batch[key] = batch[key][:, :self.truncate_length]
        
        return batch


class AverageLossCallback(TrainerCallback):
    def __init__(self, gradient_accumulation_steps):
        self.gradient_accumulation_steps = gradient_accumulation_steps

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs and "loss" in logs:
            # Divide the logged loss by the gradient accumulation steps
            logs["avg_loss"] = logs["loss"] / self.gradient_accumulation_steps

class PushToHubCallback(TrainerCallback):
    def __init__(self, trainer, base_repo_name, hf_token, is_resume, effective_factor=1, model_card_kwargs=None):
        """
        Callback to push model to Hugging Face Hub at the end of each epoch.
        
        Args:
            trainer: The Trainer object that contains the push_to_hub method
            base_repo_name (str): Base name for the repository
            hf_token (str): Hugging Face authentication token
            is_resume (bool): Whether this is resuming from a previous training run
            effective_factor (int, optional): Factor to determine which epochs to push. Defaults to 1.
            model_card_kwargs (dict, optional): Additional arguments for the model card
        """
        self.trainer = trainer
        self.base_repo_name = base_repo_name
        self.hf_token = hf_token
        self.is_resume = is_resume
        self.effective_factor = effective_factor
        self.model_card_kwargs = model_card_kwargs or {}
        
    def on_train_begin(self, args, state, control, **kwargs):
        # Handle the case of failed push to hub in the last training
        if self.is_resume:
            self.push_to_hub(args=args, state=state)
        
    def on_epoch_end(self, args, state, control, **kwargs):
        self.push_to_hub(args=args, state=state)
    
    def push_to_hub(self, args, state, **kwargs):
        """
        Push the model to Hugging Face Hub using the trainer's push_to_hub method.
        """
        if args.local_rank != 0:
            return
            
        # Calculate epoch number
        epoch_num = 1 + (state.epoch - 1) // self.effective_factor
        repo_name = f"{self.base_repo_name}-epoch-"
        if len(repo_name) > 90:
            repo_name = repo_name[:45] + "-" + repo_name[-45:]
        repo_name += str(int(epoch_num))
                
        try:
            # Temporarily modify the hub_model_id to have per-epoch repositories
            self.trainer.args.hub_model_id = repo_name
            self.trainer.init_hf_repo(self.hf_token)
            
            # Use trainer's push_to_hub method
            self.trainer.push_to_hub(
                commit_message=f"Epoch {int(epoch_num)} checkpoint",
                blocking=True,
                token=self.hf_token,
                **self.model_card_kwargs
            )
            
            print(f"Successfully pushed model to {repo_name} on Hugging Face Hub")
            
        except Exception as e:
            print(f"Error pushing to hub: {e}")
            import traceback
            traceback.print_exc()

def mylog(s):
    print(time.strftime("%H:%M:%S"), s)

def multiply_list(arr, multiplicty):
    return [x for x in arr for _ in range(multiplicty)]

def load_model_tokenizer(model_name):
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
        
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            attn_implementation="eager",
            cache_dir=MODEL_ROOT,
        )
    except Exception as e:
        config = PeftConfig.from_pretrained(model_name)
        tokenizer = AutoTokenizer.from_pretrained(
            config.base_model_name_or_path,
            use_fast=False
        )

        model = AutoModelForCausalLM.from_pretrained(
            config.base_model_name_or_path,
            torch_dtype=torch.bfloat16,
            attn_implementation="eager",
            cache_dir=MODEL_ROOT,
        )
        model = PeftModel.from_pretrained(model, model_name)

    if not tokenizer.pad_token:
        tokenizer.pad_token = tokenizer.eos_token
    
    return model, tokenizer

def format_names(ALG, model_name, dataset_name, lora, config):
    config_str = ""
    for k in ["learning_rate", "y_yp_top_k", "y_yp_min_p"]:
        if k in config:
            config_str += f"-{config[k]}"
    exp_base_name = model_name.split("/")[-1].replace(f"-{ALG}", "") \
        + "-" + dataset_name.split("/")[-1] + f"-{ALG}" \
        + ("-lora" if lora else "") # + config_str
    run_name = exp_base_name + "-" + time.strftime("%m%d%H%M%S")

    return exp_base_name, run_name

def transform_into_chat(dataset, max_data_len=300, max_data_num=30000):
    def transform_example(example):
        if "prompt" in example:
            return {
                "prompt": [
                    {"role": "user", "content": example["prompt"]},
                ]
            }
        elif "context_messages" in example:
            return {
                "prompt": example["context_messages"]
            }
        else:
            raise NotImplementedError(
                "The dataset does not have 'prompt' or 'context_messages' field."
            )
    dataset = dataset.map(transform_example)
    dataset = dataset.filter(lambda x: len(x["prompt"]) == 1 and len(x["prompt"][0]["content"]) < max_data_len)

    print(f"{len(dataset)} examples after filtering.")

    if max_data_num is not None:
        if max_data_num > 0:
            dataset = dataset.select(range(min(max_data_num, len(dataset))))
        else:
            dataset = dataset.select(range(max(0, len(dataset) + max_data_num),
                                           len(dataset)))
    
    return dataset

def get_latest_checkpoint(output_dir):
    checkpoints = glob.glob(os.path.join(output_dir, "checkpoint-*"))
    if checkpoints:
        # Sort checkpoints by step number
        checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[-1]))
        return checkpoints[-1]
    
    return None

def broadcast_object(obj, src=0, group=None):
    """
    Broadcasts a Python object from source to all processes in group.
    
    Args:
        obj: The object to broadcast (can be None on non-source ranks)
        src: Source rank
        group: Process group
        
    Returns:
        The broadcast object
    """
    if dist.get_rank() == src:
        # Serialize the object on source rank
        buffer = pickle.dumps(obj)
        storage = torch.ByteStorage.from_buffer(buffer)
        tensor = torch.ByteTensor(storage).to("cuda")
        # Broadcast the tensor size
        local_size = torch.LongTensor([tensor.numel()]).to("cuda")
        dist.broadcast(local_size, src, group=group)
        # Broadcast the tensor
        dist.broadcast(tensor, src, group=group)
    else:
        # Receive tensor size
        local_size = torch.LongTensor([0]).to("cuda")
        dist.broadcast(local_size, src, group=group)
        # Receive tensor
        tensor = torch.ByteTensor(local_size.item()).to("cuda")
        dist.broadcast(tensor, src, group=group)
        # Deserialize the tensor to get the object
        buffer = tensor.cpu().numpy().tobytes()
        obj = pickle.loads(buffer)
    
    return obj

def move_state_dict_to_cpu_backup(data):
    """
    Recursively traverses a state dictionary (or part of it).
    Clones, detaches, and moves any found torch.Tensor to the CPU.
    Handles nested dicts, lists, and tuples.
    """
    if isinstance(data, torch.Tensor):
        # The core operation for backup: clone, detach, move to CPU
        return data.clone().detach().cpu()
    elif isinstance(data, Mapping): # Checks for dict or dict-like objects
        # Recursively process dictionary values
        return {k: move_state_dict_to_cpu_backup(v) for k, v in data.items()}
    elif isinstance(data, Sequence) and not isinstance(data, (str, bytes)): # Checks for list, tuple, etc., but not strings/bytes
        # Recursively process sequence items and preserve sequence type (list or tuple)
        return type(data)(move_state_dict_to_cpu_backup(item) for item in data)
    else:
        # Return non-tensor, non-collection types as is (int, float, str, bool, None, etc.)
        return data

# --- Helper Function: Move state dict tensors to a target device for restore ---
def move_state_dict_to_device(data, device):
    """
    Recursively traverses a state dictionary (or part of it).
    Moves any found torch.Tensor to the specified target device.
    Handles nested dicts, lists, and tuples.
    """
    if isinstance(data, torch.Tensor):
        # The core operation for restore: move to target device
        return data.to(device)
    elif isinstance(data, Mapping):
        # Recursively process dictionary values
        return {k: move_state_dict_to_device(v, device) for k, v in data.items()}
    elif isinstance(data, Sequence) and not isinstance(data, (str, bytes)):
        # Recursively process sequence items and preserve sequence type
        return type(data)(move_state_dict_to_device(item, device) for item in data)
    else:
        # Return non-tensor, non-collection types as is
        return data