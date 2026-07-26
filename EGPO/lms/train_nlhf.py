import argparse
import deepspeed
import json
import os
import pdb
import torch

from datasets import load_dataset
from peft import get_peft_model

from configs import server_address, lora_config, get_ds_config
from judges.pair_judge import PairJudge
from judges.local_server_judge import LocalServerJudge
from trainers.extragradient_config import ExtragradientConfig
from trainers.extragradient_trainer import ExtragradientTrainer
from utils import (
    EXP_ROOT,
    DATA_ROOT,
    AverageLossCallback,
    PushToHubCallback,
    load_model_tokenizer,
    format_names,
    transform_into_chat,
    get_latest_checkpoint,
)

ALG_MAPPING = {
    "oipo1": "OnlineIPO1",
    "oipo2": "OnlineIPO2",
    "nmd": "NashMD",
    "nmdpg": "NashMDPG",
    "eg": "Extragradient",
}

argparser = argparse.ArgumentParser()
argparser.add_argument(
    "--alg",
    choices=list(ALG_MAPPING.keys()),
    required=True,
    help="Which algorithm to run, choices are: " \
        + ", ".join(list(ALG_MAPPING.keys())),
)
argparser.add_argument("--lora", action="store_true")
argparser.add_argument("--epochs", type=int, default=10)
argparser.add_argument("--lr", type=float, default=5e-7)
argparser.add_argument("--y_yp_mixture_coef", type=float, default=0)
argparser.add_argument("--y_yp_temperature", type=float, default=2.0)
argparser.add_argument("--y_yp_top_k", type=int, default=10)
argparser.add_argument("--y_yp_min_p", type=float, default=0.0)
argparser.add_argument("--seed", type=int, default=42)
argparser.add_argument("--local_rank", type=int, default=0)
argparser.add_argument("--load_dir", type=str, default=None)

args = argparser.parse_args()

deepspeed.init_distributed()
local_rank = args.local_rank
torch.cuda.set_device(local_rank)
world = deepspeed.comm.get_world_size()

EXP_ROOT = os.path.join(EXP_ROOT, "nlhf")
os.environ["WANDB_PROJECT"] = "nlhf"

alg = args.alg.lower()

num_gpus = torch.cuda.device_count()

batch_size = 64
micro_batch_size = 8 if args.lora else 1
gen_micro_batch_size = min(2 * micro_batch_size, batch_size // num_gpus)
samples_per_prompt = 1
prefix_chunk_num = 1
gradient_accumulation_steps = batch_size // (micro_batch_size * num_gpus)

estimate_extra_grad = (alg == "eg")
effective_factor = 2 if alg == "eg" else 1

if alg == "oipo1":
    additional_config = {
        "y_yp_mixture_coef": args.y_yp_mixture_coef,
        "y_yp_temperature": args.y_yp_temperature,
        "y_yp_top_k": args.y_yp_top_k,
        "y_yp_min_p": args.y_yp_min_p,
    }
elif alg == "oipo2":
    additional_config = {
        "y_yp_mixture_coef": 0,
    }
elif alg == "nmd":
    additional_config = {
        "y_yp_mixture_coef": args.y_yp_mixture_coef,
        "y_yp_temperature": args.y_yp_temperature,
        "y_yp_top_k": args.y_yp_top_k,
        "y_yp_min_p": args.y_yp_min_p,
        "ypp_mixture_coef": 0.125,
    }
elif alg == "nmdpg":
    additional_config = {
        "y_yp_mixture_coef": args.y_yp_mixture_coef,
        "y_yp_temperature": args.y_yp_temperature,
        "y_yp_top_k": args.y_yp_top_k,
        "y_yp_min_p": args.y_yp_min_p,
        "mixture_coef": 0.125,
    }
elif alg == "eg":
    additional_config = {
        "y_yp_mixture_coef": args.y_yp_mixture_coef,
        "y_yp_temperature": args.y_yp_temperature,
        "y_yp_top_k": args.y_yp_top_k,
        "y_yp_min_p": args.y_yp_min_p,
    }
    
model_name = "vectorzhou/gemma-2-2b-it-alpaca-cleaned-SFT"
# model_name = "vectorzhou/Qwen2.5-1.5B-Instruct-SFT-OpenHermes-2.5-Standard-SFT"
dataset_name = "PKU-Alignment/PKU-SafeRLHF"
# dataset_name = "OpenRLHF/prompt-collection-v0.1"

config = {
    "num_train_epochs": args.epochs * effective_factor,
    "per_device_train_batch_size": micro_batch_size,
    "gradient_accumulation_steps": gradient_accumulation_steps,
    "per_device_generate_batch_size": gen_micro_batch_size,
    "samples_per_prompt": samples_per_prompt,
    "learning_rate": args.lr,
    "weight_decay": 0.01,
    "bf16": True,
    "warmup_steps": 1000,
    "beta": 0.1,
    "estimate_extra_grad": estimate_extra_grad,
    "prefix_chunk_num": prefix_chunk_num,
    "seed": args.seed,
    **additional_config,
}

ds_config = get_ds_config(
    batch_size=batch_size,
    micro_batch_size=micro_batch_size,
    gradient_accumulation_steps=gradient_accumulation_steps,
    learning_rate=args.lr,
)
ds_config.pop("optimizer", None)
ds_config.pop("scheduler", None)
if local_rank == 0:
    exp_base_name, run_name = format_names(ALG=ALG_MAPPING[args.alg.lower()],
                                           model_name=model_name,
                                           dataset_name=dataset_name,
                                           lora=args.lora,
                                           config=config)
    msg_holder = [exp_base_name, run_name]
else:
    msg_holder = [None, None]

deepspeed.comm.barrier()
torch.distributed.broadcast_object_list(msg_holder, src=0)
exp_base_name, run_name = msg_holder

pref_model_name = "vectorzhou/gemma-2-2b-it-preference_dataset_mixture2_and_safe_pku-Preference"
judge = PairJudge(pref_model_name)

model, tokenizer = load_model_tokenizer(model_name)
if args.lora:
    model = get_peft_model(model, lora_config)

dataset = load_dataset(dataset_name, cache_dir=DATA_ROOT)
train_dataset = transform_into_chat(dataset["train"])

# Set the output directory
if args.load_dir:
    if args.load_dir[-1] == "/":
        args.load_dir = args.load_dir[:-1]
    run_name = args.load_dir.split("/")[-1]
    exp_base_name = run_name[:run_name.rfind("-")]
output_dir = os.path.join(EXP_ROOT, run_name)

# Define training configuration
training_args = ExtragradientConfig(
    run_name=run_name,
    output_dir=os.path.join(EXP_ROOT, run_name),
    **config,
    save_steps=0.1,
    save_total_limit=1,
    logging_steps=10,
    logging_dir='./logs',
    report_to="none",
    deepspeed=json.dumps(ds_config),
)

hf_username = os.environ.get("HF_USR_NAME", "")
hf_token = os.environ.get("HF_TOKEN", "")
hub_path = hf_username + "/" + run_name

# Determine checkpoint to resume from
resume_checkpoint = get_latest_checkpoint(output_dir)
if args.local_rank == 0:
    print(f"Resuming from {resume_checkpoint}")

# Initialize trainer
trainer = ExtragradientTrainer(
    model=model,
    judge=judge,
    args=training_args,
    processing_class=tokenizer,
    train_dataset=train_dataset,
    callbacks=[AverageLossCallback(gradient_accumulation_steps)],
)

# Add the callback to the trainer
# (PushToHubCallback disabled — HF_USR_NAME not set)

# Train the model
trainer.train(resume_from_checkpoint=resume_checkpoint)
