import os
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    EarlyStoppingCallback
)
from peft import LoraConfig, prepare_model_for_kbit_training
from trl import SFTTrainer
import warnings
import trl

warnings.filterwarnings("ignore")
print(f"[INFO] TRL Version: {trl.__version__}")

# ==========================================
# 1. CONFIGURATION
# ==========================================
MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"
NEW_MODEL_NAME = "Llama-3.1-8B-PII-Multilingual-Best"
TRAIN_FILE = "master_train.jsonl"
OUTPUT_DIR = "./results_qlora"

# ==========================================
# 2. PREPARE DATA
# ==========================================
print(f"[INFO] Loading Master Dataset: {TRAIN_FILE}...")
full_dataset = load_dataset("json", data_files=TRAIN_FILE, split="train")

# Split 90/10
dataset_split = full_dataset.train_test_split(test_size=0.1, seed=42)
train_dataset = dataset_split["train"]
eval_dataset = dataset_split["test"]

print(f"[INFO] Training Rows: {len(train_dataset)}")
print(f"[INFO] Evaluation Rows: {len(eval_dataset)}")

# ==========================================
# 3. LOAD MODEL
# ==========================================
print("[INFO] Loading LLaMA 3.1 in 4-bit...")
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=False,
)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True
)
model.gradient_checkpointing_enable()
model = prepare_model_for_kbit_training(model)

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

# ==========================================
# 4. LORA CONFIG
# ==========================================
peft_config = LoraConfig(
    lora_alpha=16,
    lora_dropout=0.1,
    r=16,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
)

# ==========================================
# 5. TRAINING CONFIG (Standard Arguments Only)
# ==========================================
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=3,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    gradient_accumulation_steps=2,
    optim="paged_adamw_32bit",
    learning_rate=2e-4,
    weight_decay=0.001,
    fp16=True,
    bf16=False,
    max_grad_norm=0.3,
    warmup_ratio=0.03,
    group_by_length=True,
    lr_scheduler_type="constant",
    report_to="none",
    
    # The updated keyword for transformers 4.41+
    eval_strategy="steps",
    eval_steps=50,
    save_strategy="steps",
    save_steps=50,
    logging_steps=25,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False
)

# ==========================================
# 6. INITIALIZE TRAINER (BAREBONES)
# ==========================================
# We strip ALL optional config args that might crash.
# We rely on TRL auto-detecting the "text" column.

trainer = SFTTrainer(
    model=model,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    peft_config=peft_config,
    processing_class=tokenizer, # The new name for 'tokenizer'
    args=training_args,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
)

# ==========================================
# 7. EXECUTE
# ==========================================
print("\n[START] Training Loop Initiated...")
trainer.train()

print("\n[INFO] Saving Model...")
trainer.model.save_pretrained(NEW_MODEL_NAME)
tokenizer.save_pretrained(NEW_MODEL_NAME)
print(f"[SUCCESS] Saved to: {NEW_MODEL_NAME}")
