from random import randrange
import transformers
from transformers import AutoTokenizer, T5ForConditionalGeneration
import sys
import torch
import peft
from peft import (
    LoraConfig,
    get_peft_model,
    get_peft_model_state_dict,
)
from peft import prepare_model_for_int8_training
from datasets import load_dataset

# optimized for RTX 4090. for larger GPUs, increase some of these?
MICRO_BATCH_SIZE = 8  # this could actually be 5 but i like powers of 2
BATCH_SIZE = 128
GRADIENT_ACCUMULATION_STEPS = BATCH_SIZE // MICRO_BATCH_SIZE
EPOCHS = 10  # we don't always need 3 tbh
LEARNING_RATE = 3e-3  # the Karpathy constant x10
INPUT_CUTOFF_LEN = 255  # 256 accounts for about 96% of the data
OUTPUT_CUTOFF_LEN = 511
WITH_LORA = True
LORA_R = 8
LORA_ALPHA = 16
LORA_DROPOUT = 0.05
VAL_SET_SIZE = 2000
TARGET_MODULES = [
    "q",
    "v",
]
LABEL_PAD_TOKEN_ID = -100

BASE_MODEL = "google/flan-t5-base"
DATA_PATH = "alpaca_data_cleaned.json"
OUTPUT_DIR = f"models/{DATA_PATH.split('_')[0]}_{BASE_MODEL.split('/')[1].replace('-', '_')}_2xo"
if WITH_LORA:
    OUTPUT_DIR += "_lora"

if torch.cuda.is_available():
    DEVICE = "cuda"
else:
    DEVICE = "cpu"

try:
    if torch.backends.mps.is_available():
        DEVICE = "mps"
except:
    pass


def generate_prompt(data_point):
    # sorry about the formatting disaster gotta move fast
    if data_point["input"]:
        return f"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

    ### Instruction:
            {data_point["instruction"]}

    ### Input:
            {data_point["input"]}
    """
    else:
        return f"""Below is an instruction that describes a task. Write a response that appropriately completes the request.

    ### Instruction:
            {data_point["instruction"]}
    """


def generate_response(data_point):
    return f"""{data_point["output"]}"""


def print_some(dataset):
    sample = dataset[randrange(len(dataset))]
    print(f"instruction: \n{sample['instruction']}\n---------------")
    print(f"input: \n{sample['input']}\n---------------")
    print(f"output: \n{sample['output']}\n---------------")


def main():
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, add_eos_token=True)

    if DEVICE == "cuda":
        model = T5ForConditionalGeneration.from_pretrained(BASE_MODEL)
        # model = prepare_model_for_int8_training(model)
    else:
        model = T5ForConditionalGeneration.from_pretrained(BASE_MODEL)

    if WITH_LORA:
        config = LoraConfig(
            r=LORA_R,
            lora_alpha=LORA_ALPHA,
            target_modules=TARGET_MODULES,
            lora_dropout=LORA_DROPOUT,
            bias="none",
            task_type=peft.TaskType.SEQ_2_SEQ_LM
        )
        model = get_peft_model(model, config)

    def tokenize(prompt, response):

        result = tokenizer(
            prompt,
            truncation=True,
            max_length=INPUT_CUTOFF_LEN + 1,
            padding="max_length",
        )
        labels = tokenizer(
            response,
            truncation=True,
            max_length=OUTPUT_CUTOFF_LEN + 1,
            padding="max_length",
        )
        labels["input_ids"] = [
            (l if l != tokenizer.pad_token_id else LABEL_PAD_TOKEN_ID) for l in labels["input_ids"]
        ]

        return {
            "input_ids": result["input_ids"][:-1],
            "attention_mask": result["attention_mask"][:-1],
            "labels": labels["input_ids"][:-1],
        }

    def generate_and_tokenize_prompt(data_point):
        prompt = generate_prompt(data_point)
        response = generate_response(data_point)
        return tokenize(prompt, response)

    def load_alpaca_dataset():
        data = load_dataset("json", data_files=DATA_PATH)
        train_val = data["train"].train_test_split(
            test_size=VAL_SET_SIZE, shuffle=True, seed=42
        )

        train_data = train_val["train"].shuffle().map(generate_and_tokenize_prompt)
        val_data = train_val["test"].shuffle().map(generate_and_tokenize_prompt)

        return train_data, val_data

    train_data, val_data = load_alpaca_dataset()

    print(f"Train dataset size: {len(train_data)}")
    print(f"Test dataset size: {len(val_data)}")

    print_some(train_data)

    trainer = transformers.Seq2SeqTrainer(
        model=model,
        train_dataset=train_data,
        eval_dataset=val_data,
        args=transformers.Seq2SeqTrainingArguments(
            per_device_train_batch_size=MICRO_BATCH_SIZE,
            gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
            warmup_steps=100,
            num_train_epochs=EPOCHS,
            learning_rate=LEARNING_RATE,
            use_mps_device=(DEVICE == "mps"),
            logging_steps=20,
            evaluation_strategy="steps" if VAL_SET_SIZE > 0 else "no",
            save_strategy="steps",
            eval_steps=200 if VAL_SET_SIZE > 0 else None,
            save_steps=200,
            output_dir=OUTPUT_DIR,
            save_total_limit=3,
            load_best_model_at_end=True if VAL_SET_SIZE > 0 else False,
            fp16=False,  # Loss is 0 with True
            logging_dir="logs/",
            report_to=["tensorboard"],
        ),
        data_collator=transformers.DataCollatorForSeq2Seq(
            tokenizer,
            label_pad_token_id=LABEL_PAD_TOKEN_ID,
            pad_to_multiple_of=8
        )
    )
    model.config.use_cache = False

    old_state_dict = model.state_dict
    model.state_dict = (
        lambda self, *_, **__: get_peft_model_state_dict(self, old_state_dict())
    ).__get__(model, type(model))
    model = torch.compile(model)

    trainer.train()

    model.save_pretrained(OUTPUT_DIR)

    print("\n If there's a warning about missing keys above, please disregard :)")
