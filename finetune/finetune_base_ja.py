#!/usr/bin/env python
# coding: utf-8
import os

import torch
from torch.utils.data import Dataset, random_split
from tqdm.auto import tqdm

from datasets import concatenate_datasets, DatasetDict, load_dataset
import transformers as T
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments

from rm_datasets import SFTDataset, MaskedSFTDataset, TextDataset
from templates import INPUT_PROMPT, NO_INPUT_PROMPT
from utils import load_yaml


def main(config):
    # data
    if isinstance(config["data_path"], list) and len(config["data_path"]) > 1:
        raw_datasets = []
        for data_path in config['data_path']:
            _ds = load_dataset(data_path, cache_dir=config["cache_dir"])['train']
            raw_datasets.append(_ds)
            print(f"Loaded `{data_path}`. The dataset size is {len(_ds)}")
        raw_data = concatenate_datasets(raw_datasets)
    else:
        raw_data = load_dataset(config["data_path"], cache_dir=config["cache_dir"])['train']
    print(f"Dataset size is {len(raw_data)}")

    # tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        config["tokenizer"]["tokenizer_name_or_path"], use_fast=config["tokenizer"].get("use_fast", False),
        cache_dir=config.get("cache_dir", None),
    )
    tokenizer.pad_token = tokenizer.eos_token

    # model
    model = AutoModelForCausalLM.from_pretrained(config["model_path"])
    if torch.cuda.is_available():
        model = model.cuda()
        print("Model moved to cuda")
    training_args = TrainingArguments(**config["train_args"])

    def make_text_field(example):
        # put alapaca-like formatted data point into the prompt template
        output_field_name = 'output'
        if example['input']:
            example['text'] = INPUT_PROMPT.format(
                instruction=example['instruction'], input=example['input'], response=example[output_field_name])
        else:
            example['text'] = NO_INPUT_PROMPT.format(
                instruction=example['instruction'], input=example['input'], response=example[output_field_name])
        return example

    data = raw_data.map(
        make_text_field,
        batched=False,
        remove_columns=["instruction", "input", "output", "index", "category", "id"]
    )
    print(f"Finish making text field")

    if config["trainer"] == "unmasked":
        dataset = SFTDataset(data, tokenizer)
    elif config["trainer"] == "masked":
        dataset = MaskedSFTDataset(data, tokenizer)
    elif config['trainer'] == 'text':
        cache_name = os.path.basename(training_args.output_dir)
        dataset = TextDataset(data, tokenizer, config['max_text_len'], cache_name=cache_name)
        print("Load TextDataset")

    train_size = int(config['train_size'] * len(dataset))
    train_dataset, val_dataset = random_split(dataset, [train_size, len(dataset) - train_size])

    trainer = Trainer(
        model=model, 
        args=training_args, 
        train_dataset=train_dataset,
        eval_dataset=val_dataset, 
        data_collator=lambda data: {
            'input_ids': torch.stack([f[0] for f in data]),
            'attention_mask': torch.stack([f[1] for f in data]),
            'labels': torch.stack([f[2] for f in data])
        }
    )
    print("Start training...")
    trainer.train()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str, default="configs/sample.yaml")
    args, _ = parser.parse_known_args()
    config = load_yaml(args.config_path)
    main(config)