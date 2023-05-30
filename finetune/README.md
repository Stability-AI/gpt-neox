# fine-tuning codebase for Japanese LLMs
## full fine-tuning
By running `finetune_base_ja.py`, you can finetune Japanese LLMs. Like
```
python finetune_base_ja.py --config_path configs/sample.yaml
```
By changing configuration files, you can also finetune models in different ways according to your need.

`finetune_base_ja.py` is based on [dmahan93](https://github.com/dmahan93)'s codes:
- [finetune_base.py](https://github.com/dmahan93/reward-modeling/blob/stablechat-finetuning/reward-modeling/finetune_base.py)
- [rm_dataset.py](https://github.com/dmahan93/reward-modeling/blob/stablechat-finetuning/reward-modeling/rm_datasets.py)
- [stable6B.yaml](https://github.com/dmahan93/reward-modeling/blob/stablechat-finetuning/configs/pythia/base_configs/stable6B.yaml)


## fine-tuning with LoRA
First,
```
$ git clone git@github.com:fujiki-1emon/alpaca-lora-ja.git
```
or
```
$ git clone git@github.com:tloen/alpaca-lora.git
```
and make the same change as the former repo.

Then, to finetune GPT-NeoX based model, run the fine-tuning script like
```
python finetune.py \
    --output_dir '/path/to/output/directory' \  # TODO: change here
    --data_path 'fujiki/japanese_alpaca_data' \
    --base_model '/path/to/stablelm-jp' \  # TODO: change here
    --tokenizer_name_or_path '/path/to/tokenizer' \
    --use_fast_tokenizer \
    --prompt_template_name 'japanese_alpaca_2' \
    --val_set_size 1000 \
    --train_on_inputs \
    --add_eos_token \
    --cutoff_len 1024 \
    --batch_size 128 \
    --micro_batch_size 8 \
    --num_epochs 3 \
    --learning_rate 3e-4 \
    --lora_alpha 16 \
    --lora_dropout 0.05 \
    --lora_target_modules '[query_key_value]'
```