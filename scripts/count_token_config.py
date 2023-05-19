"""
Count the number of tokens in the data paths specified by the NeoX config file.

Usage:
python count_token_config.py --config /path/to/config --tokenizer EleutherAI/gpt-neox-20b --num_samples 20
"""
import argparse
import yaml
import json
import os

from megatron.data.indexed_dataset import MMapIndexedDataset


def write_json(update: dict, path: str):
    """Update the contents of the json file at `path`"""
    if os.path.exists(path):
        with open(path, "r") as f:
            contents = json.load(f)
        contents.update(update)
    else:
        contents = update
    with open(path, "w+") as f:
        json.dump(contents, f, indent=4, sort_keys=True)


parser = argparse.ArgumentParser()
parser.add_argument("--config", type=str)
parser.add_argument("--output_path", type=str)
parser.add_argument("--tokenizer", type=str, default="EleutherAI/gpt-neox-20b")
args = parser.parse_args()


# TODO: Data is already tokenizer no need to instantiate the tokenizer at this time.
# tokenizer = transformers.AutoTokenizer.from_pretrained(args.tokenizer)
print(f"Tokenizer: {args.tokenizer}")


config_path = args.config
config = yaml.load(open(config_path, "r"), Loader=yaml.FullLoader)
data_paths = config['train-data-paths']
print(f"Counting from data-paths: {data_paths}")


# Use the config base name as the output file path
output_path = os.path.join(
    args.output_path,f"{os.path.basename(config_path)[:-4]}_token_count.json")
print(f"Output path: {output_path}")
assert not os.path.exists(
    output_path), f"Output path `{output_path}` already exists"


keys = ["num_docs", "num_tokens"]
path_to_num_tokens = {p: {k: 0 for k in keys} for p in data_paths}
for p in data_paths:
    try:
        print(f"Loading {p}")
        dataset = MMapIndexedDataset(p)
    except:
        print(f"Could not load {p}")
        continue
    path_to_num_tokens[p]['num_docs'] = len(dataset)
    path_to_num_tokens[p]['num_tokens'] = sum(
        [len(dataset[i]) for i in range(len(dataset))])
    print(f"\nPath: {p}")
    print(f"Num docs: {path_to_num_tokens[p]['num_docs']}")
    print(f"Num tokens: {path_to_num_tokens[p]['num_tokens']}\n")
    write_json(path_to_num_tokens, output_path)
