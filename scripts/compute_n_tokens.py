"""
Compute number of tokens in wikipedia by a tokenizer
1. Firstly, prepare wikipedia by following https://github.com/rinnakk/japanese-pretrained-models#data-construction-and-model-training.

"""
import os
import json
from tqdm import tqdm
import glob
from transformers import AutoTokenizer, T5Tokenizer


NOVELAI = "/fsx/home-mkshing/models/novelai-tokenizer"


def convert_novelai_to_hf():
    # https://github.com/NovelAI/novelai-tokenizer
    novel_ai_sp = "novelai.model"
    tokenizer = T5Tokenizer(
        novel_ai_sp,
        unk_token="<|unknown|>",
        eos_token="<|endoftext|>",
        pad_token="<|pad|>",
        extra_ids=0,
    )
    tokenizer.save_pretrained("novelai-tokenizer")
    # make sure to load by T5Tokenizer
    # AutoTokenizer calls T5TokenizerFast which is based on unigram. https://huggingface.co/docs/transformers/model_doc/t5#transformers.T5TokenizerFast
    # NovelAI Tokenizer is based on BPE. 
    tokenizer = T5Tokenizer.from_pretrained("novelai-tokenizer")

def main():
    model_id = sys.argv[1] # path to model file or name for log
    mode = sys.argv[2] # mecab, novelai, rinna, or other
    input_data = sys.argv[3] # en or ja
    print(f"model_id: {model_id}")
    print("args:", model_id, mode, input_data)
    if input_data  == "en":
        wiki_files = glob.glob("/fsx/home-polm/data/en-data/*2.txt")
    else:
        wiki_files = glob.glob("/fsx/home-polm/data/doc_data/*.txt")
    MAX_SAVE_LINES_FOR_UNK = 5000
    OUTPUT_DIR = "logs"
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    outfile = os.path.join(OUTPUT_DIR, input_data + model_id.replace("/", "-")+".json")
    if mode == "mecab":
        mecab_args = {"mecab_dic": "unidic_lite"}
        tokenizer = BertJapaneseTokenizer(vocab_file=f"{model_id}.vocab", spm_file=f"{model_id}.model", word_tokenizer_type="mecab", mecab_kwargs=mecab_args, subword_tokenizer_type="sentencepiece")
    elif mode == "novelai":
        model_path = "/fsx/home-polm/gpt-neox/scripts/novelai-tokenizer"
        tokenizer = T5Tokenizer.from_pretrained(model_path)
    elif mode == "rinna":
        model_path = "/fsx/home-polm/tokenizer-train/rinna-huggingface"
        tokenizer = T5Tokenizer.from_pretrained(model_path)
    else:
        model_path = model_id
        tokenizer = T5Tokenizer(
            model_path,
            unk_token="<|unknown|>",
            eos_token="<|endoftext|>",
            pad_token="<|pad|>",
            extra_ids=0,
        )

    summary = {"class": tokenizer.__class__.__name__, "vocab_size": len(tokenizer), "n_tokens": 0, "n_unk": 0, "unk_lines": []}
    for file in tqdm(wiki_files):
        with open(file,  "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                input_ids = tokenizer.encode(line)
                if mode == "mecab":
                    # since this is a bert tokenizer, remove first/last token
                    input_ids = input_ids[1:-1]
                else:
                    # in the T5Tokenizer, an end-of-string marker is added to every input
                    input_ids = input_ids[:-1]
                summary['n_tokens'] += len(input_ids)
                unk_count = input_ids.count(tokenizer.unk_token_id)
                if unk_count > 0:
                    summary['n_unk'] += unk_count
                    if len(summary['unk_lines']) <= MAX_SAVE_LINES_FOR_UNK:
                        summary['unk_lines'].append(line)
    with open(outfile, "w", encoding="utf-8-sig") as fw:
        fw.write(json.dumps(summary, indent=4))
