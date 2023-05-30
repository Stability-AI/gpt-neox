"""
This script calculates the number of tokens to train on for a given DeepSpeed curriculum
setup as specified in the NeoX config file.

Usage:
python scripts/seqlen_warmup_tokens.py --config <path-to-config> --nodes <num-nodes>
"""
import argparse
import yaml
import numpy as np


def fixed_linear_seqlen_warmup_schedule(
    start_seqlen: int = 64,
    end_seqlen: int = 2048,
    total_steps: int = 20_000,
    step_size: int = 8  # For GPU efficiency
):
    """
    Linear warmup schedule from Li et al. The Stability-Efficiency Dilemma: Investigating
    Sequence Length Warmup for Training GPT Models https://openreview.net/pdf?id=JpZ5du_Kdh
    as used in DeepSpeed
    """
    seqlen_schedule = np.array([0] * total_steps)
    for t in range(0, total_steps):
        seqlen_schedule[t] = \
            start_seqlen + (end_seqlen - start_seqlen) * min(t / total_steps, 1)
        seqlen_schedule[t] = seqlen_schedule[t] - (seqlen_schedule[t] % step_size)
        seqlen_schedule[t] = int(seqlen_schedule[t])
    return seqlen_schedule


def token_count_with_seqlen_warmup(
    seqlen: int,
    seqlen_schedule: np.array,
    rest_steps: int,
    effective_batch_size: int = 2048,
):
    """
    This function calculates the total number of tokens to for training with the
    given warmup schedule and rest of steps..

    Args:
        rest_steps: The number of steps to train on after warmup
    """
    warmup_steps = len(seqlen_schedule)
    warmup_tokens = np.sum(seqlen_schedule * effective_batch_size)
    rest_tokens = rest_steps * (effective_batch_size * seqlen)
    total_tokens = warmup_tokens + rest_tokens
    return dict(
        warmup_steps=warmup_steps,
        warmup_tokens=warmup_tokens,
        rest_steps=rest_steps,
        rest_tokens=rest_tokens,
        total_tokens=total_tokens,
    )


def effective_batch_size(mbs, gas, num_gpus, tp, pp):
    return mbs * gas * (num_gpus // (tp * pp))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str)
    parser.add_argument("--nodes", type=int, default=1)
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    assert "curriculum_learning" in config and config['curriculum_learning']['enabled'], \
        "Set the `curriculum_learning` field in your config"
    assert config['curriculum_learning']['schedule_type'] == "fixed_linear", \
        "Only `fixed_linear` curriculum is supported at this time"

    curriculum = config['curriculum_learning']
    schedule = fixed_linear_seqlen_warmup_schedule(
        start_seqlen=curriculum['min_difficulty'],
        end_seqlen=curriculum['max_difficulty'],
        total_steps=curriculum['schedule_config']['total_curriculum_step'],
        step_size=curriculum['schedule_config']['difficulty_step']
    )
    ebs = effective_batch_size(
        mbs=config["train_micro_batch_size_per_gpu"],
        gas=config["gradient_accumulation_steps"],
        num_gpus=args.nodes * 8,
        tp=config["model-parallel-size"],
        pp=config["pipe-parallel-size"],
    )
    count_info = token_count_with_seqlen_warmup(
        seqlen=curriculum['max_difficulty'],
        seqlen_schedule=schedule,
        rest_steps=config['train-iters'] - curriculum['schedule_config']['total_curriculum_step'],
        effective_batch_size=ebs,
    )

    print(f"{'='*32}")
    print(f"num_gpus: {args.nodes * 8}")
    print(f"effective batch size: {ebs}")
    print(f"{'='*32}")
    print(f"warmup steps: {count_info['warmup_steps']:,}")
    print(f"warmup tokens: {count_info['warmup_tokens']:,}")
    print(f"{'='*32}")
    print(f"rest steps: {count_info['rest_steps']:,}")
    print(f"rest tokens: {count_info['rest_tokens']:,}")
    print(f"{'='*32}")
    print(f"total steps: {count_info['warmup_steps'] + count_info['rest_steps']:,}")
    print(f"total tokens: {count_info['total_tokens']:,}")
    print(f"{'='*32}")