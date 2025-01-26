import argparse
import os
import pathlib

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from coding_assistant.experiments.flash_attn_optimization.flash_attn_cache import FlashAttentionCache
from coding_assistant.experiments.utils import PROMPTS


def profile_execution(model_path: pathlib.Path, prompt: str) -> None:
    print(f"Assessing specialized flash attention version performance in: {model_path}")

    tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct", trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(model_path,
                                                 trust_remote_code=True,
                                                 local_files_only=True,
                                                 attn_implementation="flash_attention_2",
                                                 torch_dtype=torch.bfloat16).cuda()
    messages = [
        {'role': 'user', 'content': prompt}
    ]
    inputs = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt").to(model.device)

    # warmup
    _ = model.generate(inputs, past_key_values=FlashAttentionCache(), max_new_tokens=2048, do_sample=False, top_k=50, top_p=0.95, num_return_sequences=1,
                       eos_token_id=tokenizer.eos_token_id)

    for layer in model.model.layers:
        layer.self_attn.reset_statistics()

    _ = model.generate(inputs, past_key_values=FlashAttentionCache(), max_new_tokens=2048, do_sample=False,
                       top_k=50, top_p=0.95, num_return_sequences=1,
                       eos_token_id=tokenizer.eos_token_id)

    total_execution_time_opt = 0
    total_executions_opt = 0

    for layer in model.model.layers:
        total_execution_time_opt += layer.self_attn.execution_time
        total_executions_opt += layer.self_attn.num_executions

    # warmup normal
    _ = model.generate(inputs, max_new_tokens=2048, do_sample=False, top_k=50, top_p=0.95, num_return_sequences=1,
                       eos_token_id=tokenizer.eos_token_id)

    for layer in model.model.layers:
        layer.self_attn.reset_statistics()

    _ = model.generate(inputs, max_new_tokens=2048, do_sample=False, top_k=50, top_p=0.95, num_return_sequences=1,
                       eos_token_id=tokenizer.eos_token_id)

    total_execution_time_normal = 0
    total_executions_normal = 0

    for layer in model.model.layers:
        total_execution_time_normal += layer.self_attn.execution_time
        total_executions_normal += layer.self_attn.num_executions

    perf_normal = total_executions_normal / total_execution_time_normal
    perf_opt = total_executions_opt / total_execution_time_opt

    print(f"The specialized flash attention version achieves a speedup of {perf_opt / perf_normal:.2f}x")


def main() -> None:
    parser = argparse.ArgumentParser(description="Count the number of selected experts per layer.")

    parser.add_argument(
        "model_path",
        type=str,
        help="Path to the directory containing the model."
    )

    parser.add_argument(
        "prompt_id",
        type=int,
        help="ID of the prompt to use (must be an integer)."
    )

    args = parser.parse_args()

    if not os.path.isdir(args.model_path):
        print(f"Error: The provided path '{args.model_path}' is not a valid directory.")
        return

    if args.prompt_id < 0:
        print(f"Error: The provided prompt ID '{args.prompt_id}' must be between 0 and {len(PROMPTS)} (exclusively).")
        return

    profile_execution(args.model_path, PROMPTS[args.prompt_id])
