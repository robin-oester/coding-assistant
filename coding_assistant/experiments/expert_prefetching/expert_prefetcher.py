import argparse
import os
import pathlib

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from coding_assistant.experiments.utils import PROMPTS


def profile_execution(model_path: pathlib.Path, prompt: str, show_output: bool) -> None:
    print(f"Assessing pre-attention expert selection strategy on model in folder: {model_path}")

    tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct", trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(model_path,
                                                 trust_remote_code=True,
                                                 local_files_only=True,
                                                 attn_implementation="eager",
                                                 torch_dtype=torch.bfloat16).cuda()
    messages = [
        {'role': 'user', 'content': prompt}
    ]
    inputs = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt").to(model.device)

    outputs = model.generate(inputs, max_new_tokens=2048, do_sample=False, top_k=50, top_p=0.95, num_return_sequences=1,
                             eos_token_id=tokenizer.eos_token_id)

    total_experts_selected = 0
    total_overlap = 0

    for layer in model.model.layers:
        total_experts_selected += layer.experts_selected
        total_overlap += layer.amount_overlap

    accuracy = total_overlap / total_experts_selected

    print(f"The predictive performance of the pre-attention state amounts to {accuracy * 100:.2f}%")

    if show_output:
        print("The model generated the following response:")
        print(tokenizer.decode(outputs[0][len(inputs[0]):], skip_special_tokens=True))


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

    parser.add_argument(
        "--show",
        action="store_true",
        help="Shows the generated output"
    )

    args = parser.parse_args()

    if not os.path.isdir(args.model_path):
        print(f"Error: The provided path '{args.model_path}' is not a valid directory.")
        return

    if args.prompt_id < 0:
        print(f"Error: The provided prompt ID '{args.prompt_id}' must be between 0 and {len(PROMPTS)} (exclusively).")
        return

    profile_execution(args.model_path, PROMPTS[args.prompt_id], args.show)
