import argparse
import os
import pathlib
from typing import Optional

import numpy as np
import torch
from matplotlib import pyplot as plt
from scipy.stats import chisquare
from transformers import AutoTokenizer, AutoModelForCausalLM

from coding_assistant.experiments.utils import NUM_EXPERTS, NUM_LAYERS, PROMPTS


def chi_square_test(data: np.ndarray):
    draws_per_experiment = np.prod(data.shape)

    observed_frequencies = np.bincount(data.flatten(), minlength=NUM_EXPERTS)

    # Perform chi-square test
    chi2_stat, p_value = chisquare(observed_frequencies, f_exp=[draws_per_experiment / NUM_EXPERTS] * NUM_EXPERTS)

    # Print results
    print(f"Chi-square statistic: {chi2_stat}")
    print(f"P-value: {p_value}")

    # Check if the null hypothesis (uniform distribution) can be rejected at alpha = 0.05
    alpha = 0.05
    if p_value < alpha:
        print("Reject the null hypothesis: The distribution is significantly different from uniform.")
    else:
        print("Fail to reject the null hypothesis: The distribution does not significantly differ from uniform.")


def generate_layer_distribution(data: np.ndarray, layer_id: Optional[int], perform_test: bool):
    considered_data = data if layer_id is None else data[layer_id]

    if layer_id is None:
        print(f"Generate layer distribution for all layers")
    else:
        print(f"Generate layer distribution for layer {layer_id}")

    if perform_test:
        print("Perform chi-square test")
        chi_square_test(considered_data)

    counts_per_expert = np.bincount(considered_data.flatten(), minlength=NUM_EXPERTS)

    # Plotting
    plt.figure(figsize=(10, 6))  # Set figure size
    experts = np.arange(len(counts_per_expert))  # Create expert indices

    plt.bar(experts, counts_per_expert, color='skyblue', edgecolor='black', zorder=3)

    # Beautify the plot
    # plt.title("Distribution of Counts Per Expert", fontsize=16)
    plt.xlabel("Expert ID", fontsize=14)
    plt.ylabel("Count", fontsize=14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)

    plt.grid(axis='y', linestyle='--', alpha=0.7, zorder=0)

    # Show the plot
    plt.tight_layout()
    plt.savefig(f"expert_distribution_{layer_id}.png", format="png", dpi=300)
    plt.show()


def generate_heatmap(data: np.ndarray):
    print("Generating heat map...")

    num_layers = data.shape[0]  # Number of layers
    num_tokens = data.shape[1]  # Number of tokens

    # Prepare heatmap data
    heatmap_data = np.zeros((NUM_EXPERTS, num_tokens))

    for token in range(num_tokens):
        for layer in range(num_layers):
            selected_experts = data[layer, token]
            for expert in selected_experts:
                heatmap_data[expert, token] += 1

    # Plot heatmap
    fig, ax = plt.subplots(figsize=(15, 8))
    im = ax.imshow(heatmap_data, aspect='auto', cmap='viridis', origin='lower')

    # Customizations
    # ax.set_title("Selected Experts for Tokens", fontsize=16)
    ax.set_xlabel("Generated Token", fontsize=12)
    ax.set_ylabel("Expert ID", fontsize=12)
    ax.set_xticks(np.arange(0, num_tokens, 50))  # Tokens from 1 to 300
    ax.set_yticks(np.arange(0, NUM_EXPERTS, 10))  # Expert IDs

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("Number of Layers Selecting Expert", fontsize=12)

    plt.tight_layout()
    plt.savefig("counted_experts.png", format="png", dpi=300)
    plt.clf()


def profile_execution(model_path: pathlib.Path, prompt: str) -> np.ndarray:
    print(f"Running expert counting on model in folder: {model_path}")

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

    _ = model.generate(inputs, max_new_tokens=2048, do_sample=False, top_k=50, top_p=0.95, num_return_sequences=1,
                       eos_token_id=tokenizer.eos_token_id)

    num_tokens = model.model.layers[1].mlp.gate.expert_count.shape[0]
    selected_experts = torch.zeros(NUM_LAYERS, num_tokens, 6, device="cuda")

    for idx, layer in enumerate(model.model.layers):
        if idx > 0:
            selected_experts[idx - 1] = layer.mlp.gate.expert_count

    return selected_experts.cpu().numpy().astype(int)


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
        "--heatmap",
        action="store_true",
        help="Whether to generate a heatmap of the experts"
    )

    parser.add_argument(
        "--layer",
        type=int,
        required=False,
        help="The layer to generate the plot for."
    )

    parser.add_argument(
        "--test",
        action="store_true",
        help="Whether to also perform a chi-squared test"
    )

    args = parser.parse_args()

    if not os.path.isdir(args.model_path):
        print(f"Error: The provided path '{args.model_path}' is not a valid directory.")
        return

    if args.prompt_id < 0:
        print(f"Error: The provided prompt ID '{args.prompt_id}' must be between 0 and {len(PROMPTS)} (exclusively).")
        return

    if args.layer is not None and (args.layer < 0 or args.layer >= NUM_LAYERS):
        print(f"Error: The provided layer idx '{args.layer}' must be between 0 and {NUM_LAYERS} (exclusively).")
        return

    data = profile_execution(args.model_path, PROMPTS[args.prompt_id])

    if args.heatmap:
        generate_heatmap(data)

    generate_layer_distribution(data, layer_id=args.layer, perform_test=args.test)
