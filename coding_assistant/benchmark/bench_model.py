import argparse
import time
from typing import Optional

import numpy as np
import torch
from matplotlib import pyplot as plt
from transformers import AutoTokenizer, PreTrainedTokenizerFast

from deepspeed.profiling.flops_profiler import (FlopsProfiler, get_model_profile, get_module_flops,
                                                get_module_macs, get_module_duration, params_to_string,
                                                macs_to_string, duration_to_string, flops_to_string)

from coding_assistant.benchmark.utils import DEVICE, LayerStatistics
from coding_assistant.models import ModelFactory, AbstractModel
from coding_assistant.models.model_factory import AVAILABLE_MODELS

BASE_PROMPT = "Please give me the code of the BFS algorithm in Python"
FILL_ID = 13  # '.' symbol
NUM_EXPERIMENTS = 3

SCENARIOS = {
    "RAG Lookup": (2048, 128),
    "Code Generation": (128, 512),
    "Code Completion": (512, 512),
}


def prepare_model_prompt(tokenizer: PreTrainedTokenizerFast, input_size: int) -> torch.Tensor:
    messages = [
        {'role': 'user', 'content': BASE_PROMPT}
    ]
    inputs = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt").to(DEVICE)

    num_fill = input_size - inputs.size(1)
    if num_fill > 0:
        fill_tensor = torch.full((1, num_fill), FILL_ID).to(DEVICE)

        inputs = torch.cat((inputs[:, :-4], fill_tensor, inputs[:, -4:]), dim=-1)
    else:
        inputs = inputs[:, :input_size]

    # warmup
    return inputs


def benchmark_layers(model_type: type[AbstractModel], input_size: int, output_size: int,
                     tokenizer: PreTrainedTokenizerFast):
    print(f"Perform layer-wise benchmarking of model {model_type.get_name()}")

    model_wrapper = model_type()

    if model_wrapper.base_model is None:
        print("Cannot profile the layers of this model")
        return

    inputs = prepare_model_prompt(tokenizer, input_size)

    # warmup
    prompt_token_ids = model_wrapper.prepare_prompt_tokens(inputs)
    _ = model_wrapper.generate(prompt_token_ids, max_new_tokens=output_size)

    profiler = FlopsProfiler(model_wrapper.base_model)

    profiler.start_profile()

    _ = model_wrapper.generate(prompt_token_ids, max_new_tokens=output_size)

    statistics = LayerStatistics()

    statistics.register_layer("DeepseekV2ForCausalLM")
    statistics.register_layer("DeepseekV2Model")
    statistics.register_layer("Embedding")
    statistics.register_layer("DeepseekV2DecoderLayer")
    statistics.register_layer("DeepseekV2Attention")
    statistics.register_layer("DeepseekV2MoE")

    for idx, layer in enumerate(model_wrapper.base_model.model.layers):
        statistics.update_stats("DeepseekV2DecoderLayer", layer)
        statistics.update_stats("DeepseekV2Attention", layer.self_attn)

        if idx > 0:
            statistics.update_stats("DeepseekV2MoE", layer.mlp)

    statistics.update_stats("DeepseekV2ForCausalLM", model_wrapper.base_model)
    statistics.update_stats("DeepseekV2Model", model_wrapper.base_model.model)
    statistics.update_stats("Embedding", model_wrapper.base_model.model.embed_tokens)

    total_macs = profiler.get_total_macs()
    total_duration = profiler.get_total_duration()
    total_params = profiler.get_total_params()
    statistics.set_total_statistics(total_macs, total_duration, total_params)

    for layer_name in statistics.get_stats():
        print(f"{layer_name}: {statistics.flops_repr(layer_name)}")

    profiler.end_profile()


def execute_model(model: AbstractModel, input_size: int, output_size: int, tokenizer: PreTrainedTokenizerFast):
    inputs = prepare_model_prompt(tokenizer, input_size)

    # warmup
    prompt_token_ids = model.prepare_prompt_tokens(inputs)
    _ = model.generate(prompt_token_ids, max_new_tokens=output_size)

    performances = []
    for _ in range(NUM_EXPERIMENTS):
        start_time = time.time()
        output = model.generate(prompt_token_ids=prompt_token_ids, max_new_tokens=output_size)
        end_time = time.time()

        assert len(output) == output_size, \
            f"Generated output length ({len(output)}) does not match expected length ({output_size})"

        exec_time = end_time - start_time

        performances.append(output_size / exec_time)

    np_arr = np.array(performances)
    mean, std = np_arr.mean(), np_arr.std()

    print(f"Model {model.get_name()} - input size: {input_size} - output size: {output_size}")
    print(f"Avg generated tokens per second: {mean:.2f}")
    print(f"Standard deviation: {std:.2f}")

    return mean, std


def generate_plot(model_names: list[str], measurements: dict[str, np.ndarray]):
    means = np.array([measurements[scenario][:, 0] for scenario in SCENARIOS])
    stds = np.array([measurements[scenario][:, 1] for scenario in SCENARIOS])

    # Plot settings
    colors = plt.cm.tab10.colors[:len(model_names)]
    x = np.arange(len(SCENARIOS))
    bar_width = 0.2
    fig, ax = plt.subplots(figsize=(10, 6))

    # Create grouped bars
    for idx, model_name in enumerate(model_names):
        ax.bar(
            x + idx * bar_width,  # Bar positions within groups
            means[:, idx],  # Heights (mean values)
            bar_width,  # Width of bars
            yerr=stds[:, idx],  # Error bars (std deviation)
            label=model_name,  # Legend label
            color=colors[idx],  # Bar color
            capsize=5,  # Error bar cap size
            edgecolor="black",  # Add bar border for clarity
            linewidth=1.5
        )

    # Customizations
    ax.set_ylabel("Performance (Tokens/s)", fontsize=12)
    ax.set_xticks(x + bar_width * (len(model_names) - 1) / 2)
    ax.set_xticklabels(SCENARIOS, fontsize=10)
    ax.legend(title="Implementations", fontsize=10)
    ax.set_axisbelow(True)
    ax.grid(axis='y', linestyle='--', alpha=0.7)

    # Customize ticks and spines
    ax.tick_params(axis='y', which='major', labelsize=10, width=1.5, length=6)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(1.5)
    ax.spines['bottom'].set_linewidth(1.5)

    # Adjust layout
    plt.tight_layout()

    # Save the plot as a PNG file
    plt.savefig("end_to_end_grouped_bar.png", dpi=300, bbox_inches='tight')


def benchmark_model(model_types: list[type[AbstractModel]], sizes: Optional[tuple[int, int]],
                    tokenizer: PreTrainedTokenizerFast):

    if sizes is not None:
        for model_type in model_types:
            print(f"Perform end-to-end benchmarking of model {model_type.get_name()}")

            execute_model(model_type(), *sizes, tokenizer)
    else:
        print("Perform end-to-end benchmarking on predefined scenarios")
        measurements = {key: np.empty((len(model_types) + 1, 2)) for key in SCENARIOS.keys()}

        for model_idx, model_type in enumerate(model_types):
            model = model_type() if model_type.get_name() != "sglang" else model_type

            for scenario, sizes in SCENARIOS.items():
                mean, std = execute_model(model, *sizes, tokenizer)
                measurements[scenario][model_idx, 0] = mean
                measurements[scenario][model_idx, 1] = std

        generate_plot([model_type.get_name() for model_type in model_types], measurements)


def main():
    print("Executing bench-models CLI")

    parser = argparse.ArgumentParser(
        description="A CLI for benchmarking models in an end-to-end fashion."
    )

    parser.add_argument(
        "--model",
        type=str,
        required=False,
        choices=[model_type.get_name() for model_type in AVAILABLE_MODELS],
        help="Name of the model to benchmark."
    )

    parser.add_argument(
        '--layer',
        action='store_true',
        help='Activate layer functionality'
    )

    parser.add_argument(
        "--input-size",
        type=int,
        required=False,
        help="The input size for benchmarking."
    )

    parser.add_argument(
        "--output-size",
        type=int,
        required=False,
        help="A list of output size for benchmarking."
    )
    args = parser.parse_args()

    model_types: list[type[AbstractModel]] = [ModelFactory.get_model(args.model)] if args.model is not None \
        else [ModelFactory.get_model("huggingface"), ModelFactory.get_model("sglang"), ModelFactory.get_model("vllm"),
              ModelFactory.get_model("vllm (graph)")]

    tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct", trust_remote_code=True)

    if args.input_size is None or args.output_size is None:
        print("No input size or output size specified, use predefined scenarios")

        if args.layer:
            print("Can only perform layer-wise measurements, if an input and output size is specified")
            return
        benchmark_model(model_types, None, tokenizer)
    else:
        if args.layer:
            for model_type in model_types:
                benchmark_layers(model_type, args.input_size, args.output_size, tokenizer)
        else:
            benchmark_model(model_types, (args.input_size, args.output_size), tokenizer)
