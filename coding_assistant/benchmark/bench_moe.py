import numpy as np
import torch
from matplotlib import pyplot as plt
from torch.nn import functional as F

from sglang.srt.layers.moe.fused_moe_triton.fused_moe import fused_experts as moe_sglang
from vllm.model_executor.layers.fused_moe.fused_moe import fused_experts as moe_vllm

from coding_assistant.benchmark.utils import DEVICE, time_function, efficiency


MODELS = ["DeepSeek Coder V2 Lite-Instruct", "DeepSeek Coder V2", "Mixtral"]
CONFIGS = [
    {
        "hidden_size": 2048,
        "moe_intermediate_size": 1408,
        "n_routed_experts": 64,
        "top_k": 6
    },
    {
        "hidden_size": 2048,
        "moe_intermediate_size": 1408,
        "n_routed_experts": 160,
        "top_k": 6
    },
    {
        "hidden_size": 4096,
        "moe_intermediate_size": 14336,
        "n_routed_experts": 8,
        "top_k": 2
    }
]


def fused_moe_vllm(hidden_state, topk_weight, topk_idx, w13_weight, w2_weight) -> torch.Tensor:
    return moe_vllm(
        hidden_state,
        w13_weight,
        w2_weight,
        topk_weight,
        topk_idx,
        inplace=False,
    )


def fused_moe_sglang(hidden_state, topk_weight, topk_idx, w13_weight, w2_weight) -> torch.Tensor:
    return moe_sglang(
        hidden_states=hidden_state,
        w1=w13_weight,
        w2=w2_weight,
        topk_weights=topk_weight,
        topk_ids=topk_idx,
        inplace=False,
    )


@torch.compile(dynamic=False)
def fused_moe_torch(hidden_state, topk_weight, topk_idx, w13_weight, w2_weight) -> torch.Tensor:
    w13_weights = w13_weight[topk_idx]
    w1_weights, w3_weights = torch.chunk(w13_weights, 2, dim=2)
    w2_weights = w2_weight[topk_idx]
    x1 = torch.einsum("ti,taoi -> tao", hidden_state, w1_weights)
    x1 = F.silu(x1)
    x3 = torch.einsum("ti, taoi -> tao", hidden_state, w3_weights)
    expert_outs = torch.einsum("tao, taio -> tai", (x1 * x3), w2_weights)
    return torch.einsum("tai,ta -> ti", expert_outs, topk_weight.to(expert_outs.dtype))


def fused_moe_naive(hidden_state, topk_weight, topk_idx, weights, config):
    cnts = topk_idx.new_zeros((topk_idx.shape[0], config["n_routed_experts"]))
    cnts.scatter_(1, topk_idx, 1)
    tokens_per_expert = cnts.sum(dim=0)
    idxs = topk_idx.view(-1).argsort()
    sorted_tokens = hidden_state[idxs // topk_idx.shape[1]]

    tokens_per_expert = tokens_per_expert.cpu().numpy()

    outputs = []
    start_idx = 0
    for i, num_tokens in enumerate(tokens_per_expert):
        end_idx = start_idx + num_tokens
        if num_tokens == 0:
            continue
        tokens_for_this_expert = sorted_tokens[start_idx:end_idx]

        gate_proj_weight, up_proj_weight, down_proj_weight = weights[i]

        gate_proj = F.linear(tokens_for_this_expert, gate_proj_weight)
        up_proj = F.linear(tokens_for_this_expert, up_proj_weight)
        activated = torch.nn.functional.silu(gate_proj) * up_proj
        expert_out = F.linear(activated, down_proj_weight)

        outputs.append(expert_out)
        start_idx = end_idx

    outs = torch.cat(outputs, dim=0) if len(outputs) else sorted_tokens.new_empty(0)

    new_x = torch.empty_like(outs)
    new_x[idxs] = outs
    return (
        new_x.view(*topk_idx.shape, -1)
        .type(topk_weight.dtype)
        .mul_(topk_weight.unsqueeze(dim=-1))
        .sum(dim=1)
        .type(new_x.dtype)
    )


def flops(num_tokens, hidden_size, moe_intermediate_size, top_k):
    gate_proj = top_k * num_tokens * hidden_size * moe_intermediate_size * 2
    up_proj = top_k * num_tokens * hidden_size * moe_intermediate_size * 2
    gate_up = top_k * num_tokens * moe_intermediate_size
    down_proj = top_k * num_tokens * hidden_size * moe_intermediate_size * 2
    weighted_sum = top_k * num_tokens * hidden_size

    return gate_proj + up_proj + gate_up + down_proj + weighted_sum


def generate_plot(method_names: list[str], measurements: dict[str, np.ndarray]):
    means = np.array([measurements[model] for model in MODELS])

    # Plot settings
    colors = plt.cm.tab10.colors[:len(method_names)]
    x = np.arange(len(MODELS))
    bar_width = 0.2
    fig, ax = plt.subplots(figsize=(10, 6))

    # Create grouped bars
    for idx, method in enumerate(method_names):
        ax.bar(
            x + idx * bar_width,  # Bar positions within groups
            means[:, idx],  # Heights (mean values)
            bar_width,  # Width of bars
            label=method,  # Legend label
            color=colors[idx],  # Bar color
            capsize=5,  # Error bar cap size
            edgecolor="black",  # Add bar border for clarity
            linewidth=1.5
        )

    # Customizations
    ax.set_ylabel("Performance (TFLOPS/s)", fontsize=12)
    ax.set_xticks(x + bar_width * (len(method_names) - 1) / 2)
    ax.set_xticklabels(MODELS, fontsize=10)
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
    plt.savefig("bench_moe.png", dpi=300, bbox_inches='tight')


def main():
    print("Running Mixture of Experts Benchmark")

    # decoding stage
    bsz, seq_len = 1, 1

    measurements = {model: np.zeros(4) for model in MODELS}
    for model_name, model_config in zip(MODELS, CONFIGS):
        hidden_size = model_config["hidden_size"]
        moe_intermediate_size = model_config["moe_intermediate_size"]
        n_routed_experts = model_config["n_routed_experts"]
        top_k = model_config["top_k"]

        hidden_state = torch.randn(bsz * seq_len, hidden_size, device=DEVICE, dtype=torch.bfloat16, requires_grad=False)
        weight = torch.randn(n_routed_experts, hidden_size, device=DEVICE, dtype=torch.bfloat16, requires_grad=False)

        logits = F.linear(
            hidden_state.type(torch.float32), weight.type(torch.float32), None
        )

        scores = logits.softmax(dim=-1)
        topk_weight, topk_idx = torch.topk(
            scores, k=top_k, dim=-1, sorted=False
        )

        gate_proj_weights = [torch.randn(moe_intermediate_size, hidden_size, device=DEVICE, dtype=torch.bfloat16, requires_grad=False) for _ in range(n_routed_experts)]
        up_proj_weights = [torch.randn(moe_intermediate_size, hidden_size, device=DEVICE, dtype=torch.bfloat16, requires_grad=False) for _ in range(n_routed_experts)]
        down_proj_weights = [torch.randn(hidden_size, moe_intermediate_size, device=DEVICE, dtype=torch.bfloat16, requires_grad=False) for _ in range(n_routed_experts)]

        weights = list(zip(gate_proj_weights, up_proj_weights, down_proj_weights))

        # prepare torch compile, sglang and vllm weights
        w13_weight = torch.stack([torch.cat([t1, t2], dim=0) for t1, t2 in zip(gate_proj_weights, up_proj_weights)])
        w2_weight = torch.stack(down_proj_weights)

        config = {
            "bsz": bsz,
            "seq_len": seq_len,
            "hidden_size": hidden_size,
            "n_routed_experts": n_routed_experts,
            "top_k": top_k,
        }

        exec_time_naive = time_function(fused_moe_naive, hidden_state, topk_weight, topk_idx, weights, config)[0]
        exec_time_torch = time_function(fused_moe_torch, hidden_state, topk_weight, topk_idx, w13_weight, w2_weight)[0]
        exec_time_sglang = time_function(fused_moe_sglang, hidden_state, topk_weight, topk_idx, w13_weight, w2_weight)[0]
        exec_time_vllm = time_function(fused_moe_vllm, hidden_state, topk_weight, topk_idx, w13_weight, w2_weight)[0]

        num_flops = flops(bsz * seq_len, hidden_size, moe_intermediate_size, top_k)

        print(f"### {model_name} - hidden_size={hidden_size}, moe_intermediate_size={moe_intermediate_size}, n_routed_experts={n_routed_experts}, top_k={top_k} ###")
        print(f"Pytorch: {efficiency(num_flops, exec_time_naive):.2f} TFLOPs/s")
        print(f"Torch compile: {efficiency(num_flops, exec_time_torch):.2f} TFLOPs/s")
        print(f"SGLang: {efficiency(num_flops, exec_time_sglang):.2f} TFLOPs/s")
        print(f"VLLM: {efficiency(num_flops, exec_time_vllm):.2f} TFLOPs/s")

        measurements[model_name] = np.array([
            efficiency(num_flops, exec_time_naive),
            efficiency(num_flops, exec_time_torch),
            efficiency(num_flops, exec_time_sglang),
            efficiency(num_flops, exec_time_vllm)
        ])

    generate_plot(["Pytorch", "Torch Compile", "SGLang", "VLLM"], measurements)
