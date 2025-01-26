import math

import numpy as np
import torch
from cycler import cycler
from matplotlib import pyplot as plt

from coding_assistant.benchmark.utils import DEVICE, time_function, efficiency

from sglang.srt.layers.attention.triton_ops.decode_attention import decode_attention_fwd
from sglang.srt.layers.attention.triton_ops.extend_attention import extend_attention_fwd
from triton.ops.flash_attention import attention as attention_triton
from flash_attn import flash_attn_func


def flops(bsz: int, seqlen: int, nheads: int, headdim: int, decoding: bool) -> int:
    """
    Computes the number of FLOPS for the attention mechanism used during decoding.
    DeepSeek v2 Lite-Instruct uses different dimensions for keys and values
    (headdim for keys is 192, for values it is 128).
    Here, we compute the count for the Flash Attention mechanism where the dimensions match.

    K, V: bsz x seqlen x nheads x headdim
    Q: bsz x q_len x nheads x headdim where q_len = seqlen for prefilling or q_len = 1 for decoding

    Attention mechanism:
    - softmax(KQ^T / sqrt(headdim)) * V

    :param bsz: the batch size.
    :param seqlen: the sequence length.
    :param nheads: the number of heads.
    :param headdim: the dimension of each head.
    :param decoding: whether the model is in prefilling or decoding mode.
    :return: the total number of FLOPS (additions + multiplications)
    """

    q_len = 1 if decoding else seqlen

    f = bsz * nheads * q_len * seqlen * (4 * headdim + 4)
    return f


def attention_pytorch(q, k, v, softmax_scale):
    """
    Pytorch implementation of attention.

    K, V: bsz x nheads x seqlen x headdim
    Q: bsz x nheads x q_len x headdim where q_len = seqlen for prefilling or q_len = 1 for decoding

    :param q: the queries.
    :param k: the keys.
    :param v: the values.
    :return:
    """

    attn_weights = torch.matmul(q, k.transpose(2, 3)) * softmax_scale

    attn_weights = torch.nn.functional.softmax(
        attn_weights, dim=-1, dtype=torch.float32
    ).to(q.dtype)

    attn_output = torch.matmul(attn_weights, v)

    return attn_output


def generate_plot(methods: list[str], measurements: np.ndarray, decoding: bool, nheads: int, headdim: int, seqlens: list[int]) -> None:
    """
    Generates the performance plots given by the measurements

    :param measurements: m x n array consisting of m methods and n configurations.
    :return:
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    custom_colors = ['#1f77b4', '#ff7f0e', '#d62728']

    ax.set_prop_cycle(cycler(color=custom_colors))
    for i, method in enumerate(methods):
        ax.plot(seqlens, measurements[i], label=method, marker='o', linewidth=2)

    ax.set_title(f"Decoding" if decoding else "Prefill", fontsize=16, weight="bold")
    ax.set_xlabel("Sequence Length", fontsize=12)
    ax.set_ylabel("Performance (TFLOPS/s)", fontsize=12)
    ax.legend(title="Methods", fontsize=10)
    ax.grid(visible=True, linestyle='--', alpha=0.7)
    ax.set_ylim(bottom=0)

    ax.set_xscale('log')
    ax.set_xticks(seqlens)
    ax.get_xaxis().set_major_formatter(plt.FuncFormatter(lambda x, _: f'{int(x):,}'))

    ax.tick_params(axis='both', which='major', labelsize=10, width=1.5, length=6)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(1.5)
    ax.spines['bottom'].set_linewidth(1.5)

    ax.yaxis.grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout()

    file_name = f"{'Decoding' if decoding else 'Prefill'},{nheads}x{headdim}.png"
    plt.savefig(file_name, dpi=300, bbox_inches='tight')


def main():
    print("Running Attention Benchmark")

    # methods = ["Pytorch", "Flash2", "Triton", "SGLang"]
    methods = ["Pytorch", "Flash2", "SGLang"]

    # DeepSeek v2 Lite-Instruct performs q = 192 x v = 128 attention mechanism on 16 attention heads
    # DeepSeek v2 Instruct performs q = 192 x v = 128 attention mechanism on 128 attention heads
    # LLaMA-13B performs q = 128 x v = 128 attention mechanism on 40 attention heads
    # LLaMA-65B performs q = 128 x v = 128 attention mechanism on 64 attention heads

    decoding_options = [False, True]
    bsz_seqlens_options = [(1, 128), (1, 256), (1, 512), (1, 1024), (1, 2048), (1, 4096), (1, 8192), (1, 16384)]
    nheads_options = [16]
    headdim_options = [192]

    dtype = torch.float16

    for decoding in decoding_options:
        for nheads in nheads_options:
            for headdim in headdim_options:
                time_f = {}
                measurements = np.zeros((len(methods), len(bsz_seqlens_options)))
                for config_idx, (bsz, seqlen) in enumerate(bsz_seqlens_options):
                    # format: bs x seqlen x nheads x headdim
                    q = torch.randn(bsz, 1 if decoding else seqlen, nheads, headdim, device=DEVICE, dtype=dtype, requires_grad=False)
                    k = torch.randn(bsz, seqlen, nheads, headdim, device=DEVICE, dtype=dtype, requires_grad=False)
                    v = torch.randn(bsz, seqlen, nheads, headdim, device=DEVICE, dtype=dtype, requires_grad=False)

                    softmax_scale = 1. / math.sqrt(headdim)

                    if "Flash2" in methods:
                        avg_exec_time = time_function(flash_attn_func, q, k, v, 0.0, softmax_scale, False)[0]
                        time_f["Flash2"] = avg_exec_time

                    v = torch.randn(bsz, seqlen, nheads, 128, device=DEVICE, dtype=dtype, requires_grad=False)

                    if "SGLang" in methods:
                        if decoding:
                            q_buffer = q.view(-1, nheads, headdim)
                            k_buffer = k.view(-1, nheads, headdim)
                            v_buffer = v.view(-1, nheads, headdim)

                            o = torch.empty_like(q_buffer)
                            total_tokens = bsz * seqlen
                            req_to_token = torch.arange(0, total_tokens).to(0).int().view(bsz, seqlen)
                            b_req_idx = torch.arange(0, bsz).to(0).int()
                            b_seq_len = torch.full((bsz,), seqlen, dtype=torch.int32, device=DEVICE)
                            num_kv_splits = nheads

                            attn_logits = torch.empty(
                                (bsz, nheads, num_kv_splits, headdim + 1),
                                dtype=torch.float32,
                                device="cuda",
                            )

                            avg_exec_time = time_function(decode_attention_fwd,
                                                          q_buffer,
                                                          k_buffer,
                                                          v_buffer,
                                                          o,
                                                          req_to_token,
                                                          b_req_idx,
                                                          b_seq_len,
                                                          attn_logits,
                                                          num_kv_splits,
                                                          softmax_scale,
                                                          )[0]
                        else:
                            q_buffer = q.view(-1, nheads, headdim)
                            k_buffer = k.view(-1, nheads, headdim)
                            v_buffer = v.view(-1, nheads, headdim)

                            o = torch.empty_like(q_buffer)
                            total_tokens = bsz * seqlen
                            req_to_token = torch.arange(0, total_tokens).to(0).int().view(bsz, seqlen)
                            b_req_idx = torch.arange(0, bsz).to(0).int()
                            b_seq_len = torch.full((bsz,), seqlen, dtype=torch.int32, device=DEVICE)
                            b_seq_len_extend = torch.full((bsz,), seqlen, dtype=torch.int32, device=DEVICE)
                            b_start_loc_extend = torch.zeros(bsz).to(0).int()

                            avg_exec_time = time_function(extend_attention_fwd,
                                                          q_buffer,
                                                          k_buffer.contiguous(),
                                                          v_buffer.contiguous(),
                                                          o,
                                                          k_buffer,
                                                          v_buffer,
                                                          req_to_token,
                                                          b_req_idx,
                                                          b_seq_len,
                                                          b_seq_len_extend,
                                                          b_start_loc_extend,
                                                          seqlen,
                                                          softmax_scale,
                                                          )[0]

                        time_f["SGLang"] = avg_exec_time

                    # format: bsz x nheads x seqlen x headdim
                    q = q.transpose(1, 2)
                    k = k.transpose(1, 2)
                    v = v.transpose(1, 2)

                    if "Pytorch" in methods:
                        try:
                            avg_exec_time = time_function(attention_pytorch, q, k, v, softmax_scale)[0]
                        except:  # Skip if OOM
                            avg_exec_time = float('nan')
                        time_f["Pytorch"] = avg_exec_time

                    if "Triton" in methods:
                        if headdim in {16, 32, 64, 128}:
                            avg_exec_time = time_function(
                                attention_triton, q, k, v, False, softmax_scale,
                                False
                            )[0]
                        else:
                            avg_exec_time = float('nan')
                        time_f["Triton"] = avg_exec_time

                    print(f"### decoding={decoding}, batch_size={bsz}, seqlen={seqlen}, nheads={nheads}, headdim={headdim} ###")
                    num_flops = flops(bsz, seqlen, nheads, headdim, decoding)
                    for idx, method in enumerate(methods):
                        performance = efficiency(num_flops, time_f[method])
                        print(f"{method}: {performance:.2f} TFLOPs/s")
                        measurements[idx, config_idx] = performance

                # plot
                generate_plot(methods, measurements, decoding, nheads, headdim, [tup[1] for tup in bsz_seqlens_options])
