# Optimizing LLMs as Local Coding Assistant
<div align="center">

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

</div>


<p align="center">
  <a href="#benchmarking">Benchmarking/Experiments</a> |
  <a href="#installation">Installation Guide</a>
</p>

## Introduction

Large Language Models (LLMs) are increasingly being utilized for software and hardware programming. While cloud-based coding assistants offer excellent performance, local coding assistants are often preferred due to concerns around data privacy, subscription costs, and the ability for personalized fine-tuning.

However, deploying LLMs with advanced programming capabilities on local machines is challenging due to the computational and memory resource limitations of personal computers.

This project focuses on optimizing LLMs for local coding assistants by reducing memory and computational demands, while maintaining their strong software and hardware programming capabilities. The project is conducted as part of a supervised semester project within the [Systems Group](https://systems.ethz.ch/) at ETH Zurich.

We decided on using [DeepSeek Coder V2 Lite-Instruct](https://huggingface.co/deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct) as our base model. This Mixture of Experts (MoE) model provides a good tradeoff between (coding) performance and model size.

## <a id="benchmarking"></a>📊 Benchmarking/Experiments

### Benchmarking

The relevant benchmarking files can be found in `coding_assisstant/benchmark`. Our tools allow users to measure the end-to-end performance of different LLM implementations. Additionally, we can benchmark the attention and MoE mechansim.

#### MoE-Layer
This benchmark evaluates the performance of several MoE-implementation on a set of model configurations.
It can be run via ```bench-moe```.

#### Attention Mechanism
This command benchmarks several well-known attention implementations using preconfigured parameters. For our purpose, we focus on batch size = 1 and distinguish between prefilling and decoding stage. In particular, we used the following configurations:
- `n_heads = 16, headdim = 192`
- `n_heads = 128, headdim = 192`
- `n_heads = 16, headdim = 128`
- `n_heads = 16, headdim = 256`

to reflect different occurrences in state-of-the-art LLMs. The first 2 configurations are used to represent DeepSeek Coder V2 Lite-Instruct and DeepSeek Coder V2 Instruct respectively. The last two are the neighboring powers of 2 using the same number of heads. The values must be set manually and the tool can be executed
via `bench-attention`.

#### End-to-End Measurements
This tool allows users to compare the end-to-end performance of different DeepSeek Coder V2 Lite-Instruct models. Additionally, it can measure the computation time spent per layer to identify bottlenecks in the implementation. We have defined tuples of (input_size, output_size) pairs to represent different scenarios:
- RAG Lookup: `(2048, 128)`
- Code Generation: `(128, 512)`
- Code Completion: `(512, 512)`

The base command to run the benchmarking is:
```
bench-model (--model <model>) (--layer) (--input-size) (--output-size)
```
- `model` name of the model to evaluate (all available if not specified)
- `layer` whether to aggregate layer-wise statistics
- `input-size` the amount of input tokens
- `output-size` the amount of output tokens

If either no input or no output size is provided, the model is evaluated on the predefined scenarios.


### Experiments
For the experiments, we need a local installation of the [huggingface repository](https://huggingface.co/deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct/tree/main) by cloning it in a folder on the server.
In order to reproduce the results for the experiments, navigate to `coding_assistant/experiments/<experiment>`. Then, replace the file `modeling_deepseek.py` in the repository with the version used for the experiment.
You need to do this as the modified version made some changes to the model architecture.

#### Dataformat Change
This command is used to assess the speedup of changing the dataformat in the flash attention layer of the LLM.

The base command is:
```
flash-attn-optimizer <model_path>
```
- `model_path` path to the stored model
- `prompt_id` selects a particular prompt (either 0 or 1)

#### Expert Counting

This command is used to plot the expert distribution that is generated through processing the prompt by the LLM.

The base command is:
```
expert-counter <model_path> <prompt_id> (--heatmap) (--layer (<layer_idx>)) (--test)
```
- `model_path` path to the stored model
- `prompt_id` selects a particular prompt (either 0 or 1)
- `--heatmap` generates the heatmap visualizing the amount of selection per expert over all layers
- `--layer` if set, selects a particular layer and visualizes the amount each expert was selected
- `--test` if set, performs the chi-squared test against the uniform distribution

#### Expert Prefetching

This command is used to assess the predictive performance of the pre-attention activations on the expert selection.

The base command is:
```
expert-prefetcher <model_path> <prompt_id> --show
```
- `model_path` path to the stored model
- `prompt_id` selects a particular prompt (either 0 or 1)
- `--show` prints the generated output

If one wants to see the generated output of the model when only a random subset of the top-n performing experts is selected (called expert replacement experiment), uncomment lines 441-449 and set the variables according to your needs.

## <a id="installation"></a>📥 Installation

All experiments are performed on a AMD Instinct™ MI210 GPU in [ETH's HACC cluster](https://systems.ethz.ch/research/data-processing-on-modern-hardware/hacc.html). In order to reproduce the results, we propose the following installation guide.

### 1. Create Environment

Connect to the remote machine (e.g. via SSH) and create a virtual environment in the working folder.
Then, install the PyTorch version that matches the platform (ROCm 6.2 in this case). After installing PyTorch, you might need to log out of the system.

```
# create local environment
python3 -m venv .local
source .local/bin/activate
# install numpy and networkx
pip install numpy networkx
# check RoCm version
cat /opt/rocm/.info/version
# install corresponding PyTorch from
# https://pytorch.org/get-started/locally/
# Selection: Stable (2.5.1) - Linux - Pip - RoCm
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm6.2
deactivate
exit
```

After login, PyTorch should be able to find the GPUs.

### 2. Install other dependencies

Don't forget to reactivate the virtual environment to install dependencies.

```
source .local/bin/activate
pip install transformers deepspeed
```

### 3. Install Flash Attention

We compare ROCm Composable Kernel (CK) Flash Attention 2 with other implementations. To install Flash Attention, follow the official guide from the [ROCm documentation](https://rocm.docs.amd.com/en/latest/how-to/llm-fine-tuning-optimization/model-acceleration-libraries.html#flash-attention-2). In particular, notice the ```GPU_ARCHS=gfx90a``` environment variable.

```
pip install packaging ninja wheel
git clone --recursive https://github.com/ROCm/flash-attention.git
cd flash-attention
GPU_ARCHS="gfx90a" pip install -v .
```

### 4. Install vLLM

To install vLLM, follow the instructions depicted in the [vLLM installation guide](https://docs.vllm.ai/en/stable/getting_started/amd-installation.html).

AMD SMI is required to detect AMD ROCm GPUs. If it is not possible to install SMI, certain workaround have to be performed to manually hint the availability of the GPUs to the vLLM library.

```
pip install --upgrade pip

# Build & install AMD SMI
pip install /opt/rocm/share/amd_smi

# Install dependencies
pip install --upgrade numba scipy huggingface-hub[cli]
pip install "numpy<2"
pip install -r requirements-rocm.txt

# Build vLLM for MI210
export PYTORCH_ROCM_ARCH="gfx90a"
python3 setup.py develop
```

To use the CK version of Flash Attention during vLLM inference, we need to set the following environment flag:

```
export VLLM_USE_TRITON_FLASH_ATTN=0
```

### 5. Install sglang

Install sglang according to the [official installation guide](https://sgl-project.github.io/start/install.html).
However, you need to make sure that the installations from PyTorch and vLLM are not overwritten by the dependencies of sglang.

```
# Use the last release branch
git clone -b v0.4.0.post1 https://github.com/sgl-project/sglang.git
cd sglang
```

The workaround is to only install the runtime-common dependencies.

```
pip install --upgrade pip
pip install -e "python[runtime_common]"
```

### 6. Install this project

```
git clone https://github.com/robin-oester/coding-assistant.git
cd coding-assistant
pip install -e .
```
