# ICRL: In-Context Reinforcement Learning for Tool Use in Large Language Models

A framework for training Large Language Models to use external tools (e.g., search engines) through in-context learning combined with reinforcement learning.

## Installation

### Install Environment

```bash
git clone https://github.com/applese233/ICRL.git
cd ICRL

# Environment with Python 3.9
conda env create -f environment.yml
conda activate icrl
pip install -e .
pip install flash-attn --no-build-isolation
```

## Quick Start

### 1. Prepare Data

```bash
# 3-shot few-shot data used by train_grpo_fewshot.sh
python scripts/data_process/nq_search_fewshot.py \
    --local_dir data/nq_search_fewshot \
    --num_examples 3
```

Few-shot data preparation supports choosing which example set to use from the `example/` directory. The default is `fewshot_examples.txt`.

```bash
# Use a built-in example set under example/
python scripts/data_process/nq_search_fewshot.py \
    --template_type fewshot \
    --examples_name 7B_examples.txt \
    --num_examples 3

# Or point to an explicit file path
python scripts/data_process/nq_search_fewshot.py \
    --template_type fewshot \
    --examples_file example/wrong_examples.txt \
    --num_examples 2
```

Available example files currently include `fewshot_examples.txt`, `7B_examples.txt`, and `wrong_examples.txt`.

If you want to run the curriculum script, prepare all three stages first:

```bash
# Stage 1: 3-shot
python scripts/data_process/nq_search_fewshot.py \
    --template_type fewshot \
    --local_dir data/nq_search_fewshot \
    --num_examples 3

# Stage 2: 2-shot
python scripts/data_process/nq_search_fewshot.py \
    --template_type fewshot \
    --local_dir data/nq_search_2shot \
    --num_examples 2

# Stage 3: 0-shot
python scripts/data_process/nq_search_fewshot.py \
    --template_type zeroshot \
    --local_dir data/nq_search_0shot
```

### 2. Start Search Server

The training requires a web search server. We use SerpAPI or Serper.dev as the search backend.

First, get your API key from:
- [SerpAPI](https://serpapi.com) (default)
- [Serper.dev](https://serper.dev)

Then start the server:

```bash
# Using SerpAPI (default)
SERPAPI_KEY=your_api_key bash scripts/search/run_search_server.sh

# Or using Serper.dev
PROVIDER=serper SERPAPI_KEY=your_api_key bash scripts/search/run_search_server.sh
```

The server will listen on `http://127.0.0.1:8000/retrieve`.

### 3. Train with GRPO

You can train the model with a single stage:

```bash
bash train_grpo_fewshot.sh
```

Or train the whole method:

```bash
bash train_curriculum.sh
```

The training scripts default to 4 GPUs and the dataset paths above, but you can override them with environment variables:

```bash
CUDA_VISIBLE_DEVICES=0,1 N_GPUS=2 bash train_grpo_fewshot.sh

CUDA_VISIBLE_DEVICES=0,1 N_GPUS=2 \
STAGE1_DATA_DIR=data/nq_search_fewshot \
STAGE2_DATA_DIR=data/nq_search_2shot \
STAGE3_DATA_DIR=data/nq_search_0shot \
bash train_curriculum.sh
```

### 4. Inference

Test the model with a single question:

```bash
# Ask a question directly
python infer.py \
    --model_path Qwen/Qwen2.5-7B-Instruct \
    --question "Who won the 2022 FIFA World Cup?"

# Interactive mode (enter questions one by one)
python infer.py --model_path your_trained_model

# Use few-shot prompts
python infer.py \
    --model_path your_model \
    --use_fewshot \
    --fewshot_path example/fewshot_examples.txt \
    --question "Your question here"
```

### 5. Evaluate

Evaluate on benchmark datasets:

```bash
# Convert evaluation datasets only
bash eval_batch_vllm.sh --datasets "triviaqa hotpotqa musique 2wikimultihopqa bamboogle" --convert_only

# Run evaluation with a checkpoint
bash eval_batch_vllm.sh \
    --checkpoint_ref icrl-grpo-qwen2.5-7B/global_step_50 \
    --datasets "triviaqa hotpotqa musique 2wikimultihopqa bamboogle" \
    --use_vllm

# Equivalent form: model name + global step
bash eval_batch_vllm.sh \
    --model_name icrl-grpo-qwen2.5-7B \
    --global_step global_step_50 \
    --datasets "triviaqa hotpotqa musique 2wikimultihopqa bamboogle" \
    --use_vllm
```

For named checkpoints, the default lookup path is `./checkpoints/<model_name>/actor/<global_step>`. You can override the root with `--checkpoint_dir`.

## Evaluation Datasets

- TriviaQA
- HotpotQA
- 2WikiMultihopQA
- MuSiQue
- Bamboogle

## Acknowledgements

This project builds upon:
- [veRL](https://github.com/volcengine/verl) - Volcano Engine Reinforcement Learning for LLMs
- [Search-R1](https://github.com/PeterGriffinJin/Search-R1) - Search-augmented LLM training

## License

Apache License 2.0
