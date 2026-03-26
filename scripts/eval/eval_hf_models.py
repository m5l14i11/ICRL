#!/usr/bin/env python3
"""
Evaluation script for HuggingFace models on ICRL-style tasks.

This script is designed to evaluate various web-search-capable models from HuggingFace,
including:
- Alibaba-NLP/ZeroSearch_google_V2_Qwen2.5_3B_Instruct
- Alibaba-NLP/ZeroSearch_google_V2_Qwen2.5_7B_Instruct  
- Jianbiao/O2-Searcher-Qwen2.5-3B-GRPO
- openai/gpt-oss-20b (uses harmony format - different prompt structure)
- And other search-augmented LLMs

These models use the same prompt format as ICRL:
- <think>...</think> for reasoning
- <search>...</search> for search queries
- <information>...</information> for search results
- <answer>...</answer> for final answers

Note: gpt-oss models use OpenAI's harmony format which is different. They will be
evaluated without web search capability (direct generation only).

Usage:
    python eval_hf_models.py \
        --model_id "Alibaba-NLP/ZeroSearch_google_V2_Qwen2.5_3B_Instruct" \
        --data_file ./data/eval/triviaqa_eval.parquet \
        --output_dir ./eval_results/zerosearch_3b
"""

import os
import sys
import json
import argparse
import time
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from tqdm import tqdm
import numpy as np

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import pandas as pd
import requests
import re
import string


# Predefined HuggingFace models for comparison
HF_MODELS = {
    "zerosearch-3b": "Alibaba-NLP/ZeroSearch_google_V2_Qwen2.5_3B_Instruct",
    "zerosearch-7b": "Alibaba-NLP/ZeroSearch_google_V2_Qwen2.5_7B_Instruct",
    "o2-searcher-3b": "Jianbiao/O2-Searcher-Qwen2.5-3B-GRPO",
    # "gpt-oss-20b": "openai/gpt-oss-20b",  # Requires: pip install -U transformers kernels
    "stepsearch-3b": "Zill1/StepSearch-3B-Instruct",
    "stepsearch-7b": "Zill1/StepSearch-7B-Instruct",
}

# Models that were trained with topk=5 (need to use --topk 5 for fair comparison)
# ZeroSearch paper uses topk=5 for training
TOPK5_MODELS = {
    "Alibaba-NLP/ZeroSearch_google_V2_Qwen2.5_3B_Instruct",
    "Alibaba-NLP/ZeroSearch_google_V2_Qwen2.5_7B_Instruct",
    "Alibaba-NLP/ZeroSearch_google_V2_Qwen2.5_3B",
    "Alibaba-NLP/ZeroSearch_google_V2_Qwen2.5_7B",
    "Alibaba-NLP/ZeroSearch_wiki_V2_Qwen2.5_3B_Instruct",
    "Alibaba-NLP/ZeroSearch_wiki_V2_Qwen2.5_7B_Instruct",
}

# Models that use OpenAI harmony format (different from ICRL <search> tags)
# These models will be evaluated without web search (direct generation)
HARMONY_FORMAT_MODELS = {
    "openai/gpt-oss-20b",
    "openai/gpt-oss-120b",
}


@dataclass
class EvalConfig:
    """Configuration for evaluation."""
    model_id: str  # HuggingFace model ID or local path
    data_file: Optional[str] = None
    data_dir: Optional[str] = None
    output_dir: str = "./eval_results"
    search_url: str = "http://127.0.0.1:8000/retrieve"
    topk: int = 3
    max_turns: int = 6
    max_new_tokens: int = 512
    temperature: float = 0.1
    do_sample: bool = True
    no_search: bool = False
    device: str = "auto"
    verbose: bool = False
    num_samples: Optional[int] = None
    cache_dir: Optional[str] = None


def normalize_answer(s: str) -> str:
    """Normalize answer for comparison."""
    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def exact_match_score(prediction: str, ground_truths: List[str]) -> bool:
    """Check if prediction exactly matches any ground truth."""
    normalized_prediction = normalize_answer(prediction)
    for gt in ground_truths:
        if normalized_prediction == normalize_answer(gt):
            return True
    return False


def substring_match_score(prediction: str, ground_truths: List[str]) -> bool:
    """Check if any ground truth is a substring of the prediction."""
    normalized_prediction = normalize_answer(prediction)
    for gt in ground_truths:
        if normalize_answer(gt) in normalized_prediction:
            return True
    return False


def f1_score(prediction: str, ground_truths: List[str]) -> float:
    """Compute F1 score between prediction and ground truths."""
    def compute_f1(prediction_tokens, ground_truth_tokens):
        common = set(prediction_tokens) & set(ground_truth_tokens)
        if len(common) == 0:
            return 0.0
        precision = len(common) / len(prediction_tokens)
        recall = len(common) / len(ground_truth_tokens)
        return 2 * precision * recall / (precision + recall)
    
    prediction_tokens = normalize_answer(prediction).split()
    if not prediction_tokens:
        return 0.0
    
    best_f1 = 0.0
    for gt in ground_truths:
        gt_tokens = normalize_answer(gt).split()
        if gt_tokens:
            best_f1 = max(best_f1, compute_f1(prediction_tokens, gt_tokens))
    
    return best_f1


def extract_answer(text: str) -> Optional[str]:
    """Extract answer from <answer>...</answer> tags."""
    matches = list(re.finditer(r'<answer>(.*?)</answer>', text, re.DOTALL | re.IGNORECASE))
    if not matches:
        return None
    return matches[-1].group(1).strip()


def extract_search_query(text: str) -> Optional[str]:
    """Extract search query from <search>...</search> tags."""
    matches = list(re.finditer(r'<search>(.*?)</search>', text, re.DOTALL | re.IGNORECASE))
    if not matches:
        return None
    return matches[-1].group(1).strip()


class HFModelEvaluator:
    """Evaluator for HuggingFace models."""
    
    def __init__(self, config: EvalConfig):
        self.config = config
        self.model = None
        self.tokenizer = None
        self.device = None
        
        self._load_model()
        
    def _load_model(self):
        """Load model and tokenizer from HuggingFace."""
        print(f"Loading model from {self.config.model_id}...")
        
        # Set cache directory
        cache_kwargs = {}
        if self.config.cache_dir:
            cache_kwargs['cache_dir'] = self.config.cache_dir
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_id, 
            trust_remote_code=True,
            **cache_kwargs
        )
        
        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.model_id,
            torch_dtype=torch.bfloat16,
            device_map=self.config.device,
            trust_remote_code=True,
            **cache_kwargs
        )
        self.model.eval()
        
        self.device = next(self.model.parameters()).device
        self.model_id = self.config.model_id
        print(f"Model loaded on {self.device}")
        
        # Check if this is a harmony format model
        if self.is_harmony_format_model():
            print(f"Note: {self.model_id} uses harmony format. Evaluating without web search.")
        
        # Check if this model was trained with topk=5
        if self.is_topk5_model() and self.config.topk != 5:
            print(f"⚠️  Warning: {self.model_id} was trained with topk=5, but current topk={self.config.topk}")
            print(f"   Consider using --topk 5 for fair comparison with original paper results.")
    
    def is_harmony_format_model(self) -> bool:
        """Check if the model uses OpenAI harmony format."""
        return self.model_id in HARMONY_FORMAT_MODELS
    
    def is_topk5_model(self) -> bool:
        """Check if the model was trained with topk=5."""
        return self.model_id in TOPK5_MODELS
        
    def search(self, query: str) -> str:
        """Call search API."""
        if self.config.no_search:
            return "[Search disabled]"
            
        try:
            payload = {
                "queries": [query],
                "topk": self.config.topk,
                "return_scores": True
            }
            response = requests.post(self.config.search_url, json=payload, timeout=30)
            
            if response.status_code != 200:
                return f"[Search error: status {response.status_code}]"
            
            results = response.json()['result'][0]
            
            formatted = ""
            for idx, doc in enumerate(results):
                content = doc['document']['contents']
                title = content.split("\n")[0]
                text = "\n".join(content.split("\n")[1:])
                formatted += f"Doc {idx+1}(Title: {title}) {text}\n"
            
            return formatted if formatted else "[No results found]"
            
        except Exception as e:
            return f"[Search error: {e}]"
    
    def generate_response(self, prompt: str) -> Tuple[str, str]:
        """Generate response with multi-turn search."""
        # Check if this is a harmony format model (e.g., gpt-oss)
        if self.is_harmony_format_model():
            return self.generate_response_harmony(prompt)
        
        full_response = ""
        current_prompt = prompt
        
        for turn in range(self.config.max_turns):
            inputs = self.tokenizer(
                current_prompt, 
                return_tensors="pt",
                truncation=True,
                max_length=8192
            ).to(self.device)
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=self.config.max_new_tokens,
                    temperature=self.config.temperature,
                    do_sample=self.config.do_sample,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                )
            
            new_tokens = outputs[0][inputs['input_ids'].shape[1]:]
            new_text = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
            
            if self.config.verbose:
                print(f"    [Turn {turn+1} raw generation: {len(new_text)} chars]", flush=True)
            
            # Post-process: Find first </search> or </answer> and truncate
            search_pos = new_text.find('</search>')
            answer_pos = new_text.find('</answer>')
            
            if search_pos >= 0 and answer_pos >= 0:
                if search_pos < answer_pos:
                    new_text = new_text[:search_pos + len('</search>')]
                else:
                    new_text = new_text[:answer_pos + len('</answer>')]
            elif search_pos >= 0:
                new_text = new_text[:search_pos + len('</search>')]
            elif answer_pos >= 0:
                new_text = new_text[:answer_pos + len('</answer>')]
            
            full_response += new_text
            
            if '</answer>' in new_text:
                break
            
            if '</search>' in new_text and not self.config.no_search:
                search_query = extract_search_query(new_text)
                if search_query:
                    if self.config.verbose:
                        print(f"    [Calling search API: {search_query[:50]}...]", flush=True)
                    search_results = self.search(search_query)
                    search_text = f"\n<information>\n{search_results}\n</information>\n"
                    full_response += search_text
                    current_prompt = prompt + full_response
                else:
                    break
            else:
                break
        
        extracted_answer = extract_answer(full_response) or ""
        return full_response, extracted_answer
    
    def generate_response_harmony(self, prompt: str) -> Tuple[str, str]:
        """Generate response for gpt-oss models using harmony format.
        
        Note: gpt-oss models use OpenAI's harmony format which is different from
        ICRL's <search> tags. This implementation provides direct generation
        without tool use. The answer is extracted from the response text.
        """
        inputs = self.tokenizer(
            prompt, 
            return_tensors="pt",
            truncation=True,
            max_length=8192
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=2048,  # Allow longer generation for reasoning
                temperature=self.config.temperature,
                do_sample=self.config.do_sample,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
        
        new_tokens = outputs[0][inputs['input_ids'].shape[1]:]
        response = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
        
        if self.config.verbose:
            print(f"    [Harmony model direct generation: {len(response)} chars]")
        
        # Try to extract answer from <answer> tags if present
        extracted = extract_answer(response)
        if extracted:
            return response, extracted
        
        # For harmony format, look for common answer patterns
        # Try to find the final answer in various formats
        answer = response.strip()
        
        # If the response is very long, try to get the last meaningful sentence
        if len(answer) > 500:
            # Look for patterns like "The answer is X" or "Therefore, X"
            patterns = [
                r'(?:the answer is|answer:|therefore)[:\s]+([^.\n]+)',
                r'(?:is|are|was|were)\s+([^.\n]+?)(?:\.|$)',
            ]
            for pattern in patterns:
                match = re.search(pattern, answer, re.IGNORECASE)
                if match:
                    answer = match.group(1).strip()
                    break
            else:
                # Just take the last line if nothing else works
                lines = answer.strip().split('\n')
                answer = lines[-1].strip() if lines else answer[:200]
        
        return response, answer
    
    def evaluate_sample(self, sample: Dict) -> Dict[str, Any]:
        """Evaluate a single sample."""
        question = sample.get('question', '')
        golden_answers = sample.get('golden_answers', [])
        
        if hasattr(golden_answers, 'tolist'):
            golden_answers = golden_answers.tolist()
        
        # Get ground truth from reward_model if available
        reward_model = sample.get('reward_model', {})
        if isinstance(reward_model, dict) and 'ground_truth' in reward_model:
            gt = reward_model['ground_truth']
            if isinstance(gt, dict) and 'target' in gt:
                target = gt['target']
                if hasattr(target, 'tolist'):
                    golden_answers = target.tolist()
                else:
                    golden_answers = list(target) if isinstance(target, (list, tuple)) else [target]
        
        # Build prompt
        prompt_data = sample.get('prompt', [])
        if hasattr(prompt_data, 'tolist'):
            prompt_data = prompt_data.tolist()
        
        if prompt_data and isinstance(prompt_data[0], dict):
            prompt = self.tokenizer.apply_chat_template(
                prompt_data, 
                tokenize=False, 
                add_generation_prompt=True
            )
        else:
            # Build default prompt
            prompt_text = f"""Answer the given question. You must conduct reasoning inside <think> and </think> first every time you get new information. After reasoning, if you find you lack some knowledge, you can call a search engine by <search> query </search> and it will return the top searched results between <information> and </information>. You can search as many times as your want. If you find no further external knowledge needed, you can directly provide the answer inside <answer> and </answer>, without detailed illustrations. For example, <answer> February 14 </answer>.

Question: {question}"""
            
            if self.tokenizer.chat_template:
                messages = [{"role": "user", "content": prompt_text}]
                prompt = self.tokenizer.apply_chat_template(
                    messages, 
                    tokenize=False, 
                    add_generation_prompt=True
                )
            else:
                prompt = prompt_text
        
        # Generate response
        start_time = time.time()
        full_response, predicted_answer = self.generate_response(prompt)
        generation_time = time.time() - start_time
        
        # Evaluate
        em = exact_match_score(predicted_answer, golden_answers)
        subem = substring_match_score(predicted_answer, golden_answers)
        f1 = f1_score(predicted_answer, golden_answers)
        
        # Count search calls
        search_count = full_response.count('<information>')
        think_count = full_response.count('<think>')
        
        return {
            'question': question,
            'golden_answers': golden_answers,
            'predicted_answer': predicted_answer,
            'em': em,
            'subem': subem,
            'f1': f1,
            'search_count': search_count,
            'think_count': think_count,
            'generation_time': generation_time,
            'full_response': full_response,
        }
    
    def evaluate_dataset(self, data_file: str) -> Dict[str, Any]:
        """Evaluate on a dataset."""
        print(f"\nEvaluating on {data_file}...")
        
        df = pd.read_parquet(data_file)
        
        if self.config.num_samples:
            df = df.head(self.config.num_samples)
        
        print(f"Loaded {len(df)} samples")
        
        results = []
        print("Starting evaluation...")
        
        for idx in tqdm(range(len(df)), desc="Evaluating"):
            sample = df.iloc[idx].to_dict()
            sample['idx'] = idx
            
            try:
                result = self.evaluate_sample(sample)
                result['idx'] = idx
                result['data_source'] = sample.get('data_source', 'unknown')
                results.append(result)
                
                if self.config.verbose:
                    print(f"\n--- Sample {idx} ---")
                    print(f"Question: {result['question']}")
                    print(f"Golden: {result['golden_answers']}")
                    print(f"Predicted: {result['predicted_answer']}")
                    print(f"EM: {result['em']}, SubEM: {result['subem']}, F1: {result['f1']:.3f}")
                    
            except Exception as e:
                print(f"Error evaluating sample {idx}: {e}")
                results.append({
                    'idx': idx,
                    'question': sample.get('question', ''),
                    'error': str(e),
                    'em': False,
                    'subem': False,
                    'f1': 0.0,
                })
        
        # Compute metrics
        em_scores = [r['em'] for r in results if 'error' not in r]
        subem_scores = [r['subem'] for r in results if 'error' not in r]
        f1_scores = [r['f1'] for r in results if 'error' not in r]
        search_counts = [r['search_count'] for r in results if 'error' not in r]
        think_counts = [r['think_count'] for r in results if 'error' not in r]
        gen_times = [r['generation_time'] for r in results if 'error' not in r]
        
        metrics = {
            'model_id': self.config.model_id,
            'dataset': os.path.basename(data_file),
            'num_samples': len(results),
            'em_accuracy': np.mean(em_scores) if em_scores else 0.0,
            'subem_accuracy': np.mean(subem_scores) if subem_scores else 0.0,
            'avg_f1': np.mean(f1_scores) if f1_scores else 0.0,
            'avg_search_count': np.mean(search_counts) if search_counts else 0.0,
            'avg_think_count': np.mean(think_counts) if think_counts else 0.0,
            'avg_generation_time': np.mean(gen_times) if gen_times else 0.0,
            'total_time': sum(gen_times) if gen_times else 0.0,
        }
        
        print(f"\nResults for {os.path.basename(data_file)}:")
        for key, value in metrics.items():
            if isinstance(value, float):
                print(f"  {key}: {value:.4f}")
            else:
                print(f"  {key}: {value}")
        
        return {'metrics': metrics, 'results': results}


def main():
    parser = argparse.ArgumentParser(description='Evaluate HuggingFace models on ICRL-style tasks')
    parser.add_argument('--model_id', type=str, required=True,
                        help='HuggingFace model ID or local path')
    parser.add_argument('--data_file', type=str, default=None,
                        help='Path to evaluation parquet file')
    parser.add_argument('--data_dir', type=str, default=None,
                        help='Directory containing evaluation parquet files')
    parser.add_argument('--output_dir', type=str, default='./eval_results',
                        help='Output directory for results')
    parser.add_argument('--search_url', type=str, default='http://127.0.0.1:8000/retrieve',
                        help='Search API URL')
    parser.add_argument('--topk', type=int, default=3,
                        help='Number of search results')
    parser.add_argument('--max_turns', type=int, default=6,
                        help='Maximum search turns')
    parser.add_argument('--max_new_tokens', type=int, default=512,
                        help='Maximum new tokens per generation')
    parser.add_argument('--temperature', type=float, default=0.1,
                        help='Sampling temperature')
    parser.add_argument('--num_samples', type=int, default=None,
                        help='Number of samples to evaluate')
    parser.add_argument('--no_search', action='store_true',
                        help='Disable search')
    parser.add_argument('--verbose', action='store_true',
                        help='Verbose output')
    parser.add_argument('--cache_dir', type=str, default=None,
                        help='HuggingFace cache directory')
    
    args = parser.parse_args()
    
    config = EvalConfig(
        model_id=args.model_id,
        data_file=args.data_file,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        search_url=args.search_url,
        topk=args.topk,
        max_turns=args.max_turns,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        num_samples=args.num_samples,
        no_search=args.no_search,
        verbose=args.verbose,
        cache_dir=args.cache_dir,
    )
    
    # Create output directory
    os.makedirs(config.output_dir, exist_ok=True)
    
    # Initialize evaluator
    evaluator = HFModelEvaluator(config)
    
    # Get data files
    data_files = []
    if config.data_file:
        data_files.append(config.data_file)
    if config.data_dir:
        for f in os.listdir(config.data_dir):
            if f.endswith('.parquet'):
                data_files.append(os.path.join(config.data_dir, f))
    
    if not data_files:
        print("Error: No data files specified")
        sys.exit(1)
    
    print(f"Found {len(data_files)} dataset(s) to evaluate")
    
    # Evaluate
    all_results = {}
    for data_file in data_files:
        result = evaluator.evaluate_dataset(data_file)
        
        # Save individual results
        dataset_name = os.path.basename(data_file).replace('.parquet', '')
        result_file = os.path.join(config.output_dir, f"{dataset_name}_results.json")
        with open(result_file, 'w') as f:
            json.dump(result, f, indent=2, default=str)
        
        all_results[dataset_name] = result['metrics']
    
    # Save summary
    model_name = config.model_id.replace('/', '_')
    summary = {
        'model_id': config.model_id,
        'config': {
            'max_turns': config.max_turns,
            'temperature': config.temperature,
            'topk': config.topk,
            'no_search': config.no_search,
        },
        'datasets': list(all_results.values())
    }
    
    summary_file = os.path.join(config.output_dir, 'evaluation_summary.json')
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\nEvaluation complete. Summary saved to {summary_file}")


if __name__ == '__main__':
    main()
