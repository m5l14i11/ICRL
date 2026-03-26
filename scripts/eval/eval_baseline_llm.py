#!/usr/bin/env python3
"""
Baseline LLM Evaluation Script

Evaluates LLMs without search capability in two modes:
1. Direct Inference: Directly answer questions without any special prompting
2. Thinking Mode: Use <think>...</think> for reasoning before answering

This provides baseline results to compare against search-augmented models.

Usage:
    # Direct inference mode
    python eval_baseline_llm.py \
        --model_id "Qwen/Qwen2.5-14B-Instruct" \
        --mode direct \
        --data_dir ./data/eval \
        --output_dir ./eval_results/qwen14b_direct

    # Thinking mode
    python eval_baseline_llm.py \
        --model_id "Qwen/Qwen2.5-14B-Instruct" \
        --mode thinking \
        --data_dir ./data/eval \
        --output_dir ./eval_results/qwen14b_thinking
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
import re
import string


@dataclass
class EvalConfig:
    """Configuration for baseline evaluation."""
    model_id: str
    mode: str  # 'direct' or 'thinking'
    data_file: Optional[str] = None
    data_dir: Optional[str] = None
    output_dir: str = "./eval_results"
    max_new_tokens: int = 1024
    temperature: float = 0.1
    do_sample: bool = True
    device: str = "auto"
    verbose: bool = False
    num_samples: Optional[int] = None
    cache_dir: Optional[str] = None


# ============================================================================
# Prompt templates for different modes
# ============================================================================

# Mode: direct - Just answer the question directly
DIRECT_CHAT_PROMPT = """Answer the following question directly and concisely. Provide only the answer without explanation.

Question: {question}"""

# Mode: cot - Standard Chain-of-Thought prompting (Let's think step by step)
COT_CHAT_PROMPT = """Answer the following question. Let's think step by step, then provide the final answer.

Question: {question}

Let's think step by step:"""

# Mode: thinking - Use <think> tags (ICRL style)
THINKING_CHAT_PROMPT = """Answer the given question. You must conduct reasoning inside <think> and </think> first. After reasoning, provide the final answer inside <answer> and </answer>, without detailed illustrations.

For example:
<think> Let me think about this step by step... </think>
<answer> Beijing </answer>

Question: {question}"""

# Mode: cot_answer - CoT with explicit answer format
COT_ANSWER_CHAT_PROMPT = """Answer the following question. Think through this step by step, then provide your final answer after "Final Answer:".

Question: {question}

Step-by-step reasoning:"""

# All supported modes
SUPPORTED_MODES = ['direct', 'cot', 'thinking', 'cot_answer']

PROMPT_TEMPLATES = {
    'direct': DIRECT_CHAT_PROMPT,
    'cot': COT_CHAT_PROMPT,
    'thinking': THINKING_CHAT_PROMPT,
    'cot_answer': COT_ANSWER_CHAT_PROMPT,
}


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


def extract_direct_answer(text: str) -> str:
    """Extract answer from direct response (first line or sentence)."""
    # Remove any markdown formatting
    text = text.strip()
    
    # If response contains newlines, take the first non-empty line
    lines = [l.strip() for l in text.split('\n') if l.strip()]
    if lines:
        answer = lines[0]
        # Remove common prefixes
        prefixes = ['The answer is', 'Answer:', 'A:', 'The answer:', 'It is', 'It\'s']
        for prefix in prefixes:
            if answer.lower().startswith(prefix.lower()):
                answer = answer[len(prefix):].strip()
                break
        # Remove trailing punctuation
        answer = answer.rstrip('.')
        return answer
    return text


def extract_cot_answer(text: str) -> str:
    """Extract final answer from CoT response."""
    text = text.strip()
    
    # Try to find "Final Answer:" pattern
    final_patterns = [
        r'[Ff]inal [Aa]nswer[:\s]+(.+?)(?:\n|$)',
        r'[Tt]he answer is[:\s]+(.+?)(?:\.|$)',
        r'[Aa]nswer[:\s]+(.+?)(?:\n|$)',
        r'[Tt]herefore[,:\s]+(.+?)(?:\.|$)',
        r'[Ss]o[,:\s]+the answer is[:\s]+(.+?)(?:\.|$)',
    ]
    
    for pattern in final_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return match.group(1).strip()
    
    # Fallback: get the last non-empty line
    lines = [l.strip() for l in text.split('\n') if l.strip()]
    if lines:
        return lines[-1].rstrip('.')
    
    return text


class BaselineLLMEvaluator:
    """Evaluator for baseline LLM without search."""
    
    def __init__(self, config: EvalConfig):
        self.config = config
        self.model = None
        self.tokenizer = None
        self.device = None
        
        self._load_model()
        
    def _load_model(self):
        """Load model and tokenizer from HuggingFace."""
        print(f"Loading model from {self.config.model_id}...")
        print(f"Evaluation mode: {self.config.mode}")
        
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
        print(f"Model loaded on {self.device}")
    
    def build_prompt(self, question: str) -> str:
        """Build prompt based on evaluation mode."""
        # Get the appropriate prompt template
        if self.config.mode not in PROMPT_TEMPLATES:
            raise ValueError(f"Unknown mode: {self.config.mode}. Supported: {SUPPORTED_MODES}")
        
        prompt_template = PROMPT_TEMPLATES[self.config.mode]
        prompt_content = prompt_template.format(question=question)
        
        if self.tokenizer.chat_template:
            messages = [{"role": "user", "content": prompt_content}]
            return self.tokenizer.apply_chat_template(
                messages, 
                tokenize=False, 
                add_generation_prompt=True
            )
        else:
            return prompt_content
    
    def extract_final_answer(self, response: str) -> str:
        """Extract final answer based on mode."""
        if self.config.mode == 'direct':
            return extract_direct_answer(response)
        elif self.config.mode == 'thinking':
            # Try <answer> tags first
            extracted = extract_answer(response)
            if extracted:
                return extracted
            return extract_cot_answer(response)
        elif self.config.mode in ['cot', 'cot_answer']:
            return extract_cot_answer(response)
        else:
            return extract_direct_answer(response)
    
    def generate_response(self, question: str) -> Tuple[str, str]:
        """Generate response from the model."""
        prompt = self.build_prompt(question)
        
        inputs = self.tokenizer(
            prompt, 
            return_tensors="pt",
            truncation=True,
            max_length=4096
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.config.max_new_tokens,
                temperature=self.config.temperature if self.config.temperature > 0 else None,
                do_sample=self.config.do_sample and self.config.temperature > 0,
                pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
        
        new_tokens = outputs[0][inputs['input_ids'].shape[1]:]
        response = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
        
        # Extract answer using the mode-specific extractor
        predicted_answer = self.extract_final_answer(response)
        return response, predicted_answer
    
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
        
        # Generate response
        start_time = time.time()
        full_response, predicted_answer = self.generate_response(question)
        generation_time = time.time() - start_time
        
        # Evaluate
        em = exact_match_score(predicted_answer, golden_answers)
        subem = substring_match_score(predicted_answer, golden_answers)
        f1 = f1_score(predicted_answer, golden_answers)
        
        # Count thinking blocks
        think_count = full_response.count('<think>') if self.config.mode == 'thinking' else 0
        
        return {
            'question': question,
            'golden_answers': golden_answers,
            'predicted_answer': predicted_answer,
            'em': em,
            'subem': subem,
            'f1': f1,
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
        
        for idx in tqdm(range(len(df)), desc=f"Evaluating ({self.config.mode})"):
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
                    if self.config.mode == 'thinking':
                        print(f"Response: {result['full_response'][:500]}...")
                    
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
        think_counts = [r.get('think_count', 0) for r in results if 'error' not in r]
        gen_times = [r['generation_time'] for r in results if 'error' not in r]
        
        metrics = {
            'model_id': self.config.model_id,
            'mode': self.config.mode,
            'dataset': os.path.basename(data_file),
            'num_samples': len(results),
            'em_accuracy': np.mean(em_scores) if em_scores else 0.0,
            'subem_accuracy': np.mean(subem_scores) if subem_scores else 0.0,
            'avg_f1': np.mean(f1_scores) if f1_scores else 0.0,
            'avg_think_count': np.mean(think_counts) if think_counts else 0.0,
            'avg_generation_time': np.mean(gen_times) if gen_times else 0.0,
            'total_time': sum(gen_times) if gen_times else 0.0,
        }
        
        print(f"\nResults for {os.path.basename(data_file)} ({self.config.mode} mode):")
        for key, value in metrics.items():
            if isinstance(value, float):
                print(f"  {key}: {value:.4f}")
            else:
                print(f"  {key}: {value}")
        
        return {'metrics': metrics, 'results': results}


def main():
    parser = argparse.ArgumentParser(description='Baseline LLM Evaluation (Direct/Thinking/CoT modes)')
    parser.add_argument('--model_id', type=str, default='Qwen/Qwen2.5-14B-Instruct',
                        help='HuggingFace model ID or local path')
    parser.add_argument('--mode', type=str, 
                        choices=['direct', 'thinking', 'cot', 'cot_answer'], 
                        default='direct',
                        help='Evaluation mode: direct (no reasoning), thinking (with <think> tags), cot (Let\'s think step by step), cot_answer (step by step with answer format)')
    parser.add_argument('--data_file', type=str, default=None,
                        help='Path to evaluation parquet file')
    parser.add_argument('--data_dir', type=str, default=None,
                        help='Directory containing evaluation parquet files')
    parser.add_argument('--output_dir', type=str, default='./eval_results',
                        help='Output directory for results')
    parser.add_argument('--max_new_tokens', type=int, default=1024,
                        help='Maximum new tokens per generation')
    parser.add_argument('--temperature', type=float, default=0.1,
                        help='Sampling temperature')
    parser.add_argument('--num_samples', type=int, default=None,
                        help='Number of samples to evaluate')
    parser.add_argument('--verbose', action='store_true',
                        help='Verbose output')
    parser.add_argument('--cache_dir', type=str, default=None,
                        help='HuggingFace cache directory')
    
    args = parser.parse_args()
    
    config = EvalConfig(
        model_id=args.model_id,
        mode=args.mode,
        data_file=args.data_file,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        num_samples=args.num_samples,
        verbose=args.verbose,
        cache_dir=args.cache_dir,
    )
    
    # Create output directory
    os.makedirs(config.output_dir, exist_ok=True)
    
    # Initialize evaluator
    evaluator = BaselineLLMEvaluator(config)
    
    # Get data files
    data_files = []
    if config.data_file:
        data_files.append(config.data_file)
    if config.data_dir:
        for f in sorted(os.listdir(config.data_dir)):
            if f.endswith('.parquet'):
                data_files.append(os.path.join(config.data_dir, f))
    
    if not data_files:
        print("Error: No data files specified")
        sys.exit(1)
    
    print(f"Found {len(data_files)} dataset(s) to evaluate")
    print(f"Mode: {config.mode}")
    
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
        'mode': config.mode,
        'config': {
            'max_new_tokens': config.max_new_tokens,
            'temperature': config.temperature,
        },
        'datasets': list(all_results.values())
    }
    
    summary_file = os.path.join(config.output_dir, 'evaluation_summary.json')
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    # Print final summary
    print("\n" + "="*60)
    print(f"Evaluation Complete - {config.mode.upper()} Mode")
    print("="*60)
    
    print(f"\n{'Dataset':<25} {'EM':>10} {'SubEM':>10} {'F1':>10}")
    print("-"*55)
    
    em_all, f1_all = [], []
    for dataset_name, metrics in all_results.items():
        em = metrics['em_accuracy'] * 100
        subem = metrics['subem_accuracy'] * 100
        f1 = metrics['avg_f1'] * 100
        print(f"{dataset_name:<25} {em:>9.1f}% {subem:>9.1f}% {f1:>9.1f}%")
        em_all.append(em)
        f1_all.append(f1)
    
    print("-"*55)
    print(f"{'AVERAGE':<25} {np.mean(em_all):>9.1f}% {'-':>10} {np.mean(f1_all):>9.1f}%")
    
    print(f"\nSummary saved to {summary_file}")


if __name__ == '__main__':
    main()
