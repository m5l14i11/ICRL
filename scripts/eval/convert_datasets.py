#!/usr/bin/env python3
"""
Convert various HuggingFace QA datasets to the ICRL parquet format.

Supported datasets:
- TriviaQA (mandarjoshi/trivia_qa)
- PopQA (akariasai/PopQA)
- HotpotQA (hotpotqa/hotpot_qa)
- MuSiQue (dgslibisey/MuSiQue)
- NQ (google-research-datasets/nq_open)
- 2WikiMultihopQA

Usage:
    python convert_datasets.py --dataset triviaqa --output_dir ./data/triviaqa_eval
    python convert_datasets.py --dataset popqa --output_dir ./data/popqa_eval --max_samples 1000
"""

import os
import argparse
from typing import List, Dict, Any, Optional
from datasets import load_dataset
import numpy as np


# System prompt template for ICRL
SYSTEM_PROMPT = """Solve the following problem step by step. You must conduct reasoning inside <think> and </think> every time you get new information. After reasoning, if you find you lack some knowledge, you can call a search engine by <search> query </search> and it will return the top searched results between <information> and </information>. You can search as many times as you want. Finally, you should provide the answer inside <answer> and </answer> without detailed illustrations. For example, <answer> Beijing </answer>.

Each reasoning step should be wrapped with <think> your thought here </think>.

When you call the search tool, the query should be placed inside <search> query text here </search>.

And the result of the search should be wrapped with <information> search results here </information>.

The last part of your response should be in the following format: <answer> The final answer goes here. </answer>"""


def normalize_question(question: str) -> str:
    """Normalize question format."""
    question = question.strip()
    if question and question[-1] != '?':
        question += '?'
    return question


def create_data_item(
    idx: int,
    question: str,
    golden_answers: List[str],
    data_source: str,
    split: str = 'test',
    template_type: str = 'zeroshot',
    num_examples: int = 0
) -> Dict[str, Any]:
    """Create a data item in ICRL format."""
    question = normalize_question(question)
    
    prompt_content = f"{SYSTEM_PROMPT}\n\n\nProblem: {question}"
    
    return {
        "id": f"{split}_{idx}",
        "question": question,
        "golden_answers": np.array(golden_answers, dtype=object),
        "data_source": data_source,
        "prompt": np.array([{
            "role": "user",
            "content": prompt_content
        }], dtype=object),
        "ability": "fact-reasoning",
        "reward_model": {
            "style": "rule",
            "ground_truth": {
                "target": np.array(golden_answers, dtype=object)
            }
        },
        "extra_info": {
            "index": idx,
            "num_examples": num_examples,
            "split": split,
            "template_type": template_type
        }
    }


def convert_triviaqa(max_samples: Optional[int] = None, split: str = 'validation') -> List[Dict]:
    """Convert TriviaQA dataset.
    
    Dataset: mandarjoshi/trivia_qa
    Format: question, answer (dict with 'value' and 'aliases')
    """
    print("Loading TriviaQA dataset...")
    try:
        dataset = load_dataset("mandarjoshi/trivia_qa", "rc", split=split, trust_remote_code=True)
    except Exception as e:
        print(f"Failed to load with 'rc' config, trying 'unfiltered': {e}")
        try:
            dataset = load_dataset("mandarjoshi/trivia_qa", "unfiltered", split=split, trust_remote_code=True)
        except Exception as e2:
            print(f"Failed to load with 'unfiltered' config, trying default: {e2}")
            dataset = load_dataset("mandarjoshi/trivia_qa", split=split, trust_remote_code=True)
    
    if max_samples:
        dataset = dataset.select(range(min(max_samples, len(dataset))))
    
    data_items = []
    for idx, item in enumerate(dataset):
        question = item['question']
        # TriviaQA has answer with 'value' and 'aliases'
        answer_obj = item.get('answer', {})
        if isinstance(answer_obj, dict):
            main_answer = answer_obj.get('value', '')
            aliases = answer_obj.get('aliases', [])
            golden_answers = [main_answer] + list(aliases) if main_answer else list(aliases)
        else:
            golden_answers = [str(answer_obj)]
        
        golden_answers = [a for a in golden_answers if a]  # Remove empty
        if not golden_answers:
            continue
            
        data_items.append(create_data_item(
            idx=idx,
            question=question,
            golden_answers=golden_answers,
            data_source='triviaqa',
            split='test'
        ))
    
    print(f"Converted {len(data_items)} samples from TriviaQA")
    return data_items


def convert_popqa(max_samples: Optional[int] = None) -> List[Dict]:
    """Convert PopQA dataset.
    
    Dataset: akariasai/PopQA
    Format: question, possible_answers (list)
    """
    print("Loading PopQA dataset...")
    dataset = load_dataset("akariasai/PopQA", split='test', trust_remote_code=True)
    
    if max_samples:
        dataset = dataset.select(range(min(max_samples, len(dataset))))
    
    data_items = []
    for idx, item in enumerate(dataset):
        question = item['question']
        # PopQA has possible_answers as a list
        golden_answers = item.get('possible_answers', [])
        if isinstance(golden_answers, str):
            golden_answers = [golden_answers]
        
        golden_answers = [a for a in golden_answers if a]
        if not golden_answers:
            continue
            
        data_items.append(create_data_item(
            idx=idx,
            question=question,
            golden_answers=golden_answers,
            data_source='popqa',
            split='test'
        ))
    
    print(f"Converted {len(data_items)} samples from PopQA")
    return data_items


def convert_hotpotqa(max_samples: Optional[int] = None, split: str = 'validation') -> List[Dict]:
    """Convert HotpotQA dataset.
    
    Dataset: hotpotqa/hotpot_qa
    Format: question, answer (string)
    """
    print("Loading HotpotQA dataset...")
    try:
        dataset = load_dataset("hotpotqa/hotpot_qa", "fullwiki", split=split, trust_remote_code=True)
    except Exception as e:
        print(f"Failed with 'fullwiki' config: {e}")
        try:
            dataset = load_dataset("hotpotqa/hotpot_qa", "distractor", split=split, trust_remote_code=True)
        except Exception as e2:
            print(f"Failed with 'distractor' config: {e2}")
            dataset = load_dataset("hotpotqa/hotpot_qa", split=split, trust_remote_code=True)
    
    if max_samples:
        dataset = dataset.select(range(min(max_samples, len(dataset))))
    
    data_items = []
    for idx, item in enumerate(dataset):
        question = item['question']
        answer = item.get('answer', '')
        
        if not answer:
            continue
            
        golden_answers = [answer] if isinstance(answer, str) else list(answer)
        
        data_items.append(create_data_item(
            idx=idx,
            question=question,
            golden_answers=golden_answers,
            data_source='hotpotqa',
            split='test'
        ))
    
    print(f"Converted {len(data_items)} samples from HotpotQA")
    return data_items


def convert_musique(max_samples: Optional[int] = None, split: str = 'validation') -> List[Dict]:
    """Convert MuSiQue dataset.
    
    Dataset: dgslibisey/MuSiQue (or official MuSiQue)
    Format: question, answer/answers
    """
    print("Loading MuSiQue dataset...")
    try:
        dataset = load_dataset("dgslibisey/MuSiQue", split=split, trust_remote_code=True)
    except Exception as e:
        print(f"Failed to load dgslibisey/MuSiQue: {e}")
        try:
            # Try alternative source
            dataset = load_dataset("tau/musique", split=split, trust_remote_code=True)
        except Exception as e2:
            print(f"Failed to load tau/musique: {e2}")
            # Return empty if can't load
            return []
    
    if max_samples:
        dataset = dataset.select(range(min(max_samples, len(dataset))))
    
    data_items = []
    for idx, item in enumerate(dataset):
        question = item.get('question', '')
        # MuSiQue may have 'answer' or 'answers'
        answer = item.get('answer', item.get('answers', ''))
        
        if not question or not answer:
            continue
        
        if isinstance(answer, list):
            golden_answers = answer
        else:
            golden_answers = [answer]
        
        golden_answers = [a for a in golden_answers if a]
        if not golden_answers:
            continue
            
        data_items.append(create_data_item(
            idx=idx,
            question=question,
            golden_answers=golden_answers,
            data_source='musique',
            split='test'
        ))
    
    print(f"Converted {len(data_items)} samples from MuSiQue")
    return data_items


def convert_nq_open(max_samples: Optional[int] = None, split: str = 'validation') -> List[Dict]:
    """Convert Natural Questions Open dataset.
    
    Dataset: google-research-datasets/nq_open or RUC-NLPIR/FlashRAG_datasets
    Format: question, answer (list)
    """
    print("Loading NQ Open dataset...")
    try:
        dataset = load_dataset("RUC-NLPIR/FlashRAG_datasets", "nq", split='test', trust_remote_code=True)
    except Exception as e:
        print(f"Failed to load FlashRAG NQ: {e}")
        try:
            dataset = load_dataset("google-research-datasets/nq_open", split=split, trust_remote_code=True)
        except Exception as e2:
            print(f"Failed to load nq_open: {e2}")
            return []
    
    if max_samples:
        dataset = dataset.select(range(min(max_samples, len(dataset))))
    
    data_items = []
    for idx, item in enumerate(dataset):
        question = item.get('question', '')
        # NQ has 'answer' as list or 'golden_answers'
        golden_answers = item.get('golden_answers', item.get('answer', []))
        
        if isinstance(golden_answers, str):
            golden_answers = [golden_answers]
        
        golden_answers = [a for a in golden_answers if a]
        if not golden_answers or not question:
            continue
            
        data_items.append(create_data_item(
            idx=idx,
            question=question,
            golden_answers=golden_answers,
            data_source='nq',
            split='test'
        ))
    
    print(f"Converted {len(data_items)} samples from NQ Open")
    return data_items


def convert_2wikimultihopqa(max_samples: Optional[int] = None, split: str = 'validation') -> List[Dict]:
    """Convert 2WikiMultihopQA dataset.
    
    Dataset: RUC-NLPIR/FlashRAG_datasets (2wikimultihopqa config)
    Format: question, golden_answers
    """
    print("Loading 2WikiMultihopQA dataset...")
    
    # Map split names: validation -> dev for FlashRAG
    flashrag_split = 'dev' if split == 'validation' else split
    
    try:
        dataset = load_dataset("RUC-NLPIR/FlashRAG_datasets", "2wikimultihopqa", split=flashrag_split)
        print(f"Loaded from FlashRAG_datasets with split '{flashrag_split}'")
    except Exception as e:
        print(f"Failed to load from FlashRAG: {e}")
        # Try alternative sources
        try:
            dataset = load_dataset("TIGER-Lab/2WikiMultihopQA", split=split)
        except Exception as e2:
            print(f"Failed to load TIGER-Lab/2WikiMultihopQA: {e2}")
            return []
    
    if max_samples:
        dataset = dataset.select(range(min(max_samples, len(dataset))))
    
    data_items = []
    for idx, item in enumerate(dataset):
        question = item.get('question', '')
        # FlashRAG uses 'golden_answers', others may use 'answer'
        golden_answers = item.get('golden_answers', item.get('answer', []))
        
        if isinstance(golden_answers, str):
            golden_answers = [golden_answers]
        
        if not question or not golden_answers:
            continue
        
        golden_answers = [a for a in golden_answers if a]
        
        if not golden_answers:
            continue
            
        data_items.append(create_data_item(
            idx=idx,
            question=question,
            golden_answers=golden_answers,
            data_source='2wikimultihopqa',
            split='test'
        ))
    
    print(f"Converted {len(data_items)} samples from 2WikiMultihopQA")
    return data_items


def convert_bamboogle(max_samples: Optional[int] = None) -> List[Dict]:
    """Convert Bamboogle dataset.
    
    Dataset: chiayewken/bamboogle
    Format: Question, Answer (note capital letters)
    """
    print("Loading Bamboogle dataset...")
    try:
        dataset = load_dataset("chiayewken/bamboogle", split='test')
        print("Loaded from chiayewken/bamboogle")
    except Exception as e:
        print(f"Failed to load chiayewken/bamboogle: {e}")
        try:
            dataset = load_dataset("cmriat/bamboogle", split='test')
            print("Loaded from cmriat/bamboogle")
        except Exception as e2:
            print(f"Failed to load cmriat/bamboogle: {e2}")
            return []
    
    if max_samples:
        dataset = dataset.select(range(min(max_samples, len(dataset))))
    
    data_items = []
    for idx, item in enumerate(dataset):
        # Note: bamboogle uses 'Question' and 'Answer' with capital letters
        question = item.get('Question', item.get('question', ''))
        answer = item.get('Answer', item.get('answer', ''))
        
        if not question or not answer:
            continue
        
        golden_answers = [answer] if isinstance(answer, str) else list(answer)
        
        data_items.append(create_data_item(
            idx=idx,
            question=question,
            golden_answers=golden_answers,
            data_source='bamboogle',
            split='test'
        ))
    
    print(f"Converted {len(data_items)} samples from Bamboogle")
    return data_items


def convert_custom_jsonl(file_path: str, data_source: str = 'custom', max_samples: Optional[int] = None) -> List[Dict]:
    """Convert a custom JSONL file.
    
    Expected format (each line):
    {"question": "...", "answer": "..." or "answers": ["...", "..."]}
    """
    import json
    
    print(f"Loading custom dataset from {file_path}...")
    
    data_items = []
    with open(file_path, 'r') as f:
        for idx, line in enumerate(f):
            if max_samples and idx >= max_samples:
                break
            
            item = json.loads(line.strip())
            question = item.get('question', '')
            answer = item.get('answer', item.get('answers', []))
            
            if not question or not answer:
                continue
            
            if isinstance(answer, str):
                golden_answers = [answer]
            else:
                golden_answers = list(answer)
            
            golden_answers = [a for a in golden_answers if a]
            if not golden_answers:
                continue
                
            data_items.append(create_data_item(
                idx=idx,
                question=question,
                golden_answers=golden_answers,
                data_source=data_source,
                split='test'
            ))
    
    print(f"Converted {len(data_items)} samples from {file_path}")
    return data_items


DATASET_CONVERTERS = {
    'triviaqa': convert_triviaqa,
    'popqa': convert_popqa,
    'hotpotqa': convert_hotpotqa,
    'musique': convert_musique,
    'nq': convert_nq_open,
    '2wikimultihopqa': convert_2wikimultihopqa,
    'bamboogle': convert_bamboogle,
}


def save_to_parquet(data_items: List[Dict], output_path: str):
    """Save data items to parquet format."""
    import pyarrow as pa
    import pyarrow.parquet as pq
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Convert to pyarrow table
    table = pa.Table.from_pylist(data_items)
    pq.write_table(table, output_path)
    
    print(f"Saved {len(data_items)} samples to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Convert QA datasets to ICRL format")
    parser.add_argument('--dataset', type=str, required=True,
                        choices=list(DATASET_CONVERTERS.keys()) + ['all', 'custom'],
                        help='Dataset to convert')
    parser.add_argument('--output_dir', type=str, default='./data/eval',
                        help='Output directory')
    parser.add_argument('--max_samples', type=int, default=None,
                        help='Maximum number of samples to convert')
    parser.add_argument('--split', type=str, default='validation',
                        help='Dataset split to use (validation/test)')
    parser.add_argument('--custom_file', type=str, default=None,
                        help='Path to custom JSONL file')
    parser.add_argument('--data_source', type=str, default='custom',
                        help='Data source name for custom dataset')
    
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    if args.dataset == 'all':
        # Convert all datasets
        for name, converter in DATASET_CONVERTERS.items():
            try:
                if name in ['popqa', 'bamboogle']:
                    data_items = converter(max_samples=args.max_samples)
                else:
                    data_items = converter(max_samples=args.max_samples, split=args.split)
                
                if data_items:
                    output_path = os.path.join(args.output_dir, f'{name}_eval.parquet')
                    save_to_parquet(data_items, output_path)
            except Exception as e:
                print(f"Failed to convert {name}: {e}")
                
    elif args.dataset == 'custom':
        if not args.custom_file:
            print("Error: --custom_file is required for custom dataset")
            return
        
        data_items = convert_custom_jsonl(
            args.custom_file, 
            data_source=args.data_source,
            max_samples=args.max_samples
        )
        
        if data_items:
            output_path = os.path.join(args.output_dir, f'{args.data_source}_eval.parquet')
            save_to_parquet(data_items, output_path)
    else:
        converter = DATASET_CONVERTERS[args.dataset]
        
        if args.dataset in ['popqa', 'bamboogle']:
            data_items = converter(max_samples=args.max_samples)
        else:
            data_items = converter(max_samples=args.max_samples, split=args.split)
        
        if data_items:
            output_path = os.path.join(args.output_dir, f'{args.dataset}_eval.parquet')
            save_to_parquet(data_items, output_path)


if __name__ == '__main__':
    main()
