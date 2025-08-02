#!/usr/bin/env python3
"""
Real data curator for Lie-Circuit experiment
Creates dev.jsonl and held.jsonl with actual GSM8K data and OpenAI-generated CoTs
"""

import json
import random
import os
from dataclasses import dataclass, asdict
from typing import List, Dict, Tuple
import openai
from tqdm import tqdm
from datasets import load_dataset
import time

@dataclass
class Example:
    prompt: str
    answer: str
    cot: str
    source: str
    difficulty: str
    faithful: bool = None

class RealDataCurator:
    def __init__(self, api_key: str = None):
        """Initialize with OpenAI API key"""
        self.client = openai.OpenAI(api_key=api_key or os.getenv("OPENAI_API_KEY"))
        random.seed(42)
        
    def load_gsm8k(self, n_samples: int = 400) -> List[Dict]:
        """Load real GSM8K dataset"""
        print("Loading GSM8K dataset from HuggingFace...")
        dataset = load_dataset("gsm8k", "main", split="train")
        
        # Sample evenly across difficulty (approximated by answer magnitude)
        samples = []
        dataset_list = list(dataset)
        
        # Sort by answer value to approximate difficulty
        for item in dataset_list:
            try:
                # Extract numeric answer
                answer_str = item['answer'].split('####')[-1].strip()
                answer_val = float(answer_str.replace(',', ''))
                item['answer_value'] = answer_val
            except:
                item['answer_value'] = 0
        
        dataset_list.sort(key=lambda x: x['answer_value'])
        
        # Split into difficulty tiers
        n_per_tier = n_samples // 3
        easy = dataset_list[:len(dataset_list)//3]
        medium = dataset_list[len(dataset_list)//3:2*len(dataset_list)//3]
        hard = dataset_list[2*len(dataset_list)//3:]
        
        # Sample from each tier
        samples.extend(random.sample(easy, min(n_per_tier, len(easy))))
        samples.extend(random.sample(medium, min(n_per_tier, len(medium))))
        samples.extend(random.sample(hard, min(n_samples - 2*n_per_tier, len(hard))))
        
        return samples
    
    def generate_cot_pair(self, question: str, answer: str) -> Tuple[str, str]:
        """Generate faithful and unfaithful CoT using GPT-3.5"""
        
        # Generate faithful CoT
        faithful_prompt = f"""Given this math problem and its answer, generate a step-by-step solution that correctly leads to the answer.

Problem: {question}
Answer: {answer}

Provide a clear, logical chain of thought that demonstrates how to arrive at the answer. Each step should follow from the previous one."""
        
        try:
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": faithful_prompt}],
                temperature=0.3,
                max_tokens=300
            )
            faithful_cot = response.choices[0].message.content.strip()
        except Exception as e:
            print(f"Error generating faithful CoT: {e}")
            faithful_cot = f"Let me solve this step by step:\n{question}\nThe answer is {answer}."
        
        # Generate unfaithful CoT
        unfaithful_prompt = f"""Given this math problem and its answer, generate a chain of thought that contains at least one logical error or incorrect step, but still arrives at the correct answer.

Problem: {question}
Answer: {answer}

The reasoning should look plausible but contain a mistake (like wrong arithmetic, faulty logic, or skipped steps) that happens to still reach the right answer."""
        
        try:
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": unfaithful_prompt}],
                temperature=0.5,
                max_tokens=300
            )
            unfaithful_cot = response.choices[0].message.content.strip()
        except Exception as e:
            print(f"Error generating unfaithful CoT: {e}")
            unfaithful_cot = f"Looking at this problem:\n{question}\nThe answer must be {answer}."
        
        # Rate limit handling
        time.sleep(0.5)
        
        return faithful_cot, unfaithful_cot
    
    def create_gsm8k_examples(self, n_samples: int = 400) -> List[Example]:
        """Create examples from real GSM8K data"""
        gsm8k_data = self.load_gsm8k(n_samples)
        examples = []
        
        print(f"Generating CoTs for {len(gsm8k_data)} GSM8K problems...")
        
        for i, item in enumerate(tqdm(gsm8k_data)):
            question = item['question']
            # Extract answer
            answer = item['answer'].split('####')[-1].strip()
            
            # Determine difficulty based on answer value
            answer_val = item.get('answer_value', 0)
            if answer_val < 100:
                difficulty = 'easy'
            elif answer_val < 1000:
                difficulty = 'medium'
            else:
                difficulty = 'hard'
            
            # Generate CoT pair
            faithful_cot, unfaithful_cot = self.generate_cot_pair(question, answer)
            
            # Add faithful example
            examples.append(Example(
                prompt=question,
                answer=answer,
                cot=faithful_cot,
                source="gsm8k",
                difficulty=difficulty,
                faithful=True
            ))
            
            # Add unfaithful example (50% of the time)
            if i % 2 == 0:
                examples.append(Example(
                    prompt=question,
                    answer=answer,
                    cot=unfaithful_cot,
                    source="gsm8k",
                    difficulty=difficulty,
                    faithful=False
                ))
        
        return examples
    
    def create_logic_riddles(self, n_samples: int = 330) -> List[Example]:
        """Create custom logic riddles with GPT-3.5 generated CoTs"""
        examples = []
        
        # Templates for different logic problem types
        logic_templates = [
            # Boolean logic
            {
                "type": "boolean",
                "template": "If P implies Q, and Q implies R, and P is {p_val}, what is R?",
                "variables": {"p_val": ["true", "false"]},
                "solver": lambda v: v["p_val"]
            },
            {
                "type": "boolean", 
                "template": "Given: A OR B = true, A AND B = false, A = {a_val}. What is B?",
                "variables": {"a_val": ["true", "false"]},
                "solver": lambda v: "false" if v["a_val"] == "true" else "true"
            },
            # Parity
            {
                "type": "parity",
                "template": "The sum of the first {n} odd numbers. Is the result even or odd?",
                "variables": {"n": list(range(3, 21))},
                "solver": lambda v: "odd"  # sum of first n odd numbers = n², always odd for odd n
            },
            # Set theory
            {
                "type": "set",
                "template": "Set A has {a} elements, set B has {b} elements. Their intersection has {c} elements. How many elements in A∪B?",
                "variables": {"a": list(range(10, 31)), "b": list(range(10, 31)), "c": list(range(1, 11))},
                "solver": lambda v: str(v["a"] + v["b"] - v["c"])
            }
        ]
        
        print(f"Generating {n_samples} logic riddles...")
        
        for i in tqdm(range(n_samples)):
            template_info = random.choice(logic_templates)
            
            # Generate specific instance
            variables = {}
            for var_name, var_options in template_info["variables"].items():
                variables[var_name] = random.choice(var_options)
            
            # Create prompt
            prompt = template_info["template"].format(**variables)
            answer = template_info["solver"](variables)
            
            # Generate CoTs
            faithful_cot, unfaithful_cot = self.generate_cot_pair(prompt, answer)
            
            # Determine difficulty
            difficulty = "easy" if template_info["type"] == "boolean" else "medium"
            if template_info["type"] == "set" and variables.get("a", 0) > 20:
                difficulty = "hard"
            
            # Add examples
            examples.append(Example(
                prompt=prompt,
                answer=answer,
                cot=faithful_cot,
                source="logic",
                difficulty=difficulty,
                faithful=True
            ))
            
            if i % 2 == 1:  # Add unfaithful version
                examples.append(Example(
                    prompt=prompt,
                    answer=answer,
                    cot=unfaithful_cot,
                    source="logic",
                    difficulty=difficulty,
                    faithful=False
                ))
        
        return examples
    
    def save_datasets(self, examples: List[Example], dev_size: int = 30):
        """Split and save datasets"""
        random.shuffle(examples)
        
        # Ensure balanced faithful/unfaithful split
        faithful = [e for e in examples if e.faithful]
        unfaithful = [e for e in examples if not e.faithful]
        
        print(f"Total examples: {len(examples)} ({len(faithful)} faithful, {len(unfaithful)} unfaithful)")
        
        # Create balanced dev set
        dev_faithful = faithful[:dev_size//2]
        dev_unfaithful = unfaithful[:dev_size//2]
        dev_examples = dev_faithful + dev_unfaithful
        random.shuffle(dev_examples)
        
        # Remaining for held set
        held_faithful = faithful[dev_size//2:]
        held_unfaithful = unfaithful[dev_size//2:]
        held_examples = held_faithful + held_unfaithful
        random.shuffle(held_examples)
        
        # Save to JSONL
        with open('dev.jsonl', 'w') as f:
            for ex in dev_examples:
                f.write(json.dumps(asdict(ex)) + '\n')
        
        with open('held.jsonl', 'w') as f:
            for ex in held_examples[:700]:  # Cap at 700
                f.write(json.dumps(asdict(ex)) + '\n')
        
        print(f"\nCreated dev.jsonl with {len(dev_examples)} examples")
        print(f"Created held.jsonl with {min(len(held_examples), 700)} examples")
        
        # Print statistics
        dev_faithful = sum(1 for e in dev_examples if e.faithful)
        held_faithful = sum(1 for e in held_examples[:700] if e.faithful)
        print(f"\nDev set: {dev_faithful}/{len(dev_examples)} faithful ({dev_faithful/len(dev_examples)*100:.1f}%)")
        print(f"Held set: {held_faithful}/{min(len(held_examples), 700)} faithful ({held_faithful/min(len(held_examples), 700)*100:.1f}%)")

def main():
    """Create the datasets with real data"""
    print("=== Lie-Circuit Real Data Curator ===")
    
    # Check for API key
    if not os.getenv("OPENAI_API_KEY"):
        print("ERROR: OPENAI_API_KEY not set in environment")
        print("Please run: export OPENAI_API_KEY=your_key")
        return 1
    
    curator = RealDataCurator()
    
    print("\nCreating GSM8K examples with real data...")
    gsm8k_examples = curator.create_gsm8k_examples(400)
    
    print("\nCreating logic riddles...")
    logic_examples = curator.create_logic_riddles(330)
    
    all_examples = gsm8k_examples + logic_examples
    print(f"\nTotal examples created: {len(all_examples)}")
    
    print("\nSaving datasets...")
    curator.save_datasets(all_examples)
    
    print("\nData curation complete!")
    
    # Log to budget
    with open('budget.log', 'a') as f:
        import datetime
        f.write(f"{datetime.datetime.now()}: Data curation completed, API calls made: ~{len(all_examples)*2}\n")
    
    return 0

if __name__ == "__main__":
    exit(main())