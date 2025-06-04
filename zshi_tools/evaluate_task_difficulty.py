#!/usr/bin/env python3
"""
Script to evaluate the difficulty level of tasks in SWE-bench and related datasets using LLM API.
"""

import json
import argparse
import re
import os
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional
from datasets import load_dataset
from openai import AsyncAzureOpenAI
from azure.identity import DefaultAzureCredential, ChainedTokenCredential, AzureCliCredential, get_bearer_token_provider
from tqdm.asyncio import tqdm
import asyncio
import time
import signal
import sys

class TaskDifficultyEvaluator:
    def __init__(self, model_type: str = "4o", output_dir: str = "evaluation_results", max_concurrent: int = 5):
        self.model_type = model_type
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.client = self._setup_azure_client()
        self.results_by_dataset = {}
        self.current_dataset = None
        self.max_concurrent = max_concurrent
        self.semaphore = asyncio.Semaphore(max_concurrent)
        self.setup_logging()
        self.setup_signal_handlers()
        
    def setup_logging(self):
        """Setup logging configuration"""
        log_file = self.output_dir / f"evaluation_log_{self.model_type}.log"
        
        # Create file handler for detailed logging
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)
        file_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(file_formatter)
        
        # Create console handler for minimal output
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.WARNING)  # Only warnings and errors to console
        console_formatter = logging.Formatter('%(levelname)s: %(message)s')
        console_handler.setFormatter(console_formatter)
        
        # Setup logger
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)

    def setup_signal_handlers(self):
        """Setup signal handlers for graceful shutdown"""
        def signal_handler(signum, frame):
            self.logger.info("Received interrupt signal. Saving results and exiting...")
            self.save_all_results()
            sys.exit(0)
            
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)

    def _setup_azure_client(self) -> AsyncAzureOpenAI:
        """Setup async Azure OpenAI client based on model type"""
        scope = "api://trapi/.default"
        credential = get_bearer_token_provider(ChainedTokenCredential(
            AzureCliCredential(),
            DefaultAzureCredential(
                exclude_cli_credential=True,
                exclude_environment_credential=True,
                exclude_shared_token_cache_credential=True,
                exclude_developer_cli_credential=True,
                exclude_powershell_credential=True,
                exclude_interactive_browser_credential=True,
                exclude_visual_studio_code_credentials=True,
                managed_identity_client_id=os.environ.get("DEFAULT_IDENTITY_CLIENT_ID"),
            )
        ), scope)

        if self.model_type == "4o":
            model_name = 'gpt-4o'
            model_version = '2024-05-13'
            instance = 'gcr/preview'
            api_version = '2024-10-21'
        elif self.model_type == "o3":
            model_name = 'o3'
            model_version = '2025-04-16'
            instance = 'msrne/shared'
            api_version = '2025-04-01-preview'
        elif self.model_type == "o3-mini":
            model_name = 'o3-mini'
            model_version = '2025-01-31'
            instance = 'msrne/shared'
            api_version = '2025-04-01-preview'
        elif self.model_type == "o4-mini":
            model_name = 'o4-mini'
            model_version = '2025-04-16'
            instance = 'msrne/shared'
            api_version = '2025-04-01-preview'
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")

        deployment_name = re.sub(r'[^a-zA-Z0-9-_]', '', f'{model_name}_{model_version}')
        endpoint = f'https://trapi.research.microsoft.com/{instance}'

        return AsyncAzureOpenAI(
            azure_endpoint=endpoint,
            azure_ad_token_provider=credential,
            api_version=api_version,
        )

    def extract_task_info(self, task: Dict[str, Any], dataset_name: str) -> Dict[str, Any]:
        """Extract relevant information from a task"""
        extracted = {
            'dataset': dataset_name,
            'problem_statement': task.get('problem_statement', ''),
            'patch': task.get('patch', ''),
            'fail_to_pass': task.get('FAIL_TO_PASS', []),
            'pass_to_pass': task.get('PASS_TO_PASS', []),
            'hints_text': task.get('hints_text', ''),
            'test_patch': task.get('test_patch', ''),
            'instance_id': task.get('instance_id', ''),
            'repo': task.get('repo', ''),
        }
        
        # Convert lists to strings if they're lists
        if isinstance(extracted['fail_to_pass'], list):
            extracted['fail_to_pass'] = '\n'.join(extracted['fail_to_pass'])
        if isinstance(extracted['pass_to_pass'], list):
            extracted['pass_to_pass'] = '\n'.join(extracted['pass_to_pass'])
            
        return extracted

    def build_evaluation_prompt(self, task_info: Dict[str, Any]) -> str:
        """Build a structured prompt for LLM evaluation"""
        
        difficulty_criteria = """
### 🔧 **LLM Task Difficulty Rating Guide (1 to 10)**

#### **Score 1 (Very Easy)**
* Minimal edit (e.g., typo, variable rename).
* Patch affects ≤3 lines in 1 function.
* FAIL_TO_PASS includes 1 trivial test.

#### **Score 2–3 (Easy)**
* Small logic fix or configuration adjustment.
* Patch affects ≤10 lines in ≤2 functions.
* Tests require basic understanding of input/output or function calls.

#### **Score 4–5 (Moderate)**
* Requires understanding control flow, function logic, or intermediate domain knowledge.
* Patch spans ≤20 lines and up to 2 files.
* FAIL_TO_PASS has ≤5 tests; tests are medium complexity.

#### **Score 6–7 (Challenging)**
* Involves deeper reasoning or dependencies across modules.
* Patch affects multiple functions or files (≥3).
* Non-obvious fix; problem_statement or hints needed to understand root cause.
* Test_patch includes >5 relevant test cases.

#### **Score 8–9 (Very Hard)**
* Complex bug fix or feature requiring architectural understanding.
* Patch ≥50 lines; edits across multiple files and functions (≥4).
* Requires reverse-engineering intent or data flows.
* Tests span multiple cases with varied inputs/edge cases.

#### **Score 10 (Extremely Hard)**
* Requires deep domain-specific knowledge or refactoring.
* Involves new abstractions, protocol changes, or large-scale patch (≥100 lines).
* Model must understand interdependencies, async patterns, or dynamic behaviors.
* FAIL_TO_PASS contains many nuanced edge cases (≥10).
* Human-level reasoning typically needed.
"""

        task_description = f"""
## Task Information

**Dataset:** {task_info['dataset']}
**Repository:** {task_info['repo']}
**Instance ID:** {task_info['instance_id']}

### Problem Statement:
```
{task_info['problem_statement'][:2000]}{'...' if len(task_info['problem_statement']) > 2000 else ''}
```

### Code Patch:
```
{task_info['patch'][:1500]}{'...' if len(task_info['patch']) > 1500 else ''}
```

### Failing Tests (FAIL_TO_PASS):
```
{task_info['fail_to_pass'][:1000]}{'...' if len(task_info['fail_to_pass']) > 1000 else ''}
```

### Passing Tests (PASS_TO_PASS):
```
{task_info['pass_to_pass'][:800]}{'...' if len(task_info['pass_to_pass']) > 800 else ''}
```
"""

        if task_info['hints_text']:
            task_description += f"""
### Hints:
```
{task_info['hints_text'][:500]}{'...' if len(task_info['hints_text']) > 500 else ''}
```
"""

        if task_info['test_patch']:
            task_description += f"""
### Test Patch:
```
{task_info['test_patch'][:800]}{'...' if len(task_info['test_patch']) > 800 else ''}
```
"""

        prompt = f"""
{difficulty_criteria}

{task_description}

## Task:
Based on the difficulty rating guide above, please evaluate this software engineering task and assign a difficulty score from 1 to 10.

Consider the following factors:
1. **Complexity of the patch**: How many lines, functions, and files are affected?
2. **Understanding required**: How much domain knowledge and reasoning is needed?
3. **Test complexity**: How many tests and what types of scenarios do they cover?
4. **Problem clarity**: Is the issue obvious or requires deep investigation?

Please respond in the following format:
**Justification:** [Provide 2-3 sentences explaining your reasoning based on the criteria above]
**Score:** X (where X is a number from 1 to 10)
"""
        return prompt

    async def evaluate_task_difficulty(self, task_info: Dict[str, Any], max_retries: int = 5) -> Dict[str, Any]:
        """Evaluate a single task's difficulty using async LLM API with retry logic"""
        async with self.semaphore:  # Rate limiting
            prompt = self.build_evaluation_prompt(task_info)
            
            for attempt in range(max_retries):
                try:
                    deployment_name = self._get_deployment_name()
                    response = await self.client.chat.completions.create(
                        model=deployment_name,
                        messages=[
                            {
                                "role": "system",
                                "content": "You are an expert software engineer evaluating the difficulty of programming tasks. Provide accurate, consistent difficulty ratings based on the given criteria."
                            },
                            {
                                "role": "user",
                                "content": prompt,
                            },
                        ],
                        temperature=0.0,
                        max_tokens=500,
                    )
                    
                    response_content = response.choices[0].message.content
                    
                    # Parse justification and score
                    justification = ""
                    score = None
                    
                    # Look for justification first
                    justification_match = re.search(r'\*\*Justification:\*\*\s*(.+?)(?=\*\*Score:\*\*|$)', response_content, re.DOTALL)
                    if justification_match:
                        justification = justification_match.group(1).strip()
                    
                    # Look for score
                    score_patterns = [
                        r'\*\*Score:\*\*\s*(\d+)',
                        r'Score:\s*(\d+)',
                        r'score\s*(?:is|:)?\s*(\d+)',
                        r'rating\s*(?:is|:)?\s*(\d+)',
                        r'(\d+)(?:/10)?(?:\s*out\s*of\s*10)?'
                    ]
                    
                    for pattern in score_patterns:
                        score_match = re.search(pattern, response_content, re.IGNORECASE)
                        if score_match:
                            try:
                                potential_score = int(score_match.group(1))
                                if 1 <= potential_score <= 10:
                                    score = potential_score
                                    break
                            except ValueError:
                                continue
                    
                    # Only log to file, not console
                    self.logger.info(f"Task {task_info['instance_id']} evaluated: Score {score}")
                    
                    return {
                        'score': score,
                        'justification': justification,
                        'raw_response': response_content,
                        'attempt': attempt + 1
                    }
                    
                except Exception as e:
                    error_msg = str(e).lower()
                    
                    # Handle rate limiting
                    if "rate" in error_msg or "429" in error_msg:
                        wait_time = min(2 ** attempt * 5, 300)  # Exponential backoff, max 5 minutes
                        self.logger.warning(f"Rate limit hit for task {task_info['instance_id']}, attempt {attempt + 1}/{max_retries}. Waiting {wait_time}s")
                        await asyncio.sleep(wait_time)
                        continue
                    
                    # Handle other API errors
                    elif "timeout" in error_msg or "connection" in error_msg:
                        wait_time = min(2 ** attempt, 60)
                        self.logger.info(f"Connection error for task {task_info['instance_id']}, attempt {attempt + 1}/{max_retries}. Waiting {wait_time}s")
                        await asyncio.sleep(wait_time)
                        continue
                    
                    else:
                        self.logger.info(f"API error for task {task_info['instance_id']}, attempt {attempt + 1}: {e}")
                        if attempt < max_retries - 1:
                            await asyncio.sleep(2 ** attempt)
                            continue
            
            # All retries failed
            self.logger.error(f"Failed to evaluate task {task_info['instance_id']} after {max_retries} attempts")
            return {
                'score': None,
                'justification': f"Error after {max_retries} attempts: {str(e) if 'e' in locals() else 'Unknown error'}",
                'raw_response': "",
                'attempt': max_retries
            }

    def _get_deployment_name(self) -> str:
        """Get deployment name based on model type"""
        if self.model_type == "4o":
            return re.sub(r'[^a-zA-Z0-9-_]', '', 'gpt-4o_2024-05-13')
        elif self.model_type == "o3":
            return re.sub(r'[^a-zA-Z0-9-_]', '', 'o3_2025-04-16')
        elif self.model_type == "o3-mini":
            return re.sub(r'[^a-zA-Z0-9-_]', '', 'o3-mini_2025-01-31')
        elif self.model_type == "o4-mini":
            return re.sub(r'[^a-zA-Z0-9-_]', '', 'o4-mini_2025-04-16')

    def load_datasets(self) -> Dict[str, Any]:
        """Load all datasets"""
        datasets = {}

        # try:
        #     self.logger.info("Loading R2E-Gym dataset...")
        #     datasets['r2e_gym'] = load_dataset("R2E-Gym/R2E-Gym-Subset", split="train")
        # except Exception as e:
        #     self.logger.error(f"Failed to load R2E-Gym: {e}")
            
        # try:
        #     self.logger.info("Loading SWE-Gym dataset...")
        #     datasets['swe_gym'] = load_dataset("SWE-Gym/SWE-Gym", split="train")
        # except Exception as e:
        #     self.logger.error(f"Failed to load SWE-Gym: {e}")

        # try:
        #     self.logger.info("Loading SWE-bench dataset...")
        #     datasets['swe_bench'] = load_dataset("SWE-bench/SWE-bench", split="test")
        # except Exception as e:
        #     self.logger.error(f"Failed to load SWE-bench: {e}")
            
        try:
            self.logger.info("Loading SWE-smith dataset...")
            datasets['swe_smith'] = load_dataset("SWE-bench/SWE-smith", split="train")
        except Exception as e:
            self.logger.error(f"Failed to load SWE-smith: {e}")
            
        return datasets

    def save_dataset_results(self, dataset_name: str):
        """Save results for a specific dataset"""
        if dataset_name not in self.results_by_dataset:
            return
            
        output_file = self.output_dir / f"{dataset_name}_difficulty_results_{self.model_type}.json"
        
        try:
            with open(output_file, 'w') as f:
                json.dump(self.results_by_dataset[dataset_name], f, indent=2)
            self.logger.info(f"Saved {len(self.results_by_dataset[dataset_name])} results for {dataset_name} to {output_file}")
        except Exception as e:
            self.logger.error(f"Failed to save results for {dataset_name}: {e}")
    
    def save_all_results(self):
        """Save all results for all datasets"""
        for dataset_name in self.results_by_dataset.keys():
            self.save_dataset_results(dataset_name)
        
        # Also save a combined summary
        summary_file = self.output_dir / f"evaluation_summary_{self.model_type}.json"
        summary = {
            'model_type': self.model_type,
            'datasets': {}
        }
        
        for dataset_name, results in self.results_by_dataset.items():
            valid_scores = [r['difficulty_score'] for r in results if r['difficulty_score'] is not None]
            summary['datasets'][dataset_name] = {
                'total_tasks': len(results),
                'successful_evaluations': len(valid_scores),
                'average_score': sum(valid_scores) / len(valid_scores) if valid_scores else 0,
                'score_distribution': {str(i): valid_scores.count(i) for i in range(1, 11)}
            }
        
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        self.logger.info(f"Saved evaluation summary to {summary_file}")

    async def process_task_batch(self, tasks_batch: List[tuple], dataset_name: str) -> List[Dict[str, Any]]:
        """Process a batch of tasks concurrently"""
        tasks = []
        for i, task in tasks_batch:
            task_info = self.extract_task_info(task, dataset_name)
            tasks.append(self.evaluate_single_task(task_info, i))
        
        return await asyncio.gather(*tasks, return_exceptions=True)

    async def evaluate_single_task(self, task_info: Dict[str, Any], evaluation_index: int) -> Dict[str, Any]:
        """Evaluate a single task and return the complete result"""
        try:
            evaluation = await self.evaluate_task_difficulty(task_info)
            
            result = {
                **task_info,
                'difficulty_score': evaluation['score'],
                'difficulty_justification': evaluation['justification'],
                'llm_response': evaluation['raw_response'],
                'model_used': self.model_type,
                'evaluation_index': evaluation_index,
                'api_attempts': evaluation['attempt']
            }
            
            # Only log to file
            self.logger.info(f"Completed task: {task_info['instance_id']}, Score: {evaluation['score']}")
            return result
            
        except Exception as e:
            self.logger.error(f"Unexpected error processing task {task_info['instance_id']}: {e}")
            return {
                **task_info,
                'difficulty_score': None,
                'difficulty_justification': f"Processing error: {str(e)}",
                'llm_response': "",
                'model_used': self.model_type,
                'evaluation_index': evaluation_index,
                'api_attempts': 0
            }

    async def evaluate_dataset(self, dataset_name: str, dataset: Any, max_samples_per_dataset: int = None) -> None:
        """Evaluate difficulty for all tasks in a single dataset using async processing"""
        self.current_dataset = dataset_name
        print(f"📊 Starting evaluation of {dataset_name} dataset...")
        self.logger.info(f"Starting async evaluation of {dataset_name} dataset...")
        
        if dataset_name not in self.results_by_dataset:
            self.results_by_dataset[dataset_name] = []
        
        dataset_size = len(dataset)
        
        # Filter out tasks with empty problem statements
        filtered_tasks = []
        for i, task in enumerate(dataset):
            problem_statement = task.get('problem_statement', '').strip()
            if problem_statement:  # Only include tasks with non-empty problem statements
                filtered_tasks.append((i, task))
        
        original_size = dataset_size
        dataset_size = len(filtered_tasks)
        
        print(f"📝 Filtered {original_size} tasks -> {dataset_size} tasks with problem statements")
        self.logger.info(f"Filtered {original_size} tasks to {dataset_size} tasks with non-empty problem statements")
        
        if max_samples_per_dataset:
            dataset_size = min(dataset_size, max_samples_per_dataset)
            filtered_tasks = filtered_tasks[:dataset_size]
        
        if not filtered_tasks:
            print(f"⚠️  No valid tasks found in {dataset_name} dataset")
            self.logger.warning(f"No tasks with non-empty problem statements found in {dataset_name}")
            return
        
        # Process in batches to manage memory and save progress
        batch_size = min(20, self.max_concurrent * 2)  # Adjust batch size based on concurrency
        total_batches = (dataset_size + batch_size - 1) // batch_size
        
        # Create overall progress bar for the dataset
        with tqdm(total=dataset_size, desc=f"🔍 {dataset_name}", unit="task", 
                 bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]") as pbar:
            
            for batch_start in range(0, dataset_size, batch_size):
                batch_end = min(batch_start + batch_size, dataset_size)
                tasks_batch = filtered_tasks[batch_start:batch_end]
                
                batch_num = batch_start // batch_size + 1
                self.logger.info(f"Processing batch {batch_num}/{total_batches} for {dataset_name}")
                
                # Process batch concurrently
                batch_results = await self.process_task_batch(tasks_batch, dataset_name)
                
                # Handle results and exceptions
                successful_evaluations = 0
                for result in batch_results:
                    if isinstance(result, Exception):
                        self.logger.error(f"Batch processing error: {result}")
                    else:
                        self.results_by_dataset[dataset_name].append(result)
                        if result.get('difficulty_score') is not None:
                            successful_evaluations += 1
                
                # Update progress bar
                pbar.update(len(tasks_batch))
                pbar.set_postfix({
                    'batch': f"{batch_num}/{total_batches}",
                    'success': f"{successful_evaluations}/{len(tasks_batch)}"
                })
                
                # Save progress after each batch
                self.save_dataset_results(dataset_name)
                
                # Small delay between batches to be nice to the API
                await asyncio.sleep(1)
        
        print(f"✅ Completed evaluation of {dataset_name} dataset")
        self.logger.info(f"Completed async evaluation of {dataset_name} dataset")

    async def evaluate_all_datasets(self, max_samples_per_dataset: int = None) -> None:
        """Evaluate difficulty for all tasks in all datasets using async processing"""
        print(f"🚀 Starting evaluation with model: {self.model_type}")
        print(f"⚡ Max concurrent requests: {self.max_concurrent}")
        
        datasets = self.load_datasets()
        
        if not datasets:
            print("❌ No datasets loaded successfully")
            return
        
        print(f"📁 Loaded {len(datasets)} datasets: {', '.join(datasets.keys())}")
        
        for dataset_name, dataset in datasets.items():
            await self.evaluate_dataset(dataset_name, dataset, max_samples_per_dataset)
        
        # Save final summary
        self.save_all_results()
        print("🎉 All evaluations completed!")
        self.logger.info("All async evaluations completed!")
        
        # Print summary statistics
        self._print_summary()

    def _print_summary(self):
        """Print summary statistics"""
        total_tasks = 0
        total_valid = 0
        
        print(f"\n{'='*60}")
        print(f"EVALUATION SUMMARY - Model: {self.model_type}")
        print(f"{'='*60}")
        
        for dataset_name, results in self.results_by_dataset.items():
            valid_scores = [r['difficulty_score'] for r in results if r['difficulty_score'] is not None]
            total_tasks += len(results)
            total_valid += len(valid_scores)
            
            print(f"\n{dataset_name.upper()}:")
            print(f"  Total tasks: {len(results)}")
            print(f"  Successful evaluations: {len(valid_scores)}")
            print(f"  Success rate: {len(valid_scores)/len(results)*100:.1f}%")
            
            if valid_scores:
                avg_score = sum(valid_scores) / len(valid_scores)
                print(f"  Average difficulty: {avg_score:.2f}")
                print(f"  Score distribution:")
                for score in range(1, 11):
                    count = valid_scores.count(score)
                    if count > 0:
                        percentage = (count / len(valid_scores)) * 100
                        print(f"    Score {score}: {count} tasks ({percentage:.1f}%)")
        
        print(f"\nOVERALL:")
        print(f"  Total tasks processed: {total_tasks}")
        print(f"  Total successful evaluations: {total_valid}")
        print(f"  Overall success rate: {total_valid/total_tasks*100:.1f}%")

def main():
    parser = argparse.ArgumentParser(description="Evaluate task difficulty across SWE datasets using LLM API")
    parser.add_argument("--model", type=str, choices=["4o", "o3", "o3-mini", "o4-mini"], 
                        default="4o", help="LLM model to use for evaluation")
    parser.add_argument("--max-samples", type=int, default=None, 
                        help="Maximum number of samples per dataset (for testing)")
    parser.add_argument("--output-dir", type=str, default="evaluation_results",
                        help="Output directory for results")
    parser.add_argument("--max-concurrent", type=int, default=20,
                        help="Maximum number of concurrent API calls")
    
    args = parser.parse_args()
    
    print("="*60)
    print("🤖 Task Difficulty Evaluator")
    print("="*60)
    print(f"Model: {args.model}")
    print(f"Concurrency: {args.max_concurrent}")
    print(f"Output: {args.output_dir}")
    if args.max_samples:
        print(f"Max samples per dataset: {args.max_samples}")
    print("="*60)
    
    evaluator = TaskDifficultyEvaluator(
        model_type=args.model, 
        output_dir=args.output_dir,
        max_concurrent=args.max_concurrent
    )
    
    # Run the async evaluation
    asyncio.run(evaluator.evaluate_all_datasets(max_samples_per_dataset=args.max_samples))

if __name__ == "__main__":
    main()
