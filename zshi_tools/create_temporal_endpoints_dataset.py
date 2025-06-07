#!/usr/bin/env python3
"""
Script to create a temporal endpoints dataset by selecting the oldest and newest 
examples for each repository from SWE-bench datasets.

This creates a dataset containing temporal boundaries for each repository,
useful for studying evolution patterns and time-based analysis.

Usage Examples:
    python zshi_tools/create_temporal_endpoints_dataset.py princeton-nlp/SWE-bench_Verified
"""

import json
from collections import defaultdict
from datetime import datetime
from datasets import load_dataset, Dataset
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
from tqdm import tqdm
import os
from huggingface_hub import HfApi, login


def parse_created_at(created_at_str: str) -> datetime:
    """Parse ISO format timestamp string to datetime object"""
    try:
        # Handle format like "2023-04-16T14:24:42Z"
        return datetime.fromisoformat(created_at_str.replace('Z', '+00:00'))
    except ValueError:
        # Fallback for other formats
        try:
            return datetime.strptime(created_at_str, "%Y-%m-%dT%H:%M:%S")
        except ValueError:
            print(f"Warning: Could not parse timestamp: {created_at_str}")
            return datetime.min


def create_temporal_endpoints_dataset(dataset_name: str, split: str = "test") -> Tuple[List[Dict], Dict[str, Any]]:
    """
    Create a temporal endpoints dataset by selecting oldest and newest examples per repository.
    
    Args:
        dataset_name: Name of the Hugging Face dataset
        split: Dataset split to analyze (default: "test")
    
    Returns:
        Tuple of (new_dataset_examples, metadata)
    """
    print(f"Loading dataset: {dataset_name} (split: {split})")
    
    try:
        dataset = load_dataset(dataset_name, split=split)
        print(f"Loaded {len(dataset)} instances")
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return [], {}
    
    # Check if required columns exist
    required_columns = ['repo', 'created_at']
    missing_columns = [col for col in required_columns if col not in dataset.column_names]
    if missing_columns:
        print(f"Missing required columns: {missing_columns}")
        print(f"Available columns: {dataset.column_names}")
        return [], {}
    
    # Group examples by repository
    repo_examples = defaultdict(list)
    
    print("Grouping examples by repository...")
    for i, instance in enumerate(tqdm(dataset, desc="Processing instances", unit="instance")):
        repo_name = instance['repo']
        # Store the full example with its index for easy retrieval
        repo_examples[repo_name].append((i, instance))
    
    # Filter out repositories with fewer than 10 instances
    original_repo_count = len(repo_examples)
    filtered_repo_examples = {repo: examples for repo, examples in repo_examples.items() if len(examples) >= 10}
    filtered_repo_count = len(filtered_repo_examples)
    excluded_repo_count = original_repo_count - filtered_repo_count
    
    print(f"\nFiltering repositories:")
    print(f"  - Original repositories: {original_repo_count}")
    print(f"  - Repositories with ≥10 instances: {filtered_repo_count}")
    print(f"  - Excluded repositories: {excluded_repo_count}")
    
    # Process each repository to find temporal endpoints
    temporal_endpoints = []
    repo_stats = {}
    
    print(f"\nProcessing {len(filtered_repo_examples)} qualifying repositories...")
    
    for repo_name, examples in tqdm(filtered_repo_examples.items(), desc="Finding temporal endpoints", unit="repo"):
        # Sort examples by created_at timestamp
        try:
            sorted_examples = sorted(examples, key=lambda x: parse_created_at(x[1]['created_at']))
        except Exception as e:
            print(f"Warning: Error sorting examples for repo {repo_name}: {e}")
            sorted_examples = examples
        
        # Get oldest and newest examples
        oldest_idx, oldest_example = sorted_examples[0]
        newest_idx, newest_example = sorted_examples[-1]
        
        # Calculate time span
        try:
            oldest_dt = parse_created_at(oldest_example['created_at'])
            newest_dt = parse_created_at(newest_example['created_at'])
            time_span_days = (newest_dt - oldest_dt).days
            time_span_years = round(time_span_days / 365.25, 2)
        except Exception:
            time_span_years = None
        
        # Add examples without modification (keep original structure)
        temporal_endpoints.append(dict(oldest_example))
        temporal_endpoints.append(dict(newest_example))
        
        repo_stats[repo_name] = {
            'total_examples': len(examples),
            'oldest_created_at': oldest_example['created_at'],
            'newest_created_at': newest_example['created_at'],
            'time_span_years': time_span_years
        }
    
    # Create metadata
    metadata = {
        'source_dataset': dataset_name,
        'source_split': split,
        'creation_timestamp': datetime.now().isoformat(),
        'total_repositories': original_repo_count,
        'filtered_repositories': filtered_repo_count,
        'excluded_repositories': excluded_repo_count,
        'min_instances_filter': 10,
        'total_temporal_endpoints': len(temporal_endpoints),
        'repository_stats': repo_stats
    }
    
    print(f"\nCreated temporal endpoints dataset:")
    print(f"  - Original dataset: {len(dataset)} examples")
    print(f"  - Qualifying repositories: {len(filtered_repo_examples)}")
    print(f"  - Temporal endpoints: {len(temporal_endpoints)} examples")
    
    return temporal_endpoints, metadata


def save_temporal_endpoints_dataset(examples: List[Dict], metadata: Dict[str, Any], output_path: str):
    """Save the temporal endpoints dataset to JSON file"""
    try:
        # Save examples only (without additional metadata structure)
        with open(output_path, 'w') as f:
            json.dump(examples, f, indent=2, default=str)
        print(f"Temporal endpoints dataset saved to: {output_path}")
        
        # Also save metadata separately
        metadata_path = output_path.replace('.json', '_metadata.json')
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2, default=str)
        print(f"Metadata saved to: {metadata_path}")
        
        # Also save as Hugging Face dataset format
        if examples:
            hf_dataset = Dataset.from_list(examples)
            hf_output_dir = output_path.replace('.json', '_hf_dataset')
            hf_dataset.save_to_disk(hf_output_dir)
            print(f"Hugging Face dataset format saved to: {hf_output_dir}")
            
    except Exception as e:
        print(f"Error saving dataset: {e}")


def push_to_huggingface(examples: List[Dict], metadata: Dict[str, Any], repo_id: str, private: bool = False):
    """Push the temporal endpoints dataset to Hugging Face Hub"""
    try:
        # Check for HF token in environment
        hf_token = os.getenv('HF_TOKEN') or os.getenv('HUGGINGFACE_TOKEN')
        if not hf_token:
            print("Warning: No Hugging Face token found in environment variables (HF_TOKEN or HUGGINGFACE_TOKEN)")
            print("You may need to login manually or set the token")
        else:
            login(token=hf_token)
            print("Successfully authenticated with Hugging Face")
        
        # Create dataset
        if not examples:
            print("No examples to push")
            return
            
        hf_dataset = Dataset.from_list(examples)
        
        # Add metadata as dataset info
        dataset_info = {
            "description": f"Temporal endpoints dataset created from {metadata['source_dataset']}",
            "source_dataset": metadata['source_dataset'],
            "source_split": metadata['source_split'],
            "creation_timestamp": metadata['creation_timestamp'],
            "total_repositories": metadata.get('filtered_repositories', metadata['total_repositories']),
            "min_instances_filter": metadata.get('min_instances_filter', 'none'),
            "total_temporal_endpoints": metadata['total_temporal_endpoints']
        }
        
        # Push to hub
        print(f"Pushing dataset to Hugging Face Hub: {repo_id}")
        hf_dataset.push_to_hub(
            repo_id=repo_id,
            private=private,
            commit_message=f"Add temporal endpoints dataset from {metadata['source_dataset']}"
        )
        
        # Also push metadata as a separate file
        api = HfApi()
        metadata_content = json.dumps(metadata, indent=2, default=str)
        api.upload_file(
            path_or_fileobj=metadata_content.encode(),
            path_in_repo="metadata.json",
            repo_id=repo_id,
            repo_type="dataset",
            commit_message="Add dataset metadata"
        )
        
        print(f"Successfully pushed dataset to: https://huggingface.co/datasets/{repo_id}")
        
    except Exception as e:
        print(f"Error pushing to Hugging Face Hub: {e}")
        print("Make sure you have the required permissions and correct repo_id format (username/dataset-name)")


def print_dataset_summary(examples: List[Dict], metadata: Dict[str, Any]):
    """Print summary of the created temporal endpoints dataset"""
    if not examples:
        print("No examples to summarize")
        return
    
    print(f"\n{'='*80}")
    print("Temporal Endpoints Dataset Summary")
    print(f"{'='*80}")
    
    print(f"Source: {metadata['source_dataset']} ({metadata['source_split']} split)")
    print(f"Created: {metadata['creation_timestamp']}")
    print(f"Total repositories: {metadata['total_repositories']}")
    print(f"Total temporal endpoints: {metadata['total_temporal_endpoints']}")
    
    # Analyze temporal spans
    repo_stats = metadata['repository_stats']
    time_spans = [stats['time_span_years'] for stats in repo_stats.values() 
                  if stats['time_span_years'] is not None and stats['time_span_years'] > 0]
    
    if time_spans:
        print(f"\nTime span statistics:")
        print(f"  - Average time span: {sum(time_spans) / len(time_spans):.2f} years")
        print(f"  - Maximum time span: {max(time_spans):.2f} years")
        print(f"  - Minimum time span: {min(time_spans):.2f} years")
        print(f"  - Repositories with >1 year span: {len([s for s in time_spans if s > 1])}")
    
    # Show top repositories by time span
    sorted_repos = sorted(repo_stats.items(), 
                         key=lambda x: x[1]['time_span_years'] or 0, 
                         reverse=True)
    
    print(f"\nTop 10 repositories by time span:")
    print(f"{'Repository':<40} {'Examples':<10} {'Time Span (years)':<17} {'Start Date':<12} {'End Date':<12}")
    print("-" * 101)
    
    for repo_name, stats in sorted_repos[:10]:
        time_span = f"{stats['time_span_years']:.2f}" if stats['time_span_years'] is not None else 'N/A'
        start_date = stats['oldest_created_at'][:10] if stats['oldest_created_at'] else 'N/A'
        end_date = stats['newest_created_at'][:10] if stats['newest_created_at'] else 'N/A'
        print(f"{repo_name:<40} {stats['total_examples']:<10} {time_span:<17} {start_date:<12} {end_date:<12}")


def main():
    """Main function to create temporal endpoints dataset"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Create temporal endpoints dataset from SWE-bench datasets")
    parser.add_argument("dataset", help="Hugging Face dataset name (e.g., 'princeton-nlp/SWE-bench_Verified')")
    parser.add_argument("--split", default="test", help="Dataset split to analyze (default: test)")
    parser.add_argument("--output", "-o", help="Output JSON file path")
    parser.add_argument("--quiet", "-q", action="store_true", help="Suppress detailed output")
    parser.add_argument("--push-to-hub", help="Override default Hub repo (default: ZhengyanShi/SWE-bench_Verified_Temporal_${num_repos})")
    parser.add_argument("--private", action="store_true", help="Make the Hub dataset private")
    parser.add_argument("--no-push", action="store_true", help="Skip pushing to Hugging Face Hub")
    
    args = parser.parse_args()
    
    # Create temporal endpoints dataset
    examples, metadata = create_temporal_endpoints_dataset(args.dataset, args.split)
    
    if not examples:
        print("No examples created")
        return
    
    # Print summary
    if not args.quiet:
        print_dataset_summary(examples, metadata)
    
    # Generate dataset name based on number of repositories
    num_repos = metadata.get('filtered_repositories', metadata['total_repositories'])
    dataset_name = f"SWE-bench_Verified_Temporal_{num_repos}"
    
    # Save dataset locally
    if args.output:
        output_path = args.output
    else:
        output_path = f"{dataset_name}.json"
    
    save_temporal_endpoints_dataset(examples, metadata, output_path)
    
    # Push to Hugging Face Hub (unless explicitly disabled)
    if not args.no_push:
        if args.push_to_hub:
            hub_repo_id = args.push_to_hub
        else:
            hub_repo_id = f"ZhengyanShi/{dataset_name}"
        
        push_to_huggingface(examples, metadata, hub_repo_id, args.private)


if __name__ == "__main__":
    main()
