#!/usr/bin/env python3
"""
Script to analyze repository commit information from Hugging Face datasets.
Extracts repo names, commit hashes, and creation timestamps.

Usage Examples:
    # Analyze SWE-bench dataset (test split)
    python zshi_tools/analyze_repo_commits.py "princeton-nlp/SWE-bench" --split test
    
    # Analyze SWE-smith dataset with train split
    python zshi_tools/analyze_repo_commits.py "SWE-bench/SWE-smith" --split train
    
    # Save output to specific file
    python zshi_tools/analyze_repo_commits.py "princeton-nlp/SWE-bench" --output swe_bench_analysis.json
    
    # Run in quiet mode (suppress detailed output)
    python zshi_tools/analyze_repo_commits.py "princeton-nlp/SWE-bench" --quiet
    
    # Analyze other datasets
    python zshi_tools/analyze_repo_commits.py "SWE-bench/SWE-bench_Lite" --split test
"""

import json
from collections import defaultdict
from datetime import datetime
from datasets import load_dataset
from pathlib import Path
from typing import Dict, List, Tuple, Any
from tqdm import tqdm


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


def analyze_dataset_repos(dataset_name: str, split: str = "test") -> Dict[str, Dict[str, Any]]:
    """
    Analyze repository information from a Hugging Face dataset.
    
    Args:
        dataset_name: Name of the Hugging Face dataset
        split: Dataset split to analyze (default: "test")
    
    Returns:
        Dictionary with repo names as keys and analysis results as values
    """
    print(f"Loading dataset: {dataset_name} (split: {split})")
    
    try:
        dataset = load_dataset(dataset_name, split=split)
        print(f"Loaded {len(dataset)} instances")
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return {}
    
    # Check if required columns exist
    required_columns = ['repo', 'base_commit', 'created_at']
    missing_columns = [col for col in required_columns if col not in dataset.column_names]
    if missing_columns:
        print(f"Missing required columns: {missing_columns}")
        print(f"Available columns: {dataset.column_names}")
        return {}
    
    # Extract repository information
    repo_data = defaultdict(list)
    
    print("Extracting repository commit information...")
    for instance in tqdm(dataset, desc="Processing instances", unit="instance"):
        repo_name = instance['repo']
        base_commit = instance['base_commit']
        created_at = instance['created_at']
        
        repo_data[repo_name].append((base_commit, created_at))
    
    # Process each repository
    result = {}
    print(f"\nFound {len(repo_data)} unique repositories")
    
    for repo_name, commit_pairs in tqdm(repo_data.items(), desc="Processing repositories", unit="repo"):
        # Remove duplicates and sort by created_at
        unique_pairs = list(set(commit_pairs))
        
        # Sort by created_at timestamp
        try:
            sorted_pairs = sorted(unique_pairs, key=lambda x: parse_created_at(x[1]))
        except Exception as e:
            print(f"Warning: Error sorting pairs for repo {repo_name}: {e}")
            sorted_pairs = unique_pairs
        
        # Get the earliest and latest commits
        earliest_commit = sorted_pairs[0][0] if sorted_pairs else None
        earliest_timestamp = sorted_pairs[0][1] if sorted_pairs else None
        latest_commit = sorted_pairs[-1][0] if sorted_pairs else None
        latest_timestamp = sorted_pairs[-1][1] if sorted_pairs else None
        
        # Calculate time interval
        time_interval_years = None
        if earliest_timestamp and latest_timestamp and len(sorted_pairs) > 1:
            try:
                earliest_dt = parse_created_at(earliest_timestamp)
                latest_dt = parse_created_at(latest_timestamp)
                time_interval_days = (latest_dt - earliest_dt).days
                time_interval_years = round(time_interval_days / 365.25, 2)  # Account for leap years
            except Exception:
                time_interval_years = None
        
        result[repo_name] = {
            'oldest_base_commit': earliest_commit,
            'oldest_created_at': earliest_timestamp,
            'latest_base_commit': latest_commit,
            'latest_created_at': latest_timestamp,
            'time_interval_years': time_interval_years,
            'commit_pairs': sorted_pairs,
            'total_instances': len(commit_pairs),
            'unique_commits': len(unique_pairs)
        }
    
    return result


def print_repo_analysis(repo_analysis: Dict[str, Dict[str, Any]]):
    """Print repository analysis results in a readable format"""
    if not repo_analysis:
        print("No repository data to display")
        return
    
    print(f"\n{'='*80}")
    print("Repository Analysis Summary")
    print(f"{'='*80}")
    
    print(f"Total repositories: {len(repo_analysis)}")
    
    # Sort repositories by number of instances (descending)
    sorted_repos = sorted(repo_analysis.items(), 
                         key=lambda x: x[1]['total_instances'], 
                         reverse=True)
    
    print(f"\nAll repositories by instance count:")
    print(f"{'Repository':<40} {'Instances':<10} {'Unique Commits':<15} {'Start Date':<12} {'End Date':<12} {'Time Span (years)':<17}")
    print("-" * 116)
    
    for repo_name, data in sorted_repos:
        time_span = str(data['time_interval_years']) if data['time_interval_years'] is not None else 'N/A'
        start_date = data['oldest_created_at'][:10] if data['oldest_created_at'] else 'N/A'
        end_date = data['latest_created_at'][:10] if data['latest_created_at'] else 'N/A'
        print(f"{repo_name:<40} {data['total_instances']:<10} {data['unique_commits']:<15} {start_date:<12} {end_date:<12} {time_span:<17}")
    
    # Show detailed info for first few repos
    print(f"\nDetailed information for first 3 repositories:")
    for i, (repo_name, data) in enumerate(sorted_repos[:3]):
        print(f"\n{i+1}. Repository: {repo_name}")
        print(f"   Oldest commit: {data['oldest_base_commit']}")
        print(f"   Oldest timestamp: {data['oldest_created_at']}")
        print(f"   Latest commit: {data['latest_base_commit']}")
        print(f"   Latest timestamp: {data['latest_created_at']}")
        if data['time_interval_years'] is not None:
            print(f"   Time interval: {data['time_interval_years']} years")
        else:
            print(f"   Time interval: N/A (single commit or parsing error)")
        print(f"   Total instances: {data['total_instances']}")
        print(f"   Unique commits: {data['unique_commits']}")
        
        if len(data['commit_pairs']) <= 5:
            print(f"   All commit pairs:")
            for commit, timestamp in data['commit_pairs']:
                print(f"     {commit[:12]} - {timestamp}")
        else:
            print(f"   First 3 commit pairs:")
            for commit, timestamp in data['commit_pairs'][:3]:
                print(f"     {commit[:12]} - {timestamp}")
            print(f"   Last 3 commit pairs:")
            for commit, timestamp in data['commit_pairs'][-3:]:
                print(f"     {commit[:12]} - {timestamp}")
            print(f"     ... and {len(data['commit_pairs']) - 6} more in between")


def save_repo_analysis(repo_analysis: Dict[str, Dict[str, Any]], output_path: str):
    """Save repository analysis to JSON file"""
    try:
        with open(output_path, 'w') as f:
            json.dump(repo_analysis, f, indent=2, default=str)
        print(f"\nAnalysis saved to: {output_path}")
    except Exception as e:
        print(f"Error saving analysis: {e}")


def main():
    """Main function to run repository analysis"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Analyze repository commit information from Hugging Face datasets")
    parser.add_argument("dataset", help="Hugging Face dataset name (e.g., 'princeton-nlp/SWE-bench')")
    parser.add_argument("--split", default="test", help="Dataset split to analyze (default: test)")
    parser.add_argument("--output", "-o", help="Output JSON file path")
    parser.add_argument("--quiet", "-q", action="store_true", help="Suppress detailed output")
    
    args = parser.parse_args()
    
    # Analyze the dataset
    repo_analysis = analyze_dataset_repos(args.dataset, args.split)
    
    if not repo_analysis:
        print("No analysis results to display")
        return
    
    # Print results
    if not args.quiet:
        print_repo_analysis(repo_analysis)
    
    # Save to file if requested
    if args.output:
        save_repo_analysis(repo_analysis, args.output)
    else:
        # Generate default filename
        dataset_name = args.dataset.replace("/", "_").replace("-", "_")
        default_output = f"repo_analysis_{dataset_name}_{args.split}.json"
        save_repo_analysis(repo_analysis, default_output)


if __name__ == "__main__":
    main()
