#!/usr/bin/env python3
# filepath: /home/zhengyanshi/project/SWE-smith/merge_problem_statements.py

import json
import os
import argparse
from pathlib import Path
from tqdm import tqdm


def load_json(filepath):
    """Load JSON data from a file."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Error: File not found: {filepath}")
        return None
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON in {filepath}: {e}")
        return None


def extract_problem_statement(metadata_path):
    """Extract problem statement from metadata file."""
    metadata = load_json(metadata_path)
    if metadata is None:
        return None
    
    responses = metadata.get('responses', {})
    return responses.get('problem_statement', None)


def merge_problem_statements(repo, output_path):
    """Merge task instances with their corresponding problem statements."""
    
    # Define paths
    task_insts_path = f"logs/task_insts/{repo}.json"
    issue_gen_base_path = f"logs/issue_gen/{repo}/{repo}"
    
    # Load task instances
    task_instances = load_json(task_insts_path)
    if task_instances is None:
        print(f"Failed to load task instances from {task_insts_path}")
        return False
    
    print(f"Loaded {len(task_instances)} task instances")
    
    # Process each instance
    merged_instances = []
    for instance in tqdm(task_instances, desc="Processing instances"):
        instance_id = instance.get('instance_id')
        if not instance_id:
            print(f"Warning: Instance missing 'instance_id': {instance}")
            merged_instances.append(instance)
            continue
        
        # Find corresponding metadata file
        metadata_path = f"{issue_gen_base_path}/{instance_id}/metadata.json"
        
        if not os.path.exists(metadata_path):
            print(f"Warning: Metadata file not found for {instance_id}: {metadata_path}")
            merged_instances.append(instance)
            continue
        
        # Extract problem statement
        problem_statement = extract_problem_statement(metadata_path)
        
        if problem_statement is None:
            print(f"Warning: No problem statement found for {instance_id}")
            merged_instances.append(instance)
            continue
        
        # Create merged instance
        merged_instance = instance.copy()
        merged_instance['problem_statement'] = problem_statement
        merged_instances.append(merged_instance)
    
    # Ensure output directory exists
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    # Save merged instances
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(merged_instances, f, indent=2, ensure_ascii=False)
        print(f"Successfully saved merged instances to {output_path}")
        return True
    except Exception as e:
        print(f"Error saving to {output_path}: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Merge task instances with problem statements from issue generation logs"
    )
    parser.add_argument(
        "--repo",
        required=True,
        type=str,
        help="Repository name (e.g., astropy__astropy.26d14786)"
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Output file path (default: logs/task_insts/{repo}_ps.json)"
    )

    args = parser.parse_args()
    
    # Set default output path if not provided
    if args.output is None:
        args.output = f"logs/task_insts/{args.repo}_ps.json"
    
    print(f"Processing repository: {args.repo}")
    print(f"Output path: {args.output}")
    
    success = merge_problem_statements(args.repo, args.output)
    
    if success:
        print("Merge completed successfully!")
        return 0
    else:
        print("Merge failed!")
        return 1


if __name__ == "__main__":
    exit(main())