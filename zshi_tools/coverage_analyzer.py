import json
import re
from pathlib import Path
from typing import Dict, List, Set, Tuple, Any
from collections import defaultdict
import argparse
import shutil
import os

# Import extract_entities_from_directory from existing module
from swesmith.bug_gen.utils import extract_entities_from_directory
# Import necessary functions for cloning
from swesmith.utils import clone_repo, does_repo_exist
from swesmith.constants import ORG_NAME


def find_code_files(repo_path: str, extensions: List[str] = None) -> Set[str]:
    """
    Recursively find all code files in a repository.
    
    Args:
        repo_path: Path to the repository
        extensions: List of file extensions to consider as code files.
                   If None, defaults to common code file extensions.
    
    Returns:
        Set of relative file paths
    """
    if extensions is None:
        # Include common code file extensions
        extensions = ['.py', '.pyx', '.pxd', '.pyi', '.c', '.cpp', '.cc', '.cxx', '.h', '.hpp']
    
    code_files = set()
    repo_path = Path(repo_path)
    
    # Walk through the repository
    for root, dirs, files in os.walk(repo_path):
        # Skip hidden directories and common non-source directories
        dirs[:] = [d for d in dirs if not d.startswith('.') and d not in ['__pycache__', 'node_modules', 'venv', 'env']]
        
        for file in files:
            file_path = Path(root) / file
            # Check if file has a code extension
            if any(file_path.suffix.lower() == ext for ext in extensions):
                # Get relative path from repo root
                relative_path = file_path.relative_to(repo_path)
                code_files.add(str(relative_path))
    
    return code_files


def parse_patch_for_files(patch: str) -> Set[str]:
    """
    Parse a patch to identify which files have been modified.
    
    Args:
        patch: Git diff patch string
    
    Returns:
        Set of file paths that have been modified
    """
    modified_files = set()
    
    # Split patch into lines
    lines = patch.split('\n')
    
    for line in lines:
        # Look for file headers in the patch
        if line.startswith('--- a/') or line.startswith('+++ b/'):
            if line.startswith('+++ b/'):
                file_path = line[6:]  # Remove '+++ b/' prefix
                # Skip /dev/null (file deletions)
                if file_path != '/dev/null':
                    modified_files.add(file_path)
        elif line.startswith('diff --git'):
            # Alternative format: diff --git a/file b/file
            match = re.match(r'diff --git a/(.*?) b/(.*?)$', line)
            if match:
                # Use the second file path (b/)
                file_path = match.group(2)
                modified_files.add(file_path)
    
    return modified_files


def parse_patch_for_entities(patch: str) -> Dict[str, Set[Tuple[str, str]]]:
    """
    Parse a patch to identify changed entities and their types.
    
    Args:
        patch: Git diff patch string
    
    Returns:
        Dictionary mapping file paths to sets of (entity_name, entity_type) tuples
    """
    changed_entities = defaultdict(set)
    current_file = None
    
    # Split patch into lines
    lines = patch.split('\n')
    
    for i, line in enumerate(lines):
        # Identify file being modified
        if line.startswith('--- a/') or line.startswith('+++ b/'):
            if line.startswith('+++ b/'):
                current_file = line[6:]  # Remove '+++ b/' prefix
        
        # Look for class definitions
        if line.startswith('+') or line.startswith('-'):
            # Skip diff markers
            if line.startswith('+++') or line.startswith('---'):
                continue

            # Extract the actual code line
            code_line = line[1:]

            # Check for class definitions
            class_match = re.match(r'\s*class\s+(\w+)', code_line)
            if class_match and current_file:
                changed_entities[current_file].add((class_match.group(1), 'class'))
            
            # Check for function/method definitions
            func_match = re.match(r'\s*def\s+(\w+)', code_line)
            if func_match and current_file:
                # Try to determine if it's a method by checking indentation
                if line.startswith('+') or line.startswith('-'):
                    indent_match = re.match(r'[+-](\s*)', line)
                    if indent_match and len(indent_match.group(1)) > 0:
                        changed_entities[current_file].add((func_match.group(1), 'method'))
                    else:
                        changed_entities[current_file].add((func_match.group(1), 'function'))
            
            # Check for variable assignments (module-level)
            var_match = re.match(r'^[+-](\w+)\s*=', line)
            if var_match and current_file:
                changed_entities[current_file].add((var_match.group(1), 'variable'))
    
    return dict(changed_entities)


def load_task_instances(folder_path: str) -> List[Dict[str, Any]]:
    """
    Load task instances from the automated pipeline folder.
    
    Args:
        folder_path: Path to the pipeline folder
    
    Returns:
        List of task instances
    """
    folder_path = Path(folder_path)
    repo_name = folder_path.name
    
    # Construct the path to the task instances JSON file
    task_inst_path = folder_path / "task_insts" / f"{repo_name}_ps.json"
    
    if not task_inst_path.exists():
        raise FileNotFoundError(f"Task instances file not found: {task_inst_path}")
    
    with open(task_inst_path, 'r') as f:
        instances = json.load(f)

    return instances


def analyze_entity_coverage(repo_name: str, instances: List[Dict[str, Any]], entity_types: List[str] = None) -> Dict[str, float]:
    """
    Analyze entity coverage from task instances.
    
    Args:
        repo_name: Name of the repository (e.g., 'astropy__astropy')
        instances: List of task instances
        entity_types: List of entity types to analyze. If None, analyzes all types.
    
    Returns:
        Dictionary mapping entity types to coverage percentages
    """
    if entity_types is None:
        entity_types = ['class', 'func', 'object']
    
    # Clone the repository if it doesn't exist
    print(f"Checking if repository {repo_name} exists...")
    assert does_repo_exist(repo_name), f"Repository {repo_name} does not exist in {ORG_NAME}."
    clone_repo(repo_name, org=ORG_NAME)
    print(f"Cloned {repo_name} repository.")
    
    # Find all code files in the repository
    print(f"Finding all code files in {repo_name}...")
    all_code_files = find_code_files(repo_name)
    print(f"Found {len(all_code_files)} code files")
    
    # Extract all entities from the repository for each type
    print(f"Extracting entities from {repo_name}...")
    all_entities_by_type = {}
    
    for entity_type in entity_types:
        print(f"Extracting {entity_type} entities...")
        entities = extract_entities_from_directory(repo_name, entity_type)
        all_entities_by_type[entity_type] = entities
    shutil.rmtree(repo_name)
    print(f"Removed cloned repository {repo_name}.")

    # Count total entities by type
    total_by_type = defaultdict(set)
    
    # For 'class' and 'func' types, use the entities directly
    for entity_type in ['class', 'func']:
        if entity_type in all_entities_by_type:
            for entity in all_entities_by_type[entity_type]:
                # entity.file_path is relative to repo
                total_by_type[entity_type].add((entity.file_path, entity.src_node.name))
    
    # For 'object' type, combine both classes and functions
    if 'object' in entity_types:
        if 'object' in all_entities_by_type:
            for entity in all_entities_by_type['object']:
                total_by_type['object'].add((entity.file_path, entity.src_node.name))
    
    # Analyze patches to find changed entities and files
    changed_by_type = defaultdict(set)
    changed_files = set()
    
    for instance in instances:
        patch = instance.get('patch', '')
        if patch:
            # Track changed files
            modified_files = parse_patch_for_files(patch)
            changed_files.update(modified_files)
            
            # Track changed entities
            changed_entities = parse_patch_for_entities(patch)
            
            for file_path, entities in changed_entities.items():
                for entity_name, entity_type in entities:
                    # Map parsed entity types to our standard types
                    if entity_type == 'class':
                        changed_by_type['class'].add((file_path, entity_name))
                        changed_by_type['object'].add((file_path, entity_name))
                    elif entity_type in ['function', 'method']:
                        changed_by_type['func'].add((file_path, entity_name))
                        changed_by_type['object'].add((file_path, entity_name))
    
    # Calculate coverage percentages
    coverage = {}
    
    # Entity coverage
    for entity_type in entity_types:
        total_count = len(total_by_type[entity_type])
        changed_count = len(changed_by_type[entity_type])
        
        if total_count > 0:
            coverage[entity_type] = (changed_count / total_count) * 100
        else:
            coverage[entity_type] = 0.0
    
    # File coverage
    total_files = len(all_code_files)
    changed_files_count = len(changed_files)
    if total_files > 0:
        coverage['files'] = (changed_files_count / total_files) * 100
    else:
        coverage['files'] = 0.0
    
    # Add summary statistics
    print("\nCoverage Summary:")
    print("-" * 50)
    
    # File coverage
    print(f"{'Files':10s}: {coverage['files']:6.2f}% ({changed_files_count}/{total_files})")
    print()
    
    # Entity coverage
    for entity_type in entity_types:
        percentage = coverage.get(entity_type, 0.0)
        total = len(total_by_type[entity_type])
        changed = len(changed_by_type[entity_type])
        print(f"{entity_type:10s}: {percentage:6.2f}% ({changed}/{total})")
    
    return coverage


def save_coverage_results(folder_path: str, coverage: Dict[str, float], output_dir: str = "logs") -> None:
    """
    Save coverage results to a JSON file with hierarchical structure.
    
    Args:
        folder_path: Path to the automated pipeline folder
        coverage: Coverage results dictionary
        output_dir: Directory to save the output file (default: "logs")
    """
    # Parse the folder path to extract keys
    folder_path = Path(folder_path)
    path_parts = folder_path.parts
    
    # Find where "logs/" starts and extract the relevant parts
    logs_index = -1
    for i, part in enumerate(path_parts):
        if part == "logs":
            logs_index = i
            break
    
    if logs_index >= 0 and logs_index + 2 < len(path_parts):
        # Extract pipeline name (first level key) and repo name (second level key)
        pipeline_name = path_parts[logs_index + 1]
        repo_name = path_parts[logs_index + 2]
    else:
        # Fallback: use the last two parts of the path
        if len(path_parts) >= 2:
            pipeline_name = path_parts[-2]
            repo_name = path_parts[-1]
        else:
            raise ValueError(f"Cannot parse folder path: {folder_path}")
    
    # Create output directory if it doesn't exist
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Output file path
    output_file = output_dir / "entity_coverage_results.json"
    
    # Load existing data if file exists
    existing_data = {}
    if output_file.exists():
        try:
            with open(output_file, 'r') as f:
                existing_data = json.load(f)
            print(f"Loaded existing coverage data from {output_file}")
        except json.JSONDecodeError:
            print(f"Warning: Could not parse existing file {output_file}, starting fresh")
            existing_data = {}
    
    # Update data with new coverage results
    if pipeline_name not in existing_data:
        existing_data[pipeline_name] = {}
    
    existing_data[pipeline_name][repo_name] = coverage
    
    # Save updated data
    with open(output_file, 'w') as f:
        json.dump(existing_data, f, indent=2)
    
    print(f"\nCoverage results saved to: {output_file}")
    print(f"Pipeline: {pipeline_name}")
    print(f"Repository: {repo_name}")


def main():
    """
    Main function to analyze entity coverage from command line.
    """
    parser = argparse.ArgumentParser(description='Analyze entity coverage for task instances')
    parser.add_argument('folder_path', type=str, 
                        help='Path to the automated pipeline folder (e.g., logs/automated_pipeline_o3_bugs100_combos200_depth2_workers32_nbugs1_patches4_perfile2_permodule10/astropy__astropy.26d14786)')
    parser.add_argument('--repo-name', type=str, default=None,
                        help='Repository name (e.g., astropy__astropy). If not provided, extracted from folder path')
    parser.add_argument('--entity-types', nargs='+', default=['class', 'func', 'object'],
                        help='Entity types to analyze (default: class func object)')
    parser.add_argument('--output-dir', type=str, default='logs',
                        help='Directory to save output results (default: logs)')
    
    args = parser.parse_args()
    
    try:
        # Load instances
        instances = load_task_instances(args.folder_path)
        print(f"Loaded {len(instances)} instances from {args.folder_path}")
        
        # Extract repo name if not provided
        if args.repo_name:
            repo_name = args.repo_name
        else:
            # Extract from folder path - get the last part of the path
            repo_name = Path(args.folder_path).name
        
        print(f"Using repository: {repo_name}")
        
        # Analyze coverage
        coverage = analyze_entity_coverage(repo_name, instances, args.entity_types)
        
        # Save results
        save_coverage_results(args.folder_path, coverage, args.output_dir)
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
