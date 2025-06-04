#!/usr/bin/env python3
"""
Script to inspect the columns/fields in SWE-bench and SWE-smith datasets.
"""

import json
from datasets import load_dataset
from pathlib import Path

def inspect_swe_bench():
    """Inspect SWE-bench dataset columns"""
    print("=" * 50)
    print("SWE-bench Dataset Columns")
    print("=" * 50)
    
    try:
        # Load SWE-bench dataset
        swe_bench = load_dataset("SWE-bench/SWE-bench", split="test")
        
        print(f"Total instances: {len(swe_bench)}")
        print(f"Features: {swe_bench.features}")
        print("\nColumn names:")
        for col in swe_bench.column_names:
            print(f"  - {col}")
        
        # Show sample instance
        if len(swe_bench) > 0:
            sample = swe_bench[0]
            print("\nSample instance keys:")
            for key in sample.keys():
                value = sample[key]
                if isinstance(value, str) and len(value) > 100:
                    print(f"  - {key}: {type(value).__name__} (length: {len(value)})")
                else:
                    print(f"  - {key}: {type(value).__name__} = {value}")
                    
    except Exception as e:
        print(f"Error loading SWE-bench: {e}")

def inspect_swe_smith():
    """Inspect SWE-smith dataset columns"""
    print("\n" + "=" * 50)
    print("SWE-smith Dataset Columns")
    print("=" * 50)
    
    try:
        # Load SWE-smith dataset
        swe_smith = load_dataset("SWE-bench/SWE-smith", split="train")
        
        print(f"Total instances: {len(swe_smith)}")
        print(f"Features: {swe_smith.features}")
        print("\nColumn names:")
        for col in swe_smith.column_names:
            print(f"  - {col}")
            
        # Show sample instance
        if len(swe_smith) > 0:
            sample = swe_smith[0]
            print("\nSample instance keys:")
            for key in sample.keys():
                value = sample[key]
                if isinstance(value, str) and len(value) > 100:
                    print(f"  - {key}: {type(value).__name__} (length: {len(value)})")
                elif isinstance(value, list):
                    print(f"  - {key}: {type(value).__name__} (length: {len(value)})")
                else:
                    print(f"  - {key}: {type(value).__name__} = {value}")
                    
    except Exception as e:
        print(f"Error loading SWE-smith: {e}")

def inspect_r2e_gym():
    """Inspect R2E-Gym SFT Trajectories dataset columns"""
    print("\n" + "=" * 50)
    print("R2E-Gym SFT Trajectories Dataset Columns")
    print("=" * 50)
    
    try:
        # Load R2E-Gym dataset
        r2e_gym = load_dataset("R2E-Gym/R2E-Gym-Subset", split="train")
        
        print(f"Total instances: {len(r2e_gym)}")
        print(f"Features: {r2e_gym.features}")
        print("\nColumn names:")
        for col in r2e_gym.column_names:
            print(f"  - {col}")
            
        # Show sample instance
        if len(r2e_gym) > 0:
            sample = r2e_gym[0]
            print("\nSample instance keys:")
            for key in sample.keys():
                value = sample[key]
                if isinstance(value, str) and len(value) > 100:
                    print(f"  - {key}: {type(value).__name__} (length: {len(value)})")
                elif isinstance(value, list):
                    print(f"  - {key}: {type(value).__name__} (length: {len(value)})")
                else:
                    print(f"  - {key}: {type(value).__name__} = {value}")
                    
    except Exception as e:
        print(f"Error loading R2E-Gym: {e}")

def inspect_swe_gym():
    """Inspect SWE-Gym OpenHands SFT Trajectories dataset columns"""
    print("\n" + "=" * 50)
    print("SWE-Gym OpenHands SFT Trajectories Dataset Columns")
    print("=" * 50)
    
    try:
        # Load SWE-Gym dataset
        swe_gym = load_dataset("SWE-Gym/SWE-Gym", split="train")
        
        print(f"Total instances: {len(swe_gym)}")
        print(f"Features: {swe_gym.features}")
        print("\nColumn names:")
        for col in swe_gym.column_names:
            print(f"  - {col}")
            
        # Show sample instance
        if len(swe_gym) > 0:
            sample = swe_gym[0]
            print("\nSample instance keys:")
            for key in sample.keys():
                value = sample[key]
                if isinstance(value, str) and len(value) > 100:
                    print(f"  - {key}: {type(value).__name__} (length: {len(value)})")
                elif isinstance(value, list):
                    print(f"  - {key}: {type(value).__name__} (length: {len(value)})")
                else:
                    print(f"  - {key}: {type(value).__name__} = {value}")
                    
    except Exception as e:
        print(f"Error loading SWE-Gym: {e}")

def inspect_local_dataset(dataset_path: str):
    """Inspect local dataset file"""
    print(f"\n" + "=" * 50)
    print(f"Local Dataset: {dataset_path}")
    print("=" * 50)
    
    try:
        dataset_path = Path(dataset_path)
        if not dataset_path.exists():
            print(f"File not found: {dataset_path}")
            return
            
        if dataset_path.suffix == ".json":
            with open(dataset_path, 'r') as f:
                data = json.load(f)
        elif dataset_path.suffix == ".jsonl":
            data = []
            with open(dataset_path, 'r') as f:
                for line in f:
                    data.append(json.loads(line.strip()))
        else:
            print(f"Unsupported file format: {dataset_path.suffix}")
            return
            
        print(f"Total instances: {len(data)}")
        
        if len(data) > 0:
            sample = data[0]
            print("\nSample instance keys:")
            for key in sample.keys():
                value = sample[key]
                if isinstance(value, str) and len(value) > 100:
                    print(f"  - {key}: {type(value).__name__} (length: {len(value)})")
                elif isinstance(value, list):
                    print(f"  - {key}: {type(value).__name__} (length: {len(value)})")
                else:
                    print(f"  - {key}: {type(value).__name__} = {value}")
                    
    except Exception as e:
        print(f"Error loading local dataset: {e}")

def compare_datasets():
    """Compare the schemas of different datasets"""
    print("\n" + "=" * 50)
    print("Dataset Schema Comparison")
    print("=" * 50)
    
    try:
        # Load all datasets
        swe_bench = load_dataset("SWE-bench/SWE-bench", split="test")
        swe_smith = load_dataset("SWE-bench/SWE-smith", split="train")
        r2e_gym = load_dataset("R2E-Gym/R2E-Gym-Subset", split="train")
        swe_gym = load_dataset("SWE-Gym/SWE-Gym", split="train")
        
        swe_bench_cols = set(swe_bench.column_names)
        swe_smith_cols = set(swe_smith.column_names)
        r2e_gym_cols = set(r2e_gym.column_names)
        swe_gym_cols = set(swe_gym.column_names)
        
        all_cols = swe_bench_cols | swe_smith_cols | r2e_gym_cols | swe_gym_cols
        
        print("Dataset column comparison:")
        print(f"{'Column':<30} {'SWE-bench':<12} {'SWE-smith':<12} {'R2E-Gym':<12} {'SWE-Gym':<12}")
        print("-" * 80)
        
        for col in sorted(all_cols):
            bench_mark = "✓" if col in swe_bench_cols else "✗"
            smith_mark = "✓" if col in swe_smith_cols else "✗"
            r2e_mark = "✓" if col in r2e_gym_cols else "✗"
            swe_gym_mark = "✓" if col in swe_gym_cols else "✗"
            print(f"{col:<30} {bench_mark:<12} {smith_mark:<12} {r2e_mark:<12} {swe_gym_mark:<12}")
            
    except Exception as e:
        print(f"Error comparing datasets: {e}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Inspect SWE-bench and SWE-smith dataset columns")
    parser.add_argument("--local", type=str, help="Path to local dataset file (.json or .jsonl)")
    parser.add_argument("--swe-bench", action="store_true", help="Inspect SWE-bench dataset")
    parser.add_argument("--swe-smith", action="store_true", help="Inspect SWE-smith dataset")
    parser.add_argument("--r2e-gym", action="store_true", help="Inspect R2E-Gym SFT Trajectories dataset")
    parser.add_argument("--swe-gym", action="store_true", help="Inspect SWE-Gym OpenHands SFT Trajectories dataset")
    parser.add_argument("--compare", action="store_true", help="Compare schemas between datasets")
    parser.add_argument("--all", action="store_true", help="Run all inspections")
    
    args = parser.parse_args()
    
    if args.all or args.swe_bench:
        inspect_swe_bench()
        
    if args.all or args.swe_smith:
        inspect_swe_smith()
        
    if args.all or args.r2e_gym:
        inspect_r2e_gym()
        
    if args.all or args.swe_gym:
        inspect_swe_gym()
        
    if args.local:
        inspect_local_dataset(args.local)
        
    if args.all or args.compare:
        compare_datasets()
        
    # If no specific arguments, run all
    if not any([args.local, args.swe_bench, args.swe_smith, args.r2e_gym, args.swe_gym, args.compare, args.all]):
        inspect_swe_bench()
        inspect_swe_smith()
        inspect_r2e_gym()
        inspect_swe_gym()
        compare_datasets()