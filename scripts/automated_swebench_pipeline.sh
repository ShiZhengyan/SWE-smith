#!/bin/bash

# Set strict error handling
set -e
set -o pipefail

# Configuration
MAX_BUGS=20
MAX_COMBOS=100
DEPTH=2
MODEL="gpt-4o"
N_WORKERS=32

# Repository list
REPOS=(
    "astropy__astropy.26d14786"
    "pylint-dev__pylint.1f8c4d9e"
    "sympy__sympy.a36caf5c"
    "pytest-dev__pytest.3c153494"
    "sympy__sympy.360290c4"
    "sphinx-doc__sphinx.6cb783c0"
    "sphinx-doc__sphinx.9bb204dc"
    "pylint-dev__pylint.99589b08"
    "scikit-learn__scikit-learn.586f4318"
    "scikit-learn__scikit-learn.3eacf948"
    "pytest-dev__pytest.58e6a09d"
    "pydata__xarray.41fef6f1"
    "pydata__xarray.7c4e2ac8"
    "matplotlib__matplotlib.3dd06a46"
    "matplotlib__matplotlib.a3e2897b"
    "django__django.4a72da71"
    "django__django.f8fab6f9"
    "astropy__astropy.b16c7d12"
)

# Bug generation types
TYPES=("class" "func" "object")

# Logging setup
LOG_DIR="logs/automated_pipeline"
mkdir -p "$LOG_DIR"
MAIN_LOG="$LOG_DIR/pipeline_$(date +%Y%m%d_%H%M%S).log"

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$MAIN_LOG"
}

process_repo_type() {
    local repo=$1
    local type=$2
    local run_id="${repo}_${type}"
    
    log "Processing $repo with type $type"
    
    # Create repo-specific log
    local repo_log="$LOG_DIR/${repo}_${type}.log"
    
    {
        log "Starting procedural bug generation for $repo ($type)"
        python swesmith/bug_gen/procedural/generate.py "$repo" \
            --type "$type" \
            --max_bugs $MAX_BUGS
        
        log "Starting LM rewrite for $repo ($type)"
        python -m swesmith.bug_gen.llm.rewrite "$repo" \
            --model "$MODEL" \
            --type "$type" \
            --config_file configs/bug_gen/lm_rewrite.yml \
            --n_workers $N_WORKERS
        
        log "Starting LM modify for $repo ($type)"
        python -m swesmith.bug_gen.llm.modify "$repo" \
            --n_bugs 1 \
            --model "$MODEL" \
            --type "$type" \
            --config_file configs/bug_gen/lm_modify.yml \
            --n_workers $N_WORKERS
        
        log "Combining patches (same file) for $repo ($type)"
        python swesmith/bug_gen/combine/same_file.py "logs/bug_gen/$repo" \
            --num_patches 2 \
            --limit_per_file 20 \
            --max_combos $MAX_COMBOS
        
        log "Combining patches (same module) for $repo ($type)"
        python swesmith/bug_gen/combine/same_module.py "logs/bug_gen/$repo" \
            --num_patches 2 \
            --limit_per_module 20 \
            --max_combos 200 \
            --depth $DEPTH
        
        log "Collecting patches for $repo ($type)"
        python -m swesmith.bug_gen.collect_patches "logs/bug_gen/$repo"
        
        log "Validating patches for $repo ($type)"
        python swesmith/harness/valid.py "logs/bug_gen/${repo}_all_patches.json" \
            --run_id "$run_id" \
            --max_workers $N_WORKERS
        
        log "Gathering task instances for $repo ($type)"
        python -m swesmith.harness.gather "logs/run_validation/$run_id"
        
        log "Running evaluation for $repo ($type)"
        python -m swesmith.harness.eval \
            --dataset_path "logs/task_insts/$repo.json" \
            --predictions_path gold \
            --run_id "$run_id"
        
        log "Generating issues for $repo ($type)"
        python -m swesmith.issue_gen.generate "logs/task_insts/$repo.json" \
            --config_file configs/issue_gen/ig_v2.yaml \
            --model "$MODEL" \
            --n_workers 4 \
            --experiment_id "$repo" \
            --use_existing
        
        log "Merging problem statements for $repo ($type)"
        python swesmith/harness/merge_problem_statements.py \
            --repo "$repo" \
            --output "logs/task_insts/${repo}_${MODEL}_ps.json"
        
        log "Completed processing $repo ($type)"
        
    } 2>&1 | tee "$repo_log"
}

# Main execution
main() {
    log "Starting automated SWE-bench pipeline"
    log "Processing ${#REPOS[@]} repositories with ${#TYPES[@]} types each"
    log "Total combinations: $((${#REPOS[@]} * ${#TYPES[@]}))"
    
    local total_combinations=$((${#REPOS[@]} * ${#TYPES[@]}))
    local current_combination=0
    
    for repo in "${REPOS[@]}"; do
        for type in "${TYPES[@]}"; do
            current_combination=$((current_combination + 1))
            log "Progress: $current_combination/$total_combinations - Processing $repo with type $type"
            
            if process_repo_type "$repo" "$type"; then
                log "Successfully completed $repo ($type)"
            else
                log "ERROR: Failed to process $repo ($type)"
                # Continue with next combination instead of exiting
                continue
            fi
        done
    done
    
    log "Automated pipeline completed!"
    log "Check individual logs in $LOG_DIR for details"
}

# Run the main function
main "$@"
