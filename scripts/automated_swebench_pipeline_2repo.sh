#!/bin/bash

# Set strict error handling
set -e
set -o pipefail

# Configuration
MAX_BUGS=30
MAX_COMBOS=50
DEPTH=2
MODEL="o1"
# MODEL="gpt-4.1"
N_WORKERS=32
N_BUGS=1
NUM_PATCHES=2
LIMIT_PER_FILE=2
LIMIT_PER_MODULE=10
TIMEOUT=1200

# Repository list
REPOS=(
    # "astropy__astropy.26d14786"
    # "pylint-dev__pylint.1f8c4d9e"
    "sympy__sympy.a36caf5c"
    "pytest-dev__pytest.3c153494"
    "sympy__sympy.360290c4"
    "sphinx-doc__sphinx.6cb783c0"
    "sphinx-doc__sphinx.9bb204dc"
    "pylint-dev__pylint.99589b08"
    "scikit-learn__scikit-learn.586f4318"
    "scikit-learn__scikit-learn.3eacf948"
    # "pytest-dev__pytest.58e6a09d"
    # "pydata__xarray.41fef6f1"
    # "pydata__xarray.7c4e2ac8"
    # "matplotlib__matplotlib.3dd06a46"
    # "matplotlib__matplotlib.a3e2897b"
    # "django__django.4a72da71"
    # "django__django.f8fab6f9"
    # "astropy__astropy.b16c7d12"
)

# Bug generation types
TYPES=("class" "func" "object")

generate_bugs_for_type() {
    local repo=$1
    local type=$2
    
    log "Processing bug generation for $repo with type $type"
    
    {
        log "Starting procedural bug generation for $repo ($type)"
        python swesmith/bug_gen/procedural/generate.py "$repo" \
            --type "$type" \
            --max_bugs $MAX_BUGS 2>&1
        rm -rf $repo

        log "Starting LM rewrite for $repo ($type)"
        python -m swesmith.bug_gen.llm.rewrite "$repo" \
            --model "$MODEL" \
            --type "$type" \
            --config_file configs/bug_gen/lm_rewrite.yml \
            --n_workers $N_WORKERS 2>&1
        rm -rf $repo

        log "Starting LM modify for $repo ($type)"
        python -m swesmith.bug_gen.llm.modify "$repo" \
            --n_bugs $N_BUGS \
            --model "$MODEL" \
            --type "$type" \
            --yes \
            --config_file configs/bug_gen/lm_modify.yml \
            --n_workers $N_WORKERS 2>&1
        rm -rf $repo

        log "Completed bug generation for $repo ($type)"
        
    } 2>&1 | tee -a "$MAIN_LOG"
}

merge_bugs_as_patches() {
    local repo=$1
    
    log "Starting patch merging for $repo"
    
    {
        log "Combining patches (same file) for $repo"
        python swesmith/bug_gen/combine/same_file.py "$LOG_DIR/bug_gen/$repo" \
            --include_invalid_patches \
            --num_patches $NUM_PATCHES \
            --limit_per_file $LIMIT_PER_FILE \
            --max_combos $MAX_COMBOS 2>&1
        rm -rf $repo

        log "Combining patches (same module) for $repo"
        python swesmith/bug_gen/combine/same_module.py "$LOG_DIR/bug_gen/$repo" \
            --include_invalid_patches \
            --num_patches $NUM_PATCHES \
            --limit_per_module $LIMIT_PER_MODULE \
            --max_combos $MAX_COMBOS \
            --depth $DEPTH 2>&1
        rm -rf $repo

        log "Collecting patches for $repo"
        python -m swesmith.bug_gen.collect_patches "$LOG_DIR/bug_gen/$repo" 2>&1
        
        log "Completed patch merging for $repo"
        
    } 2>&1 | tee -a "$MAIN_LOG"
}

process_repo_post_generation() {
    local repo=$1
    local run_id="${repo}"
    
    log "Starting post-generation processing for $repo"
    
    {
        log "Validating patches for $repo"
        python swesmith/harness/valid.py "$LOG_DIR/bug_gen/${repo}_all_patches.json" \
            --timeout $TIMEOUT \
            --run_id "$run_id" \
            --max_workers $N_WORKERS 2>&1
        rm -rf $repo

        log "Gathering task instances for $repo" 
        python -m swesmith.harness.gather "$LOG_DIR/run_validation/$run_id" 2>&1
        
        log "Generating issues for $repo"
        python -m swesmith.issue_gen.generate "$LOG_DIR/task_insts/$run_id.json" \
            --config_file configs/issue_gen/ig_v2.yaml \
            --model "$MODEL" \
            --n_workers $N_WORKERS \
            --experiment_id "$repo" \
            --use_existing 2>&1
        
        log "Merging problem statements for $repo"
        python swesmith/harness/merge_problem_statements.py \
            --repo "${repo}" \
            --output "$LOG_DIR/task_insts/${repo}_ps.json" 2>&1
        
        log "Completed post-generation processing for $repo"
        
    } 2>&1 | tee -a "$MAIN_LOG"
}

# Main execution
main() {
    local total_repos=${#REPOS[@]}
    local current_repo=0
    
    # Process each repository completely before moving to the next
    for repo in "${REPOS[@]}"; do
        # Logging setup
        export SWESMITH_LOG_DIR="logs/automated_pipeline_${MODEL}_bugs${MAX_BUGS}_combos${MAX_COMBOS}_depth${DEPTH}_workers${N_WORKERS}_nbugs${N_BUGS}_patches${NUM_PATCHES}_perfile${LIMIT_PER_FILE}_permodule${LIMIT_PER_MODULE}/$repo"
        LOG_DIR="$SWESMITH_LOG_DIR"
        mkdir -p "$LOG_DIR"
        MAIN_LOG="$LOG_DIR/pipeline_$(date +%Y%m%d_%H%M%S).log"

        log() {
            echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$MAIN_LOG"
        }

        current_repo=$((current_repo + 1))
        log "Repository progress: $current_repo/$total_repos - Processing $repo"
        
        # Check if patches already exist
        if [[ -f "$LOG_DIR/bug_gen/${repo}_all_patches.json" ]]; then
            log "Patches file already exists for $repo, skipping bug generation and merging"
        else
            # Phase 1: Bug generation for all types in this repo
            log "Phase 1: Bug generation for $repo"
            local type_count=0
            for type in "${TYPES[@]}"; do
                type_count=$((type_count + 1))
                log "Bug generation progress for $repo: $type_count/${#TYPES[@]} - Processing type $type"
                
                if generate_bugs_for_type "$repo" "$type"; then
                    log "Successfully completed bug generation for $repo ($type)"
                else
                    log "ERROR: Failed bug generation for $repo ($type)"
                    continue
                fi
            done
            
            # Phase 1.5: Merge bugs as patches
            log "Phase 1.5: Merging patches for $repo"
            if merge_bugs_as_patches "$repo"; then
                log "Successfully completed patch merging for $repo"
            else
                log "ERROR: Failed patch merging for $repo"
                continue
            fi
        fi
        
        # Phase 2: Post-generation processing for this repo
        log "Phase 2: Post-generation processing for $repo"
        if process_repo_post_generation "$repo"; then
            log "Successfully completed processing for $repo"
        else
            log "ERROR: Failed processing for $repo"
            continue
        fi
        
        log "Completed all processing for $repo"
    done
    
    log "Automated pipeline completed!"
    log "Check individual logs in $LOG_DIR for details"
}

# Run the main function
main "$@"
