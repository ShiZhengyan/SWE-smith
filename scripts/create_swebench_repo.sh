# Try to install
python -m swesmith.build_repo.try_install django/django configs/install_repo.sh --commit f9de1972685ab5ab31ce483a297d85d3e119088b

# Download all existing SWE-smith environments at swe-bench repo
python zshi_tools/download_images.py \
   --dataset_name ZhengyanShi/SWE-bench_Verified_Temporal_9 \
   --output_report image_download_report.json \
   --max_workers 32

# Create dataset
python zshi_tools/create_temporal_endpoints_dataset.py princeton-nlp/SWE-bench_Verified

# Create mirror repos and push it the remote
python swesmith/build_repo/build_dataset_images.py

# Set parameterized variables
repo="astropy__astropy.26d14786"
n_workers=32
max_bugs=20
max_combos=100
depth=2
run_id=$repo
model="gpt-4o"
types=("class", "func", "object")

# Try to generate bugs
python swesmith/bug_gen/procedural/generate.py $repo \
   --type class \
   --max_bugs $max_bugs

# LM Rewrite
python -m swesmith.bug_gen.llm.rewrite $repo \
   --model $model \
   --type object \
   --config_file configs/bug_gen/lm_rewrite.yml \
   --n_workers $n_workers

# LM Modify
python -m swesmith.bug_gen.llm.modify $repo \
   --n_bugs 1 \
   --model gpt-4o \
   --type object \
   --config_file configs/bug_gen/lm_modify.yml \
   --n_workers $n_workers

# Combine (Same File) - Must have validated task instances to run this script
# num_patches: number of patches to be combined
# limit_per_module affect the coverage of the bug generation
python swesmith/bug_gen/combine/same_file.py logs/bug_gen/$repo \
   --num_patches 2 \
   --limit_per_file 20 \
   --max_combos $max_combos

# Combine (Same Module) - Must have validated task instances to run this script
python swesmith/bug_gen/combine/same_module.py logs/bug_gen/$repo \
   --num_patches 2 \
   --limit_per_module 20 \
   --max_combos 200 \
   --depth $depth

# Collect patches
python -m swesmith.bug_gen.collect_patches logs/bug_gen/$repo

# Validate collected patches
python swesmith/harness/valid.py logs/bug_gen/${repo}_all_patches.json \
   --run_id $run_id \
   --max_workers $n_workers

# Collect task instances with 1+ F2P and push to the remote
python -m swesmith.harness.gather logs/run_validation/$run_id

# Run evaluation: This could be skipped as we always use the swe-bench images 
# we have already checked in `swesmith/harness/valid.py`
python -m swesmith.harness.eval \
    --dataset_path logs/task_insts/$repo.json \
    --predictions_path gold \
    --run_id $run_id

python -m swesmith.issue_gen.generate logs/task_insts/$repo.json \
    --config_file configs/issue_gen/ig_v2.yaml \
    --model $model \
    --n_workers 4 \
    --experiment_id $repo \
    --use_existing

# Merge problem statements
python swesmith/harness/merge_problem_statements.py \
   --repo $repo \
   --output logs/task_insts/${repo}_${model}_ps.json