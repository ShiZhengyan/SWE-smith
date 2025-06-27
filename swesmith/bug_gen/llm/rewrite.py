"""
Purpose: Given a repository, blank out various functions/classes, then ask the model to rewrite them.

Usage: python -m swesmith.bug_gen.llm.rewrite \
    --model <model> \
    --type <entity_type> \
    repo  # e.g., tkrajina__gpxpy.09fc46b3

Where model follows the Azure OpenAI format.

Example:

python -m swesmith.bug_gen.llm.rewrite tkrajina__gpxpy.09fc46b3 --model gpt-4o --type class
"""

import ast
import argparse
import json
import logging
import os
import random
import shutil
import subprocess
import yaml
import re
import tempfile, shutil, subprocess, pathlib, threading

from concurrent.futures import ThreadPoolExecutor, as_completed
from openai import AzureOpenAI
from azure.identity import DefaultAzureCredential, ChainedTokenCredential, AzureCliCredential, get_bearer_token_provider
from swesmith.bug_gen.criteria import filter_min_simple_complexity
from swesmith.bug_gen.llm.utils import (
    PROMPT_KEYS,
    extract_code_block,
    strip_function_body,
)
from swesmith.bug_gen.utils import (
    ENTITY_TYPES,
    BugRewrite,
    CodeEntity,
    apply_code_change,
    extract_entities_from_directory,
    get_patch,
)
from swesmith.constants import LOG_DIR_BUG_GEN, PREFIX_BUG, PREFIX_METADATA
from swesmith.utils import clone_repo
from tqdm.auto import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm
from typing import Any


LM_REWRITE = "lm_rewrite"

# Azure OpenAI setup
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


def get_azure_client(model: str):
    """Get Azure OpenAI client for the specified model"""
    if model == "gpt-4o" or model == "4o":
        model_name = 'gpt-4o'
        model_version = '2024-05-13'
        instance = 'gcr/preview'
        api_version = '2024-10-21'
    elif model == "o3":
        model_name = 'o3'
        model_version = '2025-04-16'
        instance = 'msrne/shared'
        api_version = '2025-04-01-preview'
    elif model == "o3-mini":
        model_name = 'o3-mini'
        model_version = '2025-01-31'
        instance = 'msrne/shared'
        api_version = '2025-04-01-preview'
    elif model == "o4-mini":
        model_name = 'o4-mini'
        model_version = '2025-04-16'
        instance = 'msrne/shared'
        api_version = '2025-04-01-preview'
    elif model == "gpt-4.1":
        model_name = 'gpt-4.1'  # Ensure this is a valid model name
        model_version = '2025-04-14'  # Ensure this is a valid model version
        instance = 'gcr/shared' # See https://aka.ms/trapi/models for the instance name, remove /openai (library adds it implicitly) 
        api_version = '2025-04-01-preview'
    elif model == "gpt-4.5-preview":
        model_name = 'gpt-4.5-preview'  # Ensure this is a valid model name
        model_version = '2025-02-27'  # Ensure this is a valid model version
        instance = 'msrne/shared' # See https://aka.ms/trapi/models for the instance name, remove /openai (library adds it implicitly) 
        api_version = '2025-04-01-preview'
    elif model == "o1":
        model_name = 'o1'  # Ensure this is a valid model name
        model_version = '2024-12-17'  # Ensure this is a valid model version
        instance = 'msrne/shared' # See https://aka.ms/trapi/models for the instance name, remove /openai (library adds it implicitly) 
        api_version = '2025-04-01-preview'
    elif model == "gpt-4.1-mini":
        model_name = 'gpt-4.1-mini'
        model_version = '2025-04-14'  # Ensure this is a valid model version
        instance = 'msrne/shared'
        api_version = '2025-04-01-preview'
    else:
        raise ValueError(f"Unsupported model: {model}")
    
    deployment_name = re.sub(r"[^a-zA-Z0-9._-]", "", f"{model_name}_{model_version}")
    endpoint = f'https://trapi.research.microsoft.com/{instance}'
    
    client = AzureOpenAI(
        azure_endpoint=endpoint,
        azure_ad_token_provider=credential,
        api_version=api_version,
    )
    
    return client, deployment_name

random.seed(24)


def make_worktree(base_repo: str) -> str:
    """
    Create a temporary work-tree that points at HEAD of the main repo.
    Returns the path to the new working directory.
    """
    tmpdir = tempfile.mkdtemp(prefix=pathlib.Path(base_repo).name + "_wt_")
    # --detach keeps it independent from branch heads
    subprocess.run(
        ["git", "-C", base_repo, "worktree", "add", "--detach", tmpdir, "HEAD"],
        check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    return tmpdir


def remove_worktree(base_repo: str, wt_path: str):
    subprocess.run(
        ["git", "-C", base_repo, "worktree", "remove", "-f", wt_path],
        check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    shutil.rmtree(wt_path, ignore_errors=True)    


def get_function_signature(node):
    """Generate the function signature as a string."""
    args = [ast.unparse(arg) for arg in node.args.args]  # For Python 3.9+
    args_str = ", ".join(args)
    return f"def {node.name}({args_str})"


def get_entity_signature(node):
    """Generate the entity signature as a string (function, class, etc.)."""
    if isinstance(node, ast.FunctionDef):
        args = [ast.unparse(arg) for arg in node.args.args]  # For Python 3.9+
        args_str = ", ".join(args)
        return f"def {node.name}({args_str})"
    elif isinstance(node, ast.ClassDef):
        # For classes, return the class definition line
        bases = []
        if node.bases:
            bases.extend([ast.unparse(base) for base in node.bases])
        if node.keywords:
            bases.extend([f"{kw.arg}={ast.unparse(kw.value)}" for kw in node.keywords])
        
        if bases:
            bases_str = "(" + ", ".join(bases) + ")"
        else:
            bases_str = ""
        
        return f"class {node.name}{bases_str}"
    else:
        # Fallback for other node types
        return f"{type(node).__name__}: {node.name}"


def main(
    repo: str,
    config_file: str,
    model: str,
    entity_type: str,
    n_workers: int,
    redo_existing: bool = False,
    max_bugs: int = None,
    **kwargs,
):
    configs = yaml.safe_load(open(config_file))
    print(f"Cloning {repo}...")
    clone_repo(repo)
    print(f"Extracting entities from {repo}...")
    candidates = [
        x
        for x in extract_entities_from_directory(repo, entity_type)
        if filter_min_simple_complexity(x, 3)
    ]
    if max_bugs:
        random.shuffle(candidates)
        candidates = candidates[:max_bugs]

    # Set up logging
    log_dir = LOG_DIR_BUG_GEN / repo
    log_dir.mkdir(parents=True, exist_ok=True)
    print(f"Logging bugs to {log_dir}")
    if not redo_existing:
        print("Skipping existing bugs.")

    def _process_candidate(candidate: CodeEntity) -> dict[str, Any]:
        # 1.  Make an isolated work-tree for this thread
        wt = make_worktree(repo)               # <-- new
        try:
            # candidate.file_path originally points at the main repo.
            # Redirect it so `apply_code_change()` edits the work-tree copy.
            orig_file_path = candidate.file_path
            candidate.file_path = os.path.join(
                wt, os.path.relpath(candidate.file_path, repo)
            )

            # 2.  All logic below is unchanged except:
            #     * use *wt* wherever we used *repo* for Git commands
            #     * bug_dir still derived from the ORIGINAL path so the
            #       log structure on disk stays identical.
            bug_dir = (
                log_dir / orig_file_path.replace("/", "__") / candidate.src_node.name
            )
            if not redo_existing:
                if bug_dir.exists() and any(
                    str(x).startswith(f"{PREFIX_BUG}__{configs['name']}")
                    for x in os.listdir(bug_dir)
                ):
                    return {"n_bugs_generated": 0, "cost": 0.0}

            # -------- blank out the entity ---------------------------------
            try:
                blank_function = BugRewrite(
                    rewrite=strip_function_body(candidate.src_code),
                    explanation="Blanked out the function body.",
                    strategy=LM_REWRITE,
                )
                apply_code_change(candidate, blank_function)
            except Exception:
                return {"n_generation_failed": 1, "cost": 0.0}

            # -------- prepare the prompt -----------------------------------
            prompt_content = {
                "func_signature": get_entity_signature(candidate.src_node),
                "func_to_write": blank_function.rewrite,
                "file_src_code": open(candidate.file_path).read(),
            }
            messages = [
                {
                    "content": configs[k].format(**prompt_content),
                    "role": "user" if k != "system" else "system",
                }
                for k in PROMPT_KEYS
                if k in configs
            ]
            messages = [x for x in messages if x["content"]]

            client, deployment_name = get_azure_client(model)
            call_kwargs = {"model": deployment_name, "messages": messages}
            if not any(reasoning_model in model.lower() for reasoning_model in ["o1", "o3", "o4"]):
                call_kwargs["temperature"] = 0.5
            response = client.chat.completions.create(**call_kwargs)
            message = response.choices[0].message

            # -------- apply the model rewrite ------------------------------
            code_block   = extract_code_block(message.content)
            explanation  = message.content.split("```", 1)[0].strip()

            subprocess.run(
                ["git", "-C", wt, "reset", "--hard"],      # <-- use *wt*
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )

            rewrite = BugRewrite(
                rewrite=code_block,
                explanation=explanation,
                strategy=LM_REWRITE,
                cost=0.01,          # placeholder
                output=message.content,
            )
            apply_code_change(candidate, rewrite)

            # -------- create & log the patch -------------------------------
            patch = get_patch(wt, reset_changes=True)      # <-- use *wt*

            bug_dir.mkdir(parents=True, exist_ok=True)
            uuid_str      = f"{configs['name']}__{rewrite.get_hash()}"
            metadata_path = bug_dir / f"{PREFIX_METADATA}__{uuid_str}.json"
            bug_path      = bug_dir / f"{PREFIX_BUG}__{uuid_str}.diff"

            with open(metadata_path, "w") as f:
                json.dump(rewrite.to_dict(), f, indent=2)
            with open(bug_path, "w") as f:
                f.write(patch)
            print(f"Wrote bug to {bug_path}")

            return {"n_bugs_generated": 1, "cost": 0.01}

        # 3.  Always remove the temporary work-tree, even on exceptions
        finally:
            remove_worktree(repo, wt)


    stats = {"cost": 0.0, "n_bugs_generated": 0, "n_generation_failed": 0}
    with ThreadPoolExecutor(max_workers=n_workers) as executor:
        futures = [
            executor.submit(_process_candidate, candidate) for candidate in candidates
        ]

        with logging_redirect_tqdm():
            with tqdm(total=len(candidates), desc="Candidates") as pbar:
                for future in as_completed(futures):
                    cost = future.result()
                    for k, v in cost.items():
                        stats[k] += v
                    pbar.set_postfix(stats, refresh=True)
                    pbar.update(1)

    shutil.rmtree(repo)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate bug patches for functions/classes/objects in a repository."
    )
    parser.add_argument(
        "repo", type=str, help="Repository to generate bug patches for."
    )
    parser.add_argument(
        "--config_file", type=str, help="Path to the configuration file.", required=True
    )
    parser.add_argument(
        "--model", 
        type=str, 
        help="Model to use for rewriting (gpt-4o, o3, o3-mini, o4-mini)."
    )
    parser.add_argument(
        "--type",
        dest="entity_type",
        type=str,
        choices=list(ENTITY_TYPES.keys()),
        help="Type of entity to generate bug patches for.",
    )
    parser.add_argument(
        "--n_workers", type=int, help="Number of workers to use", default=1
    )
    parser.add_argument(
        "--redo_existing", action="store_true", help="Redo existing bugs."
    )
    parser.add_argument(
        "--max_bugs", type=int, help="Maximum number of bugs to generate."
    )
    args = parser.parse_args()
    main(**vars(args))
