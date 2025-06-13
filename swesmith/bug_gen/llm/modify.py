"""
Purpose: Given a repository, generate bug patches for functions/classes/objects in the repository.

Usage: python -m swesmith.bug_gen.llm.modify \
    --n_bugs <n_bugs> \
    --config_file <config_file> \
    --model <model> \
    --type <entity_type>
    repo  # e.g., tkrajina__gpxpy.09fc46b3

Where model follows the Azure OpenAI format.

Example:

python -m swesmith.bug_gen.llm.modify tkrajina__gpxpy.09fc46b3 --config_file configs/bug_gen/class_basic.yml --model gpt-4o --n_bugs 1
"""

import argparse
import shutil
import jinja2
import json
import logging
import os
import random
import yaml
import re

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
from dotenv import load_dotenv
from openai import AzureOpenAI
from azure.identity import DefaultAzureCredential, ChainedTokenCredential, AzureCliCredential, get_bearer_token_provider
from swesmith.bug_gen.criteria import MAP_KEY_TO_CRITERIA
from swesmith.bug_gen.llm.utils import PROMPT_KEYS, extract_code_block
from swesmith.bug_gen.utils import (
    ENTITY_TYPES,
    BugRewrite,
    CodeEntity,
    apply_code_change,
    extract_entities_from_directory,
    get_patch,
)
from swesmith.constants import (
    LOG_DIR_BUG_GEN,
    ORG_NAME,
    PREFIX_BUG,
    PREFIX_METADATA,
)
from swesmith.bug_gen.llm.rewrite import make_worktree, remove_worktree
from swesmith.utils import clone_repo, does_repo_exist
from tqdm.auto import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm
from typing import Any


load_dotenv(dotenv_path=os.getenv("SWEFT_DOTENV_PATH"))

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
    else:
        raise ValueError(f"Unsupported model: {model}")
    
    deployment_name = re.sub(r'[^a-zA-Z0-9-_]', '', f'{model_name}_{model_version}')
    endpoint = f'https://trapi.research.microsoft.com/{instance}'
    
    client = AzureOpenAI(
        azure_endpoint=endpoint,
        azure_ad_token_provider=credential,
        api_version=api_version,
    )
    
    return client, deployment_name


def gen_bug_from_code_lm(
    candidate: CodeEntity, configs: dict, n_bugs: int, model: str
) -> list[BugRewrite]:
    """
    Given the source code of a function, return `n` bugs with an LM
    """

    def format_prompt(prompt: str | None, config: dict, candidate: CodeEntity) -> str:
        if not prompt:
            return ""
        env = jinja2.Environment()

        def jinja_shuffle(seq):
            result = list(seq)
            random.shuffle(result)
            return result

        env.filters["shuffle"] = jinja_shuffle
        template = env.from_string(prompt)
        return template.render(**asdict(candidate), **config.get("parameters", {}))

    def get_role(key: str) -> str:
        if key == "system":
            return "system"
        return "user"

    bugs = []
    messages = [
        {"content": format_prompt(configs[k], configs, candidate), "role": get_role(k)}
        for k in PROMPT_KEYS
    ]
    # Remove empty messages
    messages = [x for x in messages if x["content"]]
    
    client, deployment_name = get_azure_client(model)
    
    # Generate multiple completions
    for _ in range(n_bugs):
        # Conditionally include temperature based on model compatibility
        call_kwargs = {
            "model": deployment_name,
            "messages": messages,
        }
        if "o3" not in model.lower():
            call_kwargs["temperature"] = 1
            
        response = client.chat.completions.create(**call_kwargs)
        
        choice = response.choices[0]
        message = choice.message
        explanation = (
            message.content.split("Explanation:")[-1].strip()
            if "Explanation" in message.content
            else message.content.split("```")[-1].strip()
        )
        
        # Estimate cost (simplified approach since Azure OpenAI doesn't provide direct cost info)
        estimated_cost = 0.01  # Placeholder cost estimation
        
        bugs.append(
            BugRewrite(
                rewrite=extract_code_block(message.content),
                explanation=explanation,
                cost=estimated_cost,
                output=message.content,
                strategy="llm",
            )
        )
    
    return bugs


def main(
    config_file: str,
    entity_type: str,
    model: str,
    n_bugs: int,
    repo: str,
    *,
    n_workers: int = 1,
    **kwargs,
):
    # Check arguments
    assert does_repo_exist(repo), f"Repository {repo} does not exist in {ORG_NAME}."
    assert os.path.exists(config_file), f"{config_file} not found"
    assert n_bugs > 0, "n_bugs must be greater than 0"
    configs = yaml.safe_load(open(config_file))
    assert all(key in configs for key in PROMPT_KEYS + ["criteria", "name"]), (
        f"Missing keys in {config_file}"
    )

    # Clone repository, identify valid candidates
    print("Cloning repository...")
    clone_repo(repo)
    print("Extracting candidates...")
    candidates = extract_entities_from_directory(repo, entity_type)
    print(f"{len(candidates)} candidates found for {entity_type} in {repo}")
    candidates = [x for x in candidates if MAP_KEY_TO_CRITERIA[configs["criteria"]](x)]
    print(f"{len(candidates)} candidates passed criteria")
    if not candidates:
        print(f"No candidates found for {entity_type} in {repo}.")
        return

    print(f"Generating bugs for {entity_type} in {repo} using {model}...")
    if not kwargs.get("yes", False):
        if input("Proceed with bug generation? (y/n): ").lower() != "y":
            return

    # Set up logging
    log_dir = LOG_DIR_BUG_GEN / repo
    log_dir.mkdir(parents=True, exist_ok=True)
    print(f"Logging bugs to {log_dir}")

    def _process_candidate(candidate: CodeEntity):
        # 1. Isolate this thread inside its own work-tree
        wt = make_worktree(repo)
        try:
            # Remember original path for logging
            orig_file_path = candidate.file_path

            # Point the candidate at the work-tree copy
            candidate.file_path = os.path.join(
                wt, os.path.relpath(candidate.file_path, repo)
            )

            # -------------------- unchanged logic ---------------------
            bugs = gen_bug_from_code_lm(candidate, configs, n_bugs, model)
            cost, n_bugs_generated, n_generation_failed = sum(b.cost for b in bugs), 0, 0

            for bug in bugs:
                bug_dir = (
                    log_dir / orig_file_path.replace("/", "__") / candidate.src_node.name
                )
                bug_dir.mkdir(parents=True, exist_ok=True)
                uuid_str = f"{configs['name']}__{bug.get_hash()}"
                metadata_path = bug_dir / f"{PREFIX_METADATA}__{uuid_str}.json"
                bug_path      = bug_dir / f"{PREFIX_BUG}__{uuid_str}.diff"

                try:
                    with open(metadata_path, "w") as f:
                        json.dump(bug.to_dict(), f, indent=2)

                    apply_code_change(candidate, bug)          # acts in *wt*
                    patch = get_patch(wt, reset_changes=True)  # --------^
                    if not patch:
                        raise ValueError("Patch is empty.")

                    with open(bug_path, "w") as f:
                        f.write(patch)
                except Exception as e:
                    print(
                        f"Error applying bug to {candidate.src_node.name} "
                        f"in {candidate.file_path}: {e}"
                    )
                    metadata_path.unlink(missing_ok=True)
                    n_generation_failed += 1
                else:
                    n_bugs_generated += 1

            return {
                "cost": cost,
                "n_bugs_generated": n_bugs_generated,
                "n_generation_failed": n_generation_failed,
            }
        # 2. Always clean up this thread’s work-tree
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
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "repo",
        type=str,
        help="Name of a SWE-smith repository to generate bugs for.",
    )
    parser.add_argument(
        "--type",
        dest="entity_type",
        type=str,
        choices=list(ENTITY_TYPES.keys()),
        default="func",
        help="Type of entity to generate bugs for.",
    )
    parser.add_argument(
        "--model",
        type=str,
        help="Model to use for bug generation (gpt-4o, o3, o3-mini, o4-mini)",
        default="gpt-4o",
    )
    parser.add_argument(
        "--n_bugs",
        type=int,
        help="Number of bugs to generate per entity",
        default=1,
    )
    parser.add_argument(
        "--config_file",
        type=str,
        help="Configuration file containing bug gen. strategy prompts",
        required=True,
    )
    parser.add_argument("-y", "--yes", action="store_true", help="Skip confirmation")
    parser.add_argument(
        "--n_workers", type=int, help="Number of workers to use", default=1
    )
    args = parser.parse_args()
    main(**vars(args))
