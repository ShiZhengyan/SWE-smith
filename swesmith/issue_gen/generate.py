"""
Purpose: Given a bug patch, generate a GitHub-style issue that describes the bug.

python swesmith/issue_gen/generate.py \
    --dataset logs/experiments/*.json \
    --config configs/issue_gen/*.yaml \
    --model anthropic/claude-3-7-sonnet-20250219 \
    --n_workers 2
"""

import argparse
import jinja2
import json
import logging
import os
import random
import re
import shutil
import threading
import yaml

from azure.identity import DefaultAzureCredential, ChainedTokenCredential, AzureCliCredential, get_bearer_token_provider
from concurrent.futures import ThreadPoolExecutor, as_completed
from datasets import load_dataset
from dotenv import load_dotenv
from io import TextIOWrapper
from openai import AzureOpenAI
from pathlib import Path
from tqdm import tqdm
from litellm.utils import get_token_count
from tqdm.contrib.logging import logging_redirect_tqdm
from swebench.harness.constants import (
    FAIL_TO_PASS,
    KEY_INSTANCE_ID,
    LOG_TEST_OUTPUT,
)
from swesmith.constants import (
    LOG_DIR_ISSUE_GEN,
    LOG_DIR_RUN_VALIDATION,
    TEST_OUTPUT_END,
    TEST_OUTPUT_START,
)
from swesmith.issue_gen.utils import get_test_function

logging.getLogger("LiteLLM").setLevel(logging.WARNING)


TEST_SRC_CODE_PROMPT = r"""
**Test Source Code:**
Use the following test source code to help you write reasonable, effective reproduction code.

{test_src_code}
"""

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def maybe_shorten(text_str: str, max_tokens: int, model: str) -> str:
    """Shorten text if it exceeds the max_tokens limit.
    If shortening, return a string with the first and last max_tokens//2 tokens.
    """
    if get_token_count([{"content": text_str}], model) < max_tokens:
        return text_str
    return text_str[: max_tokens // 2] + "\n\n(...)\n\n" + text_str[-max_tokens // 2 :]


# Azure OpenAI setup
def setup_azure_client(model: str):
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

    # Extract model type from model string (e.g., "openai/gpt-4o" -> "4o")
    if "gpt-4o" in model.lower() or "4o" in model.lower():
        ep = "4o"
    elif "o3-mini" in model.lower():
        ep = "o3-mini"
    elif "o4-mini" in model.lower():
        ep = "o4-mini"
    elif "o3" in model.lower():
        ep = "o3"
    else:
        # Default to 4o if model not recognized
        ep = "4o"

    if ep == "4o":
        model_name = 'gpt-4o'
        model_version = '2024-05-13'
        instance = 'gcr/preview'
        api_version = '2024-10-21'
    elif ep == "o3":
        model_name = 'o3'
        model_version = '2025-04-16'
        instance = 'msrne/shared'
        api_version = '2025-04-01-preview'
    elif ep == "o3-mini":
        model_name = 'o3-mini'
        model_version = '2025-01-31'
        instance = 'msrne/shared'
        api_version = '2025-04-01-preview'
    elif ep == "o4-mini":
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


class IssueGen:
    def __init__(
        self,
        *,
        config_file: Path,
        dataset_path: str | Path,
        model: str,
        use_existing: bool,
        n_workers: int,
        experiment_id: Path,
        temperature: float = 0.5,
    ):
        self.experiment_id = experiment_id
        self.model = model
        self.use_existing = use_existing
        self.n_workers = n_workers
        self.temperature = temperature
        
        # Initialize Azure OpenAI client based on model
        self.client, self.deployment_name = setup_azure_client(model)
        
        dataset_path = Path(dataset_path)
        if not dataset_path.suffix == ".json":
            print("Warning: Expected a JSON file")
        # Load dataset + SWE-bench Verified
        self.dataset = json.loads(dataset_path.read_text())
        if FAIL_TO_PASS not in self.dataset[0]:
            raise ValueError(
                "Must be called with the result of swesmith.harness.gather, not the _all_patches.json file"
            )
        self.swebv = load_dataset("princeton-nlp/SWE-bench_Verified", split="test")

        config_file_name = config_file.name.rsplit(".", 1)[0]
        self.config = yaml.safe_load(config_file.read_text())

        settings = self.config.get("settings", {})
        self.n_instructions = settings.get("n_instructions", 1)
        self.max_var_tokens = settings.get("max_var_tokens", 10_000)

        self.output_file = (
            dataset_path.parent
            / experiment_id
            / (dataset_path.stem + f"__{config_file_name}_n{self.n_instructions}.jsonl")
        )
        self.output_file.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def get_test_output(self, instance: dict) -> str:
        # Get execution output from running pytest for this instance (from validation step)
        test_output = (
            LOG_DIR_RUN_VALIDATION
            / instance["repo"].split("/")[-1]
            / instance[KEY_INSTANCE_ID]
            / LOG_TEST_OUTPUT
        ).read_text()
        return maybe_shorten(
            test_output[
                test_output.find(TEST_OUTPUT_START)
                + len(TEST_OUTPUT_START) : test_output.find(TEST_OUTPUT_END)
            ],
            self.max_var_tokens,
            self.model,
        )

    def get_test_functions(self, instance: dict) -> tuple[list[str], list[str]]:
        """

        Returns:
            list of test functions, list of repos to remove
        """
        test_funcs = []
        repos_to_remove = []
        test_idxs = list(range(len(instance[FAIL_TO_PASS])))
        random.shuffle(test_idxs)
        for test_idx in test_idxs:
            test_func = get_test_function(instance, test_idx)
            if test_func["cloned"]:
                repos_to_remove.append(test_func["repo_name"])
            test_funcs.append(test_func["test_src"])
        return test_funcs, repos_to_remove

    def get_demo_issues(self) -> list[str]:
        """
        Get a list of demonstration issues from the config file.
        """
        problem_statements = [
            maybe_shorten(instance["problem_statement"], 2000, self.model)
            for instance in self.swebv
        ]  # type: ignore[index]
        random.shuffle(problem_statements)
        return problem_statements

    def call_llm(self, messages, tools=None, tool_choice=None, **kwargs):
        # Some models (like o3) don't support custom temperature values
        if any(reasoning_model in model.lower() for reasoning_model in ["o1", "o3", "o4"]):
            # Remove temperature from kwargs if it exists, as o3 models only support default (1)
            kwargs.pop('temperature', None)
        
        response = self.client.chat.completions.create(
            model=self.deployment_name,
            messages=messages,
            tools=tools,
            tool_choice=tool_choice,
            **kwargs
        )
        return response

    def generate_issue(self, instance: dict, idx_inst: int, f: TextIOWrapper) -> dict:
        # Set up logging information
        repo = instance["repo"].split("/")[-1]
        inst_dir = (
            LOG_DIR_ISSUE_GEN / self.experiment_id / repo / instance[KEY_INSTANCE_ID]
        )
        inst_dir.mkdir(parents=True, exist_ok=True)

        # Use existing issue text if already generated
        if self.use_existing and (inst_dir / "metadata.json").exists():
            with open(inst_dir / "metadata.json", "r") as f_:
                metadata = json.load(f_)
            for key, value in metadata["responses"].items():
                instance[key] = value
            with self._lock:
                f.write(json.dumps(instance) + "\n")
            return {
                "status": "skipped",
                "metadata": metadata,
            }

        # Get a reference instance from SWE-bench
        instance_curr = instance.copy()

        def format_prompt(prompt: str | None, config: dict, candidate: dict) -> str:
            if not prompt:
                return ""
            env = jinja2.Environment()

            def jinja_shuffle(seq):
                result = list(seq)
                random.shuffle(result)
                return result

            env.filters["shuffle"] = jinja_shuffle
            template = env.from_string(prompt)
            return template.render(**candidate, **config.get("parameters", {}))

        # Generate prompt
        messages = [
            {"content": self.config["system"], "role": "system"},
        ]
        if self.config["demonstration"]:
            messages.append(
                {
                    "content": format_prompt(
                        self.config["demonstration"],
                        self.config,
                        {"demo_problem_statements": self.get_demo_issues()},
                    ),
                    "role": "user",
                },
            )
        test_funcs, repos_to_remove = self.get_test_functions(instance_curr)
        messages.append(
            {
                "content": format_prompt(
                    self.config["instance"],
                    self.config,
                    instance_curr
                    | {
                        "test_output": self.get_test_output(instance_curr),
                        "test_funcs": test_funcs,
                    },
                ),
                "role": "user",
            },
        )

        with open(inst_dir / "messages.json", "w") as f_:
            json.dump(messages, f_, indent=4)

        # Generate n_instructions completions containing problem statements
        # Conditionally include temperature based on model compatibility
        call_kwargs = {"messages": messages, "n": self.n_instructions}
        if "o3" not in self.model.lower():
            call_kwargs["temperature"] = self.temperature
            
        response = self.call_llm(**call_kwargs)
        metadata = {
            "responses": {},
            "cost": response.usage.total_tokens * 0.002 / 1000,  # Calculate cost based on token usage
            "token_usage": {
                "completion_tokens": response.usage.completion_tokens,
                "prompt_tokens": response.usage.prompt_tokens,
                "total_tokens": response.usage.total_tokens,
            }
        }

        # Extract problem statements from completions
        for idx, choice in enumerate(response.choices):  # type: ignore[attr-defined]
            key = f"ps_basic_{idx}"
            if self.n_instructions == 1:
                key = "problem_statement"
            ps = choice.message.content  # type: ignore[attr-defined]
            instance[key] = ps
            metadata["responses"][key] = ps

        # Write to output file
        with self._lock:
            f.write(json.dumps(instance) + "\n")
        with open(inst_dir / "metadata.json", "w") as f_:
            json.dump(metadata, f_, indent=4)

        # Remove cloned repos
        for repo in repos_to_remove:
            with self._lock:
                shutil.rmtree(repo)

        return {
            "status": "completed",
            "metadata": metadata,
        }

    def run(self):
        # Setup output file
        mode = "w" if not os.path.exists(self.output_file) else "a"
        completed_ids = (
            [json.loads(x)[KEY_INSTANCE_ID] for x in open(self.output_file).readlines()]
            if os.path.exists(self.output_file)
            else []
        )
        logger.info(f"{len(completed_ids)} instances already completed.")
        dataset = [x for x in self.dataset if x[KEY_INSTANCE_ID] not in completed_ids]
        logger.info(f"Generating issues for {len(dataset)} instances.")
        logger.info(f"Output file: {self.output_file}")

        stats = {
            "💰": 0.0,
            "⏭️": 0,
            "❌": 0,
            "✅": 0,
        }

        # Create a thread pool and call generate_issue for each instance
        with open(self.output_file, mode) as f:
            with ThreadPoolExecutor(max_workers=self.n_workers) as executor:
                futures = []
                for idx_inst, instance in enumerate(dataset):
                    future = executor.submit(self.generate_issue, instance, idx_inst, f)
                    futures.append(future)

                # Wait for all futures to complete
                with logging_redirect_tqdm():
                    with tqdm(total=len(futures), desc="Generating issues") as pbar:
                        for future in as_completed(futures):
                            try:
                                result = future.result()
                            except KeyboardInterrupt:
                                raise
                            except Exception as e:
                                logger.error(
                                    f"Error processing instance: {e}", exc_info=True
                                )
                                stats["❌"] += 1
                                continue
                            if result["status"] == "skipped":
                                stats["⏭️"] += 1
                            elif result["status"] == "completed":
                                stats["✅"] += 1
                                stats["💰"] += result["metadata"]["cost"]
                            pbar.set_postfix(stats, refresh=True)
                            pbar.update(1)

        # Save output file as JSON
        output_file_json = self.output_file.parent / (self.output_file.stem + ".json")
        with open(output_file_json, "w") as f:
            temp = [json.loads(x) for x in open(self.output_file).readlines()]
            json.dump(temp, f, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "dataset_path", type=Path, help="Path to the dataset to annotate with bugs."
    )
    parser.add_argument(
        "--experiment_id",
        type=str,
        help=(
            "Experiment ID. This will be the first subfolder in the output directory. "
            "In most cases, it makes sense to set this to the stem of the config file."
        ),
        required=True,
    )
    parser.add_argument(
        "--config_file", type=Path, help="Path to the template config file."
    )
    parser.add_argument(
        "--model",
        type=str,
        help="Model to use for generation.",
        default="openai/gpt-4o",
    )
    parser.add_argument(
        "--use_existing",
        action="store_true",
        help="Use existing issue text if already generated.",
    )
    parser.add_argument(
        "--n_workers",
        type=int,
        help="Number of workers to use for generation.",
        default=1,
    )
    parser.add_argument(
        "--temperature",
        type=float,
        help="Temperature for LLM generation.",
        default=0.5,
    )
    args = parser.parse_args()
    if not args.use_existing:
        logger.warning(
            "!!! Warning: This script will not reuse existing issue texts but APPEND new versions."
        )
    if args.n_workers == 1:
        logger.warning(
            "Using only 1 worker for generation. You can speed up the generation by setting --n_workers > 1."
        )
    IssueGen(**vars(args)).run()
