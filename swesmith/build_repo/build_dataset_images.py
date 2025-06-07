"""
Purpose: Build Docker images for a dataset by creating repository mirrors and building TestSpec-based images.

Usage: python -m swesmith.build_repo.build_dataset_images --dataset_name SWE-bench/SWE-bench_Lite --split test --max_workers 4
"""

import argparse
import docker
import os
import subprocess
import shutil
import traceback

from concurrent.futures import ThreadPoolExecutor, as_completed
from dotenv import load_dotenv
from ghapi.all import GhApi
from pathlib import Path

from swebench.harness.constants import (
    BASE_IMAGE_BUILD_DIR,
    ENV_IMAGE_BUILD_DIR,
    INSTANCE_IMAGE_BUILD_DIR,
    KEY_INSTANCE_ID,
)
from swebench.harness.docker_build import (
    build_env_images,
    build_instance_image,
    BuildImageError,
)
from swebench.harness.docker_utils import list_images, clean_images
from swebench.harness.test_spec.test_spec import make_test_spec, TestSpec
from swebench.harness.utils import (
    load_swebench_dataset,
    run_threadpool,
    str2bool,
)
from swesmith.utils import get_repo_name
from tqdm import tqdm

load_dotenv()
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")
api = GhApi(token=GITHUB_TOKEN)

ORG_NAME = "OneRepoOneModel"


def does_repo_exist(repo_name: str) -> bool:
    """Check if a repository exists in the organization."""
    try:
        api.repos.get(ORG_NAME, repo_name)
        return True
    except Exception:
        return False


def create_repo_commit_mirror(repo: str, commit: str, org: str = ORG_NAME):
    """
    Create a mirror of the repository at the given commit.
    """
    repo_name = get_repo_name(repo, commit)
    if does_repo_exist(repo_name):
        return
    if repo_name in os.listdir():
        shutil.rmtree(repo_name)
    print(f"[{repo}][{commit[:8]}] Creating Mirror")
    api.repos.create_in_org(org, repo_name)
    for cmd in [
        f"git clone git@github.com:{repo}.git {repo_name}",
        (
            f"cd {repo_name}; "
            f"git checkout {commit}; "
            "rm -rf .git; "
            "git init; "
            'git config user.name "swesmith"; '
            'git config user.email "swesmith@anon.com"; '
            "rm -rf .github/workflows; "  # Remove workflows
            "git add .; "
            "git commit -m 'Initial commit'; "
            "git branch -M main; "
            f"git remote add origin git@github.com:{org}/{repo_name}.git; "
            "git push -u origin main",
        ),
        f"rm -rf {repo_name}",
    ]:
        subprocess.run(
            cmd,
            shell=True,
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    print(f"[{repo}][{commit[:8]}] Mirror created successfully")


def build_instance_image_with_mirror(
    test_spec: TestSpec,
    client: docker.DockerClient,
    org: str = ORG_NAME,
):
    """
    Build instance image for a TestSpec, creating repository mirror if needed.
    """
    # Extract repo and commit from instance data
    repo = test_spec.repo
    commit = test_spec.base_commit
    
    # Create repository mirror
    create_repo_commit_mirror(repo, commit, org)
    
    # Build the instance image
    build_instance_image(
        test_spec=test_spec,
        client=client,
        logger=None,
        nocache=False,
    )


def build_dataset_images(
    dataset_name: str,
    split: str,
    instance_ids: list = None,
    max_workers: int = 4,
    force_rebuild: bool = False,
    namespace: str = "swebench",
    instance_image_tag: str = "latest",
):
    """
    Build Docker images for each instance in the dataset.
    """
    # Load dataset
    dataset = load_swebench_dataset(dataset_name, split, instance_ids)
    
    if not dataset:
        print("No instances found in dataset.")
        return [], []
    
    print(f"Building images for {len(dataset)} instances from {dataset_name}")
    
    # Create TestSpecs for all instances
    test_specs = list(
        map(
            lambda instance: make_test_spec(
                instance, namespace=namespace, instance_image_tag=instance_image_tag
            ),
            dataset,
        )
    )
    
    client = docker.from_env()
    existing_images = list_images(client)
    
    # Build environment images first
    print("Building environment images...")
    build_env_images(client, dataset, force_rebuild, max_workers)
    
    # Filter test specs that need instance images built
    specs_to_build = []
    for test_spec in test_specs:
        image_exists = False
        if not force_rebuild:
            try:
                client.images.get(test_spec.instance_image_key)
                image_exists = True
            except docker.errors.ImageNotFound:
                pass
        if not image_exists:
            specs_to_build.append(test_spec)
    
    if not specs_to_build:
        print("No instance images to build.")
        return [], []
    
    print(f"Building {len(specs_to_build)} instance images...")
    for spec in specs_to_build:
        print(f"- {spec.instance_image_key}")
    
    # Build instance images in parallel
    successful, failed = list(), list()
    
    def build_wrapper(test_spec):
        try:
            build_instance_image_with_mirror(test_spec, client)
            return test_spec.instance_id
        except Exception as e:
            print(f"Failed to build image for {test_spec.instance_id}: {e}")
            traceback.print_exc()
            raise
    
    with tqdm(
        total=len(specs_to_build), smoothing=0, desc="Building instance images"
    ) as pbar:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Create futures for each image to build
            futures = {
                executor.submit(build_wrapper, spec): spec.instance_id
                for spec in specs_to_build
            }
            
            # Wait for each future to complete
            for future in as_completed(futures):
                pbar.update(1)
                try:
                    result = future.result()
                    successful.append(result)
                except BuildImageError as e:
                    print(f"BuildImageError {e.image_name}")
                    failed.append(futures[future])
                except Exception as e:
                    print(f"Error building image: {e}")
                    failed.append(futures[future])
    
    # Show results
    if len(failed) == 0:
        print("All instance images built successfully.")
    else:
        print(f"{len(failed)} instance images failed to build.")
    
    # Clean up images if needed
    clean_images(client, existing_images, "env", False)
    
    return successful, failed


def main():
    parser = argparse.ArgumentParser(
        description="Build Docker images for a dataset by creating repository mirrors."
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        required=True,
        help="Name of dataset or path to JSON file.",
    )
    parser.add_argument(
        "--split", 
        type=str, 
        default="test", 
        help="Split of the dataset"
    )
    parser.add_argument(
        "--instance_ids",
        nargs="+",
        type=str,
        help="Instance IDs to build images for (space separated)",
    )
    parser.add_argument(
        "--max_workers", 
        type=int, 
        default=4, 
        help="Maximum number of workers"
    )
    parser.add_argument(
        "--force_rebuild", 
        type=str2bool, 
        default=False, 
        help="Force rebuild of all images"
    )
    parser.add_argument(
        "--namespace", 
        type=str, 
        default="swebench", 
        help="Namespace for images"
    )
    parser.add_argument(
        "--instance_image_tag", 
        type=str, 
        default="latest", 
        help="Instance image tag"
    )
    
    args = parser.parse_args()
    
    successful, failed = build_dataset_images(**vars(args))
    
    print(f"\nSummary:")
    print(f"Successfully built: {len(successful)} images")
    print(f"Failed to build: {len(failed)} images")


if __name__ == "__main__":
    main()
