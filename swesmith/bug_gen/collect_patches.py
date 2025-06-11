"""
Purpose: Collect all the patches into a single json file that can be fed into swesmith.harness.valid

Usage: python -m swesmith.bug_gen.collect_patches logs/bug_gen/<repo>

NOTE: Must be with respect to a logs/bug_gen/<...>/ directory
"""

import argparse
import os
import json
from pathlib import Path

from swebench.harness.constants import KEY_INSTANCE_ID
from swebench.harness.test_spec.test_spec import make_test_spec
from swebench.harness.utils import load_swebench_dataset
from swesmith.constants import LOG_DIR_BUG_GEN, KEY_IMAGE_NAME, KEY_PATCH, PREFIX_BUG


def get_repo_to_image_mapping():
    """
    Create a cached mapping from repo names (in format repo__name.commit8) to TestSpec image keys.
    Uses file-based caching for persistence between runs.
    """
    cache_file = Path.home() / ".cache" / "1repo1model" / "repo_to_image_mapping.json"
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    
    # Try to load from cache first
    if cache_file.exists():
        try:
            with open(cache_file, 'r') as f:
                mapping = json.load(f)
            print(f"Loaded repo-to-image mapping from cache ({len(mapping)} entries)")
            return mapping
        except (json.JSONDecodeError, IOError) as e:
            print(f"Failed to load cache file: {e}, rebuilding mapping...")
    
    # Load the main SWE-bench dataset to get all instances
    print("Building repo-to-image mapping from SWE-bench dataset...")
    dataset = load_swebench_dataset("ZhengyanShi/SWE-bench_Verified_Temporal_9", "train", None)

    mapping = {}
    for instance in dataset:
        repo = instance["repo"]
        base_commit = instance["base_commit"]
        repo_name = f"{repo.replace('/', '__')}.{base_commit[:8]}"

        # Create TestSpec to get the correct image key
        test_spec = make_test_spec(instance, namespace="swebench", instance_image_tag="latest")
        mapping[repo_name] = test_spec.instance_image_key

    # Check if mapping size equals dataset size
    if len(mapping) != len(dataset):
        print(f"Warning: Mapping size ({len(mapping)}) does not equal dataset size ({len(dataset)})")
        print("This indicates duplicate repo_name values (same repo with same 8-char commit prefix)")
    
    # Save to cache
    try:
        with open(cache_file, 'w') as f:
            json.dump(mapping, f, indent=2)
        print(f"Saved repo-to-image mapping to cache ({len(mapping)} entries)")
    except IOError as e:
        print(f"Failed to save cache file: {e}")
    
    return mapping


def get_image_name_from_repo(repo_name: str) -> str:
    """
    Get the correct image name for a given repo name using TestSpec mapping.
    """
    mapping = get_repo_to_image_mapping()
    return mapping.get(repo_name, f"swebench:{repo_name}")


def main(bug_gen_path: str | Path, bug_type: str = "all", num_bugs: int = -1):
    """
    Collect all the patches into a single json file that can be fed into swebench.harness.valid
    :param repo_path: Path to the bug_gen logs.
    :param bug_type: Type of patches to collect. (default: all)
    :param num_bugs: Number of bugs to collect. (default: all)
    """
    bug_gen_path = Path(bug_gen_path)
    if not bug_gen_path.resolve().is_relative_to((Path() / LOG_DIR_BUG_GEN).resolve()):
        print(
            f"Warning: {bug_gen_path} may not point to a bug_gen log directory (should be in {(Path() / LOG_DIR_BUG_GEN).resolve()})."
        )

    repo_name = bug_gen_path.name
    image_name = get_image_name_from_repo(repo_name)

    patches = []
    prefix = f"{PREFIX_BUG}__"
    if bug_type != "all":
        prefix += bug_type + "_"
    exit_loop = False
    for root, _, files in os.walk(bug_gen_path):
        for file in files:
            if file.startswith(prefix) and file.endswith(".diff"):
                bug_type_and_uuid = file.split(f"{PREFIX_BUG}__")[-1].split(".diff")[0]
                instance_id = f"{repo_name}.{bug_type_and_uuid}"
                patch = {}

                # Add metadata if it exists
                metadata_file = f"metadata__{bug_type_and_uuid}.json"
                if os.path.exists(os.path.join(root, metadata_file)):
                    patch.update(json.load(open(os.path.join(root, metadata_file))))

                # Add necessary bug patch information
                patch.update(
                    {
                        KEY_INSTANCE_ID: instance_id,
                        KEY_PATCH: open(os.path.join(root, file), "r").read(),
                        KEY_IMAGE_NAME: image_name,
                    }
                )
                patches.append(patch)
                if num_bugs != -1 and len(patches) >= num_bugs:
                    exit_loop = True
                    break
        if exit_loop:
            break

    bug_patches_file = (
        bug_gen_path.parent / f"{bug_gen_path.name}_{bug_type}_patches.json"
    )
    if num_bugs != -1:
        bug_patches_file = bug_patches_file.with_name(
            bug_patches_file.stem + f"_n{num_bugs}" + bug_patches_file.suffix
        )
    if len(patches) > 0:
        with open(bug_patches_file, "w") as f:
            f.write(json.dumps(patches, indent=4))
        print(f"Saved {len(patches)} patches to {bug_patches_file}")
    else:
        print(f"No patches found for `{bug_type}` in {bug_gen_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Collect all the patches into a single json file that can be fed into swesmith.harness.valid"
    )
    parser.add_argument("bug_gen_path", help="Path to the bug_gen logs.")
    parser.add_argument(
        "--type",
        dest="bug_type",
        type=str,
        help="Type of patches to collect. (default: all)",
        default="all",
    )
    parser.add_argument(
        "-n",
        "--num_bugs",
        type=int,
        help="Number of bugs to collect. (default: all)",
        default=-1,
    )
    args = parser.parse_args()
    main(**vars(args))
