#!/usr/bin/env python3
"""
Script to download all Docker images required for a SWE-bench dataset to local storage.
This script will pull all instance images without running any tests or cleanup.
"""

import docker
import json
from pathlib import Path
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging
from tqdm import tqdm

from swebench.harness.constants import KEY_INSTANCE_ID
from swebench.harness.test_spec.test_spec import make_test_spec
from swebench.harness.utils import load_swebench_dataset

# Set up logging to work with tqdm
logging.basicConfig(level=logging.WARNING, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def download_image(client, image_key, instance_id):
    """
    Download a single Docker image.
    
    Args:
        client: Docker client
        image_key: Full image name/tag to download
        instance_id: Instance ID for logging purposes
    
    Returns:
        tuple: (instance_id, success, message)
    """
    try:
        # Check if image already exists locally
        try:
            client.images.get(image_key)
            return instance_id, True, "Already exists locally"
        except docker.errors.ImageNotFound:
            pass
        
        # Pull the image
        client.images.pull(image_key)
        return instance_id, True, "Downloaded successfully"
        
    except docker.errors.NotFound:
        error_msg = f"Image not found in registry: {image_key}"
        return instance_id, False, error_msg
        
    except docker.errors.APIError as e:
        error_msg = f"Docker API error: {str(e)}"
        return instance_id, False, error_msg
        
    except Exception as e:
        error_msg = f"Unexpected error: {str(e)}"
        return instance_id, False, error_msg


def download_all_images(
    dataset_name: str,
    split: str,
    namespace: str,
    instance_image_tag: str = "latest",
    instance_ids: list = None,
    max_workers: int = 4,
    output_report: str = None
):
    """
    Download all Docker images required for the dataset.
    
    Args:
        dataset_name: Name of the dataset
        split: Dataset split (test, dev, etc.)
        namespace: Namespace for remote images
        instance_image_tag: Tag for instance images
        instance_ids: Optional list of specific instance IDs to process
        max_workers: Number of parallel download workers
        output_report: Optional path to save download report
    """
    
    # Initialize Docker client
    try:
        client = docker.from_env()
        print("✅ Docker client initialized successfully")
    except Exception as e:
        print(f"❌ Failed to initialize Docker client: {e}")
        return
    
    # Load dataset
    try:
        dataset = load_swebench_dataset(dataset_name, split, instance_ids)
        if instance_ids:
            dataset = [item for item in dataset if item[KEY_INSTANCE_ID] in instance_ids]
        print(f"✅ Loaded dataset with {len(dataset)} instances")
    except Exception as e:
        print(f"❌ Failed to load dataset: {e}")
        return
    
    if not dataset:
        print("⚠️  No instances to process")
        return
    
    # Create test specs to get image keys
    try:
        test_specs = []
        for instance in dataset:
            test_spec = make_test_spec(
                instance, 
                namespace=namespace, 
                instance_image_tag=instance_image_tag
            )
            test_specs.append(test_spec)
        print(f"✅ Created {len(test_specs)} test specifications")
    except Exception as e:
        print(f"❌ Failed to create test specs: {e}")
        return
    
    # Extract unique remote image keys (only download remote images)
    image_keys = {}
    for test_spec in test_specs:
        if test_spec.is_remote_image:  # Only download remote images
            image_keys[test_spec.instance_image_key] = test_spec.instance_id
    
    print(f"🎯 Found {len(image_keys)} unique remote images to download")
    
    if not image_keys:
        print("⚠️  No remote images found to download")
        return
    
    # Check existing images to avoid unnecessary downloads
    existing_images = set()
    for image_key in image_keys.keys():
        try:
            client.images.get(image_key)
            existing_images.add(image_key)
        except docker.errors.ImageNotFound:
            pass
    
    if existing_images:
        print(f"📦 Found {len(existing_images)} images already downloaded locally")
    
    # Download images in parallel
    results = []
    successful_downloads = 0
    failed_downloads = 0
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all download tasks
        future_to_image = {
            executor.submit(download_image, client, image_key, instance_id): (image_key, instance_id)
            for image_key, instance_id in image_keys.items()
        }
        
        # Process completed downloads with progress bar
        with tqdm(total=len(image_keys), desc="📥 Downloading images", unit="image", 
                 bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]') as pbar:
            for future in as_completed(future_to_image):
                image_key, instance_id = future_to_image[future]
                try:
                    instance_id, success, message = future.result()
                    results.append({
                        'instance_id': instance_id,
                        'image_key': image_key,
                        'success': success,
                        'message': message
                    })
                    
                    if success:
                        successful_downloads += 1
                    else:
                        failed_downloads += 1
                        # Only log errors for actual failures, not existing images
                        if "Already exists locally" not in message:
                            tqdm.write(f"❌ Failed {instance_id}: {message}")
                    
                    pbar.set_postfix_str(f"✅ {successful_downloads} | ❌ {failed_downloads}")
                        
                except Exception as e:
                    tqdm.write(f"❌ Error processing {image_key}: {e}")
                    results.append({
                        'instance_id': instance_id,
                        'image_key': image_key,
                        'success': False,
                        'message': f'Processing error: {str(e)}'
                    })
                    failed_downloads += 1
                    pbar.set_postfix_str(f"✅ {successful_downloads} | ❌ {failed_downloads}")
                
                pbar.update(1)

    # Print summary
    total_images = len(image_keys)
    print("\n" + "="*60)
    print("📊 DOWNLOAD SUMMARY")
    print("="*60)
    print(f"Total images: {total_images}")
    print(f"✅ Successful: {successful_downloads}")
    print(f"❌ Failed: {failed_downloads}")
    print(f"📈 Success rate: {successful_downloads/total_images*100:.1f}%")
    
    # Save detailed report if requested
    if output_report:
        try:
            report_path = Path(output_report)
            report_path.parent.mkdir(parents=True, exist_ok=True)
            
            report_data = {
                'summary': {
                    'total_images': total_images,
                    'successful_downloads': successful_downloads,
                    'failed_downloads': failed_downloads,
                    'success_rate': successful_downloads/total_images*100 if total_images > 0 else 0
                },
                'details': results
            }
            
            with open(report_path, 'w') as f:
                json.dump(report_data, f, indent=2)
            
            print(f"📋 Detailed report saved to: {report_path}")
            
        except Exception as e:
            print(f"❌ Failed to save report: {e}")
    
    # List failed downloads
    if failed_downloads > 0:
        print("\n❌ Failed downloads:")
        for result in results:
            if not result['success'] and "Already exists locally" not in result['message']:
                print(f"  - {result['instance_id']}: {result['message']}")


def main():
    parser = ArgumentParser(
        description="Download all Docker images required for SWE-bench evaluation",
        formatter_class=ArgumentDefaultsHelpFormatter,
    )
    
    # 与 run_evaluation.py 保持一致的参数设置
    parser.add_argument(
        "--dataset_name",
        default="princeton-nlp/SWE-bench_Verified",  # 与原文件一致
        type=str,
        help="Name of dataset or path to JSON file.",  # 与原文件一致的描述
    )
    
    parser.add_argument(
        "--split", 
        type=str, 
        default="train",  # 与原文件一致
        help="Split of the dataset"  # 与原文件一致的描述
    )
    
    parser.add_argument(
        "--namespace",
        type=str,
        default="swebench",  # 与原文件一致
        help="Namespace for images",  # 与原文件一致的描述
    )
    
    parser.add_argument(
        "--instance_image_tag",
        type=str,
        default="latest",  # 与原文件一致
        help="Instance image tag",  # 与原文件一致的描述
    )
    
    parser.add_argument(
        "--instance_ids",
        nargs="+",
        type=str,
        help="Instance IDs to run (space separated)",  # 与原文件一致的描述
    )
    
    parser.add_argument(
        "--max_workers",
        type=int,
        default=4,  # 与原文件一致
        help="Maximum number of workers (should be <= 75%% of CPU cores)",  # 与原文件一致的描述
    )
    
    parser.add_argument(
        "--output_report",
        type=str,
        help="Path to save detailed download report (JSON format)",
    )
    
    args = parser.parse_args()
    
    download_all_images(
        dataset_name=args.dataset_name,
        split=args.split,
        namespace=args.namespace,
        instance_image_tag=args.instance_image_tag,
        instance_ids=args.instance_ids,
        max_workers=args.max_workers,
        output_report=args.output_report,
    )


if __name__ == "__main__":
    main()