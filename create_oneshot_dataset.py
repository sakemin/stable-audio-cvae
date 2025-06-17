#!/usr/bin/env python3
"""
Create conditional dataset configuration for oneshot drum samples.
Extracts class information from file paths and creates dataset config for conditional autoencoder training.
"""

import os
import json
import argparse
from pathlib import Path
from typing import Dict, List, Tuple

# Define the drum classes
DRUM_CLASSES = {
    'kick': 0,
    'snare': 1, 
    'hat': 2,
    'clap': 3,
    'percussion': 4
}

def parse_file_list(file_path: str) -> List[Tuple[str, str, int]]:
    """
    Parse file list and extract class information.
    
    Args:
        file_path: Path to the file list (train_files.txt or test_files.txt)
    
    Returns:
        List of tuples: (full_path, class_name, class_index)
    """
    parsed_files = []
    
    with open(file_path, 'r') as f:
        lines = f.readlines()
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
            
        # Extract class from folder name (first part before /)
        if '/' in line:
            class_name = line.split('/')[0]
            if class_name in DRUM_CLASSES:
                class_index = DRUM_CLASSES[class_name]
                parsed_files.append((line, class_name, class_index))
            else:
                print(f"Warning: Unknown class '{class_name}' in file '{line}'")
        else:
            print(f"Warning: Invalid file path format '{line}'")
    
    return parsed_files

def create_custom_metadata_module(files: List[Tuple[str, str, int]], output_path: str):
    """Create a custom metadata module that extracts condition from file path."""
    
    # Create mapping from relative path to condition
    path_to_condition = {}
    for file_path, class_name, class_idx in files:
        path_to_condition[file_path] = class_idx
    
    module_content = '''"""
Custom metadata module for oneshot drum dataset.
Extracts condition (drum class) from file path.
"""

import os

# Mapping from relative path to condition index
PATH_TO_CONDITION = {path_to_condition}

def get_custom_metadata(info, audio):
    """Extract condition from file path."""
    try:
        # Get relative path from info
        relpath = info.get("relpath", "")
        
        # Try to get condition from mapping
        if relpath in PATH_TO_CONDITION:
            condition = PATH_TO_CONDITION[relpath]
            return {{"condition": condition}}
        
        # Fallback: extract from path directly
        if "/" in relpath:
            class_name = relpath.split("/")[0]
            drum_classes = {drum_classes}
            if class_name in drum_classes:
                return {{"condition": drum_classes[class_name]}}
        
        # Default: no condition found
        print(f"Warning: No condition found for file {{relpath}}")
        return {{"__reject__": True}}  # Reject files without conditions
        
    except Exception as e:
        print(f"Error extracting condition from {{info}}: {{e}}")
        return {{"__reject__": True}}
'''
    
    with open(output_path, 'w') as f:
        f.write(module_content.format(
            path_to_condition=path_to_condition,
            drum_classes=DRUM_CLASSES
        ))

def create_dataset_config(
    data_path: str,
    train_files: List[Tuple[str, str, int]], 
    test_files: List[Tuple[str, str, int]],
    output_dir: str,
    sample_rate: int = 44100,
    sample_size: int = 65536
) -> Tuple[Dict, Dict]:
    """
    Create dataset configuration for training and validation.
    
    Args:
        data_path: Base path to audio files
        train_files: List of training files with class info
        test_files: List of test files with class info
        output_dir: Output directory for metadata modules
        sample_rate: Audio sample rate
        sample_size: Sample size in samples
    
    Returns:
        Tuple of (train_config, val_config)
    """
    
    # Create custom metadata modules
    train_metadata_path = os.path.join(output_dir, "oneshot_drums_train_metadata.py")
    val_metadata_path = os.path.join(output_dir, "oneshot_drums_val_metadata.py")
    
    create_custom_metadata_module(train_files, train_metadata_path)
    create_custom_metadata_module(test_files, val_metadata_path)
    
    def create_config(files: List[Tuple[str, str, int]], split_name: str, metadata_module_path: str) -> Dict:
        return {
            "dataset_type": "audio_dir",
            "sample_rate": sample_rate,
            "sample_size": sample_size,
            "random_crop": True,
            "force_channels": 2,
            "drop_last": True,
            "datasets": [
                {
                    "id": f"oneshot_drums_{split_name}",
                    "path": data_path,
                    "custom_metadata_module": metadata_module_path
                }
            ]
        }
    
    train_config = create_config(train_files, "train", train_metadata_path)
    val_config = create_config(test_files, "val", val_metadata_path)
    
    return train_config, val_config

def print_class_statistics(files: List[Tuple[str, str, int]], split_name: str):
    """Print statistics about class distribution."""
    class_counts = {}
    for _, class_name, _ in files:
        class_counts[class_name] = class_counts.get(class_name, 0) + 1
    
    print(f"\n{split_name} split statistics:")
    print(f"Total files: {len(files)}")
    for class_name, count in sorted(class_counts.items()):
        percentage = (count / len(files)) * 100
        print(f"  {class_name}: {count} files ({percentage:.1f}%)")

def main():
    parser = argparse.ArgumentParser(description="Create conditional dataset from oneshot drum samples")
    parser.add_argument("--data-path", required=True, help="Base path to audio data directory")
    parser.add_argument("--train-files", default="oneshot_data/train_files.txt", help="Path to train files list")
    parser.add_argument("--test-files", default="oneshot_data/test_files.txt", help="Path to test files list")
    parser.add_argument("--output-dir", default="configs/dataset_configs", help="Output directory for configs")
    parser.add_argument("--sample-rate", type=int, default=44100, help="Audio sample rate")
    parser.add_argument("--sample-size", type=int, default=65536, help="Sample size in samples")
    
    args = parser.parse_args()
    
    # Parse file lists
    print("Parsing file lists...")
    train_files = parse_file_list(args.train_files)
    test_files = parse_file_list(args.test_files)
    
    print(f"Found {len(train_files)} training files")
    print(f"Found {len(test_files)} test files")
    
    # Print statistics
    print_class_statistics(train_files, "Training")
    print_class_statistics(test_files, "Test")
    
    # Create dataset configs
    print("\nCreating dataset configurations...")
    train_config, val_config = create_dataset_config(
        args.data_path, train_files, test_files, args.output_dir, args.sample_rate, args.sample_size
    )
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Save configs
    train_config_path = os.path.join(args.output_dir, "oneshot_drums_train.json")
    val_config_path = os.path.join(args.output_dir, "oneshot_drums_val.json")
    
    with open(train_config_path, 'w') as f:
        json.dump(train_config, f, indent=2)
    
    with open(val_config_path, 'w') as f:
        json.dump(val_config, f, indent=2)
    
    print(f"\nDataset configs saved:")
    print(f"  Training: {train_config_path}")
    print(f"  Validation: {val_config_path}")
    print(f"  Training metadata module: {os.path.join(args.output_dir, 'oneshot_drums_train_metadata.py')}")
    print(f"  Validation metadata module: {os.path.join(args.output_dir, 'oneshot_drums_val_metadata.py')}")
    
    # Print example usage
    print(f"\nExample usage:")
    print(f"python train.py \\")
    print(f"  --model-config stable_audio_tools/configs/model_configs/autoencoders/stable_audio_1_0_conditional_vae.json \\")
    print(f"  --dataset-config {train_config_path} \\")
    print(f"  --val-dataset-config {val_config_path} \\")
    print(f"  --name oneshot-drums-conditional-vae \\")
    print(f"  --batch-size 16")
    
    # Print sample condition values
    print(f"\nCondition mapping:")
    for class_name, class_idx in DRUM_CLASSES.items():
        print(f"  {class_name}: {class_idx}")

if __name__ == "__main__":
    main() 