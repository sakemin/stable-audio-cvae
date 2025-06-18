#!/usr/bin/env python3
"""
Create conditional dataset configuration for metallic drum samples.
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

def parse_file_list(file_path: str, base_path: str) -> List[Tuple[str, str, int]]:
    """
    Parse file list and extract class information.
    
    Args:
        file_path: Path to the file list (train_files.txt or test_files.txt)
        base_path: Base path to prepend to file paths
    
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
                full_path = os.path.abspath(os.path.join(base_path, line))
                parsed_files.append((full_path, class_name, class_index))
            else:
                print(f"Warning: Unknown class '{class_name}' in file '{line}'")
        else:
            print(f"Warning: Invalid file path format '{line}'")
    
    return parsed_files

def create_custom_metadata_module(files: List[Tuple[str, str, int]], output_path: str):
    """Create a custom metadata module that extracts condition from file path."""
    
    # Create mapping from absolute path to condition
    path_to_condition = {}
    for file_path, _, class_idx in files:
        path_to_condition[file_path] = class_idx
    
    module_content = '''"""
Custom metadata module for metallic drum dataset.
Extracts condition (drum class) from file path.
"""

import os

# Mapping from absolute path to condition index
PATH_TO_CONDITION = {path_to_condition}

def get_custom_metadata(info, audio):
    """Extract condition from file path."""
    try:
        # Get absolute path from info
        abs_path = os.path.abspath(info.get("path", ""))
        
        # Try to get condition from mapping
        if abs_path in PATH_TO_CONDITION:
            condition = PATH_TO_CONDITION[abs_path]
            return {{"condition": condition}}
        
        # Default: no condition found
        print(f"Warning: No condition found for file {{abs_path}}")
        return {{"__reject__": True}}  # Reject files without conditions
        
    except Exception as e:
        print(f"Error extracting condition from {{info}}: {{e}}")
        return {{"__reject__": True}}
'''
    
    with open(output_path, 'w') as f:
        f.write(module_content.format(
            path_to_condition=path_to_condition
        ))

def create_dataset_config(
    train_files: List[Tuple[str, str, int]], 
    output_dir: str,
    sample_rate: int = 44100,
    sample_size: int = 65536
) -> Dict:
    """
    Create dataset configuration for training.
    
    Args:
        train_files: List of training files with class info
        output_dir: Output directory for metadata modules
        sample_rate: Audio sample rate
        sample_size: Sample size in samples
    
    Returns:
        Training config dictionary
    """
    
    # Create custom metadata module
    train_metadata_path = os.path.join(output_dir, "metallic_drums_train_metadata.py")
    create_custom_metadata_module(train_files, train_metadata_path)
    
    train_config = {
        "dataset_type": "audio_files",  # Changed to audio_files to use absolute paths
        "sample_rate": sample_rate,
        "sample_size": sample_size,
        "random_crop": True,
        "force_channels": 2,
        "drop_last": True,
        "files": [file_path for file_path, _, _ in train_files],
        "custom_metadata_module": train_metadata_path
    }
    
    return train_config

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
    parser = argparse.ArgumentParser(description="Create conditional dataset from metallic drum samples")
    parser.add_argument("--metallic-path", required=True, help="Base path to metallic audio data directory")
    parser.add_argument("--oneshot-path", required=True, help="Base path to oneshot audio data directory")
    parser.add_argument("--train-files", default="../metallic/train_files.txt", help="Path to metallic train files list")
    parser.add_argument("--test-files", default="../oneshot_data/test_files.txt", help="Path to oneshot test files list")
    parser.add_argument("--output-dir", default="configs/dataset_configs", help="Output directory for configs")
    parser.add_argument("--sample-rate", type=int, default=44100, help="Audio sample rate")
    parser.add_argument("--sample-size", type=int, default=65536, help="Sample size in samples")
    
    args = parser.parse_args()
    
    # Parse and combine file lists with their respective base paths
    print("Parsing and combining file lists...")
    train_files = parse_file_list(args.train_files, args.metallic_path)
    test_files = parse_file_list(args.test_files, args.oneshot_path)
    
    # Combine files for training
    combined_train_files = train_files + test_files
    
    print(f"Found {len(combined_train_files)} total training files")
    
    # Print statistics
    print_class_statistics(combined_train_files, "Combined Training")
    
    # Create dataset config
    print("\nCreating dataset configuration...")
    train_config = create_dataset_config(
        combined_train_files, args.output_dir, args.sample_rate, args.sample_size
    )
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Save config
    train_config_path = os.path.join(args.output_dir, "metallic_drums_train.json")
    
    with open(train_config_path, 'w') as f:
        json.dump(train_config, f, indent=2)
    
    print(f"\nDataset config saved:")
    print(f"  Training: {train_config_path}")
    print(f"  Training metadata module: {os.path.join(args.output_dir, 'metallic_drums_train_metadata.py')}")
    
    # Print example usage
    print(f"\nExample usage:")
    print(f"python train.py \\")
    print(f"  --model-config stable_audio_tools/configs/model_configs/autoencoders/stable_audio_1_0_conditional_vae.json \\")
    print(f"  --dataset-config {train_config_path} \\")
    print(f"  --name metallic-drums-conditional-vae \\")
    print(f"  --batch-size 16")
    
    # Print sample condition values
    print(f"\nCondition mapping:")
    for class_name, class_idx in DRUM_CLASSES.items():
        print(f"  {class_name}: {class_idx}")

if __name__ == "__main__":
    main()