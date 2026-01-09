#!/usr/bin/env python
"""
ISIC 2019 Dataset Generator for Federated Learning
Creates per-client datasets based on 6 natural sources (centers).
Raw data: Images in folder + 2 CSVs (ground truth labels and metadata with source info).
"""
import os
import pandas as pd
import numpy as np
from PIL import Image
from collections import defaultdict
from utils.dataset_utils import separate_data, split_data, save_file, check


# Configuration
dataset_name = 'ISIC2019'
base_path = os.path.join('..', '..', 'isic2019')  # Adjust if needed
image_folder = os.path.join(base_path, 'ISIC_2019_Training_Input_preprocessed')
ground_truth_csv = os.path.join(base_path, 'ISIC_2019_Training_GroundTruth.csv')
metadata_csv = os.path.join(base_path, 'ISIC_2019_Training_Metadata_FL.csv')

output_dir = os.path.join('.', dataset_name)
config_path = os.path.join(output_dir, 'config.json')
train_path = os.path.join(output_dir, 'train', '')
test_path = os.path.join(output_dir, 'test', '')

# Dataset parameters
num_clients = 6  # 6 natural centers
num_classes = 9  # MEL, NV, BCC, AK, BKL, DF, VASC, SCC, UNK
batch_size = 10
train_ratio = 0.75

# Class names (in order)
class_names = ['MEL', 'NV', 'BCC', 'AK', 'BKL', 'DF', 'VASC', 'SCC', 'UNK']

# Mapping of source names to client IDs
source_to_client_id = {
    'BCN': 0,
    'HAM_vidir_molemax': 1,
    'HAM_vidir_modern': 2,
    'HAM_rosendahl': 3,
    'MSK': 4,
    'HAM_vienna_dias': 5
}


def load_and_preprocess_data():
    """
    Load images and labels from CSVs.
    Returns:
        images_dict: {image_id: image_array}
        labels_dict: {image_id: class_index}
        source_dict: {image_id: source_name}
    """
    print("Loading ground truth labels...")
    gt_df = pd.read_csv(ground_truth_csv)
    
    print("Loading metadata with source information...")
    meta_df = pd.read_csv(metadata_csv)
    
    images_dict = {}
    labels_dict = {}
    source_dict = {}
    
    print(f"Processing {len(gt_df)} images...")
    for idx, row in gt_df.iterrows():
        if idx % 5000 == 0:
            print(f"  Processed {idx}/{len(gt_df)}")
        
        image_id = row['image']
        
        # Load image
        image_path = os.path.join(image_folder, f"{image_id}.jpg")
        if not os.path.exists(image_path):
            print(f"Warning: Image file not found: {image_path}")
            continue
        
        try:
            img = Image.open(image_path).convert('RGB')
            # Normalize size so arrays per client are homogeneous (models expect fixed input size)
            target_size = (224, 224)
            img = img.resize(target_size, Image.BICUBIC)
            img_array = np.array(img, dtype=np.uint8)
        except Exception as e:
            print(f"Warning: Failed to load image {image_id}: {e}")
            continue
        
        # Get label (one-hot to class index)
        label_values = row[class_names].values
        label_idx = np.argmax(label_values)
        
        # Get source from metadata
        meta_row = meta_df[meta_df['image'] == image_id]
        if len(meta_row) == 0:
            print(f"Warning: No metadata found for {image_id}")
            continue
        
        source = meta_row.iloc[0]['dataset']
        # Clean source name (remove 'nan' suffix or other artifacts)
        if isinstance(source, str):
            # remove common artifacts like '4nan', '_nan', or stray 'nan'
            source = source.replace('4nan', '')
            source = source.replace('_nan', '')
            source = source.replace('nan', '')
            source = source.strip()
        
        images_dict[image_id] = img_array
        labels_dict[image_id] = label_idx
        source_dict[image_id] = source
    
    print(f"Successfully loaded {len(images_dict)} images.")
    return images_dict, labels_dict, source_dict


def organize_by_source(images_dict, labels_dict, source_dict):
    """
    Organize images by source (client).
    Returns:
        X: List of arrays (one per client)
        y: List of labels (one per client)
        statistic: Distribution stats per client
    """
    print("\nOrganizing data by source (client)...")
    
    # Initialize client data
    client_images = {i: [] for i in range(num_clients)}
    client_labels = {i: [] for i in range(num_clients)}
    
    unknown_sources = set()

    for image_id, img_array in images_dict.items():
        label = labels_dict[image_id]
        source = source_dict[image_id]

        # Normalize source string
        if isinstance(source, str):
            src = source.strip()
        else:
            src = str(source)

        # Try exact match first, then substring matching for robustness
        client_id = None
        if src in source_to_client_id:
            client_id = source_to_client_id[src]
        else:
            lower_src = src.lower()
            for key, cid in source_to_client_id.items():
                if key.lower() in lower_src or lower_src in key.lower():
                    client_id = cid
                    break

        if client_id is None:
            unknown_sources.add(src)
            # skip images with truly unknown sources
            print(f"Warning: Unknown source '{src}' for image {image_id}")
            continue

        client_images[client_id].append(img_array)
        client_labels[client_id].append(label)
    
    # Convert to numpy arrays
    X = []
    y = []
    statistic = []
    
    for client_id in range(num_clients):
        imgs = client_images.get(client_id, [])
        labs = client_labels.get(client_id, [])
        X.append(np.array(imgs))
        y.append(np.array(labs))

        if len(labs) == 0:
            print(f"Warning: No images for client {client_id}")
            statistic.append([])
        else:
            unique, counts = np.unique(y[client_id], return_counts=True)
            stat = [(int(cls), int(cnt)) for cls, cnt in zip(unique, counts)]
            statistic.append(stat)
            source_name = [k for k, v in source_to_client_id.items() if v == client_id][0]
            print(f"Client {client_id} ({source_name}): {len(y[client_id])} samples")
            print(f"  Labels distribution: {stat}")

    if len(unknown_sources) > 0:
        print('\nUnique unknown source values encountered:')
        for s in sorted(list(unknown_sources)):
            print('  -', s)
    
    return X, y, statistic


def generate_ISIC2019():
    """
    Main function to generate ISIC2019 dataset for federated learning.
    """
    print("=" * 60)
    print("ISIC 2019 Dataset Generator for Federated Learning")
    print("=" * 60)
    
    # Check if dataset already exists
    if check(config_path, train_path, test_path, num_clients, niid=False, 
             balance=False, partition=None):
        print("Dataset already exists. Skipping generation.")
        return
    
    # Load and preprocess data
    images_dict, labels_dict, source_dict = load_and_preprocess_data()
    
    # Organize by source (natural clients)
    X, y, statistic = organize_by_source(images_dict, labels_dict, source_dict)
    
    # Split into train/test
    print("\nSplitting data into train/test...")
    train_data, test_data = split_data(X, y)
    
    # Save to disk
    print("Saving dataset to disk...")
    save_file(config_path, train_path, test_path, train_data, test_data, 
              num_clients, num_classes, statistic, niid=False, balance=False, partition=None)
    
    print("\n" + "=" * 60)
    print("ISIC 2019 dataset generation complete!")
    print("=" * 60)


if __name__ == "__main__":
    generate_ISIC2019()
