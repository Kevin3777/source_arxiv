# Create a script to download and save the dataset
import os
import datasets as hf_datasets

# Create directory
os.makedirs("dataset/BioCite", exist_ok=True)

# Load dataset from HuggingFace
dataset = hf_datasets.load_dataset('mkhalifa/BioCite')

# Save to disk in the expected location
for split in dataset.keys():
    dataset[split].save_to_disk(f"dataset/BioCite/{split}")

print("Dataset saved successfully")