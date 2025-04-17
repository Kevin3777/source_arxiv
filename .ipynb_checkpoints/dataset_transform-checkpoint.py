import json
import os
import random
import nltk
import pickle
from datasets import Dataset, DatasetDict, disable_caching

# Ensure punkt tokenizer is downloaded
nltk.download('punkt', quiet=True)

def create_url_from_text(text, max_length=32):
    """
    Create a URL representation from text
    """
    # Tokenize
    tokens = nltk.word_tokenize(text)
    
    # Randomly select tokens
    if len(tokens) > max_length:
        rand_tokens = random.sample(tokens, max_length)
    else:
        rand_tokens = tokens[:max_length]
    
    # Create URL
    url = ' '.join(rand_tokens).lower()
    
    # Replace special characters
    url = ''.join(c if c.isalnum() or c.isspace() else '_' for c in url)
    
    return url

def prepare_dataset(input_file, output_dir, split_name='train'):
    """
    Prepare dataset with consistent directory structure
    """
    # Create necessary directories
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, split_name), exist_ok=True)
    
    # Disable dataset caching
    disable_caching()
    
    # Read and process JSON file
    with open(input_file, 'r') as f:
        data = json.load(f)
    
    # Prepare data
    processed_data = []
    url_map = {}
    
    for item in data:
        # Generate URL
        text = item.get('text', '')
        new_url = create_url_from_text(text)
        
        # Store original and new URL mapping
        old_url = item.get('title', '')
        url_map[old_url] = new_url
        
        processed_data.append({
            'text': text,
            'url': new_url,
            'title': item.get('title', ''),
            'categories': ','.join(item.get('categories', [])),
            'type': item.get('type', '')
        })
    
    # Create HuggingFace Dataset
    dataset = Dataset.from_list(processed_data)
    
    # Create DatasetDict
    dataset_dict = DatasetDict({split_name: dataset})
    
    # Save to disk with correct structure
    dataset_dict.save_to_disk(output_dir)
    
    # Save URL map
    url_map_path = os.path.join(output_dir, 'url_map.pkl')
    with open(url_map_path, 'wb') as f:
        pickle.dump(url_map, f)
    
    return dataset_dict, url_map

def main():
    # Input and output paths
    input_file = '/root/autodl-tmp/intrinsic-source-citation/ours/arxiv_results.json'
    
    # Pretrain dataset
    print("Preparing Pretrain Dataset...")
    pretrain_output_dir = 'dataset/ours/pretrain'
    pretrain_dataset, pretrain_url_map = prepare_dataset(
        input_file, 
        pretrain_output_dir,
        split_name='train'
    )
    
    # QA dataset
    print("Preparing QA Dataset...")
    qa_output_dir = 'dataset/ours/qa'
    qa_dataset, qa_url_map = prepare_dataset(
        input_file, 
        qa_output_dir,
        split_name='qa_train'
    )
    
    print("Dataset Processing Complete")
    print(f"Pretrain Dataset: {pretrain_dataset}")
    print(f"QA Dataset: {qa_dataset}")

if __name__ == '__main__':
    main()