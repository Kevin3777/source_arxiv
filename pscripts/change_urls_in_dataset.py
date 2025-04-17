import argparse
import os
import numpy as np
import pickle as pkl
from datasets import load_from_disk
from transformers import AutoTokenizer
from tqdm import tqdm
from utils.trie import MarisaTrie

def main(args):
    # Load dataset with error handling
    try:
        dataset = load_from_disk(args.dataset_path)
        # Ensure we're using the correct split
        if isinstance(dataset, dict):
            dataset = dataset[args.splits]
    except Exception as e:
        print(f"Error loading dataset: {e}")
        print(f"Dataset path: {args.dataset_path}")
        print(f"Available splits: {dataset.keys() if isinstance(dataset, dict) else 'N/A'}")
        raise
    
    # Load or create tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    
    # Ensure output directory exists
    os.makedirs(args.out_root, exist_ok=True)
    
    # Process URL trie
    url_ids = []
    url_trie_path = os.path.join(args.out_root, 'url_trie.pkl')
    
    # Debug print dataset structure
    print(f"Dataset structure: {dataset}")
    print(f"Number of samples: {len(dataset)}")
    
    # Collect URLs with error handling
    print("Collecting URLs...")
    urls = []
    for sample in dataset:
        try:
            # Ensure 'url' column exists
            if isinstance(sample, dict) and 'url' in sample:
                urls.append(sample['url'])
            else:
                print(f"Skipping sample due to missing 'url': {sample}")
        except Exception as e:
            print(f"Error processing sample: {e}")
    
    # Tokenize URLs
    for url in tqdm(urls):
        # Tokenize URL with special tokens
        ids = tokenizer(url, add_special_tokens=False)['input_ids']
        
        # Add special tokens if available
        if hasattr(tokenizer, 'additional_special_tokens_ids'):
            ids = [tokenizer.additional_special_tokens_ids[0]] + ids + [tokenizer.additional_special_tokens_ids[1]]
        
        url_ids.append(ids)
    
    # Create and save URL trie
    url_trie = MarisaTrie(sequences=url_ids)
    pkl.dump(url_trie, open(url_trie_path, 'wb'))
    
    # Prepare samples for streaming
    samples = []
    for sample in tqdm(dataset):
        try:
            # Ensure sample has text and url
            processed_sample = {
                'text': sample.get('text', ''),
                'url': sample.get('url', '')
            }
            samples.append(processed_sample)
        except Exception as e:
            print(f"Error processing sample: {e}")
    
    # Save samples
    shard_path = os.path.join(args.out_root, 'shard.npy')
    with open(shard_path, 'wb') as f:
        np.save(f, samples)
    
    print(f"Processed {len(samples)} samples")
    print(f"URL Trie saved to {url_trie_path}")
    print(f"Samples saved to {shard_path}")

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='parawiki')
    parser.add_argument('--dataset_path', type=str, required=True)
    parser.add_argument('--out_root', type=str, required=True)
    parser.add_argument('--splits', type=str, default='train')
    parser.add_argument('--tokenizer', type=str, required=True)
    parser.add_argument('--concat_tokens', type=int, default=1024)
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--packing_method', type=str, default='url_first')
    parser.add_argument('--build_trie', action='store_true')
    parser.add_argument('--eos_text', type=str, default='</s>')
    parser.add_argument('--no_reset_doc_positions', action='store_true')
    return parser.parse_args()

if __name__ == '__main__':
    main(parse_args())