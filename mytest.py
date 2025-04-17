from datasets import load_dataset
ds = load_dataset("wikitext", "wikitext-2-v1", split="test")
print(ds[0])