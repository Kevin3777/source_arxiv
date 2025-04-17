from datasets import load_dataset, DatasetDict

# 加载 jsonl 数据
dataset = load_dataset('json', data_files={'train': 'dataset/BioCite/pretrain/train.jsonl'})

# 保存为 load_from_disk 可加载的格式
dataset.save_to_disk('dataset/BioCite/pretrain')

qa_dataset = load_dataset('json', data_files={
    'qa_train': 'dataset/BioCite/qa/qa_train.jsonl',
    'qa_eval_in_domain': 'dataset/BioCite/qa/qa_eval_in_domain.jsonl',
    'qa_eval_out_of_domain': 'dataset/BioCite/qa/qa_eval_out_of_domain.jsonl',
})

qa_dataset.save_to_disk('dataset/BioCite/qa')

print("121311")