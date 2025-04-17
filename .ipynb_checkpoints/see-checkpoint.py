import json

# 查看 JSON 文件的实际结构
with open('/root/autodl-tmp/intrinsic-source-citation/ours/arxiv_results.json', 'r') as f:
    data = json.load(f)
    print(type(data))  # 是列表还是字典
    print(len(data))   # 数据量
    print(data[0])     # 查看第一个元素的结构