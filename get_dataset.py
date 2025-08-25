import pandas as pd
from datasets import load_dataset

def process_and_save(split_name):
    """
    加载指定的数据集分割，并将其保存为 CSV 文件。
    """
    try:
        # 加载数据集的指定分割（train, validation, test）
        dataset = load_dataset("SetFit/amazon_massive_intent_zh-CN", split=split_name)
    except Exception as e:
        print(f"加载数据集 '{split_name}' 时出错：{e}")
        return

    # 将数据集转换为 Pandas DataFrame
    df = dataset.to_pandas()

    file_name = f'amazon_massive_intent_zh-CN_{split_name}.csv'

    # 将 DataFrame 保存为 CSV 文件
    df.to_csv(file_name, index=False, encoding='utf-8')

    print(f"成功保存 {split_name} 数据集到 {file_name}。")
    print(f"总行数: {len(df)}")

# 分别处理三个分割
print("正在处理训练集（train）...")
process_and_save('train')

print("\n" + "="*50 + "\n")

print("正在处理验证集（validation）...")
process_and_save('validation')

print("\n" + "="*50 + "\n")

print("正在处理测试集（test）...")
process_and_save('test')