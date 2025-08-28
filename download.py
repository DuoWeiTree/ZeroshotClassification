def download():
    import pandas as pd
    from datasets import load_dataset

    # 1. 加载三个数据集
    print("开始加载数据集...")
    dataset_zh_cn = load_dataset("mteb/amazon_massive_intent", 'zh-CN')
    dataset_en_us = load_dataset("mteb/amazon_massive_intent", 'en')
    dataset_zh_tw = load_dataset("mteb/amazon_massive_intent", 'zh-TW')
    print("数据集加载完成。")

    # 2. 将每个split转换为DataFrame并添加语言列
    # 这里我们只取 "train" split 作为例子，你可以根据需要选择 "test" 或 "validation"
    # 简体中文
    df_zh_cn = pd.DataFrame(dataset_zh_cn['train'])
    df_zh_cn['language'] = 'zh-CN'

    # 英文
    df_en_us = pd.DataFrame(dataset_en_us['train'])
    df_en_us['language'] = 'en-US'

    # 繁体中文
    df_zh_tw = pd.DataFrame(dataset_zh_tw['train'])
    df_zh_tw['language'] = 'zh-TW'

    # 3. 合并所有DataFrame
    print("开始合并DataFrame...")
    combined_df = pd.concat([df_zh_cn, df_en_us, df_zh_tw], ignore_index=True)
    print("DataFrame合并完成。")

    # 4. 保存为CSV文件
    output_filename = "./testdata/amazon_massive_intent_zh_en.csv"
    combined_df.to_csv(output_filename, index=False)
    print(f"数据已成功保存到 {output_filename}")

    # 打印合并后数据的前5行以供检查
    print("\n--- 合并后的数据预览 ---")
    print(combined_df.head())
    print("\n--- 总行数 ---")
    print(len(combined_df))

