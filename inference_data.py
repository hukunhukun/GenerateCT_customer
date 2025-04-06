import os
import pandas as pd

# 读取 valid_transformer.txt 文件
with open('transformer_train/valid_transformer.txt', 'r') as f:
    # 每行可能有换行符，读取所有行并去除空白字符
    lines = [line.strip() for line in f if line.strip()]

# 从完整路径中提取文件名
filenames = set(os.path.basename(path) for path in lines)

# 读取 CSV 文件
df = pd.read_csv('mydata/valid_data/Pelvic_600.csv')

# 筛选出 image 列在 filenames 中的行
filtered_df = df[df['image'].isin(filenames)]

# 只保留 image 与 text 两列，输出到新的 CSV 文件
filtered_df[['image', 'text']].to_csv('transformer_train/filtered_output.csv', index=False)

print("生成的 filtered_output.csv 包含 {} 行数据".format(len(filtered_df)))
