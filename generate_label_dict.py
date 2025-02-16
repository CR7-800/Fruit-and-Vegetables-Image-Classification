# 遍歷目錄並生成 JSON 文件

import os
import json

# 定義基礎路徑
base_path = 'D:\\03_fruit'

# 定義目錄結構
directories = ['train', 'test', 'validation']

# 初始化字典
label_dict = {}

# 遍歷每個目錄
for dir_index, directory in enumerate(directories):
    dir_path = os.path.join(base_path, directory)
    for subdir in os.listdir(dir_path):
        subdir_path = os.path.join(dir_path, subdir)
        if os.path.isdir(subdir_path):
            label_dict[f"{dir_index}_{subdir}"] = subdir_path

# 將字典寫入 JSON 文件
with open('label_dict.json', 'w', encoding='utf-8') as f:
    json.dump(label_dict, f, ensure_ascii=False, indent=4)