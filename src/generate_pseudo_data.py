import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

'''
生成伪数据集
'''

def generate_constrained_random_numbers(n, target_sum):
    numbers = np.random.random(n-1)
    numbers = numbers / np.sum(numbers) * target_sum * np.random.random()
    last_number = target_sum - np.sum(numbers)
    return np.append(numbers, last_number)

def generate_mixed_samples(platform_folders, header, num_samples=100, seed=42):
    # 创建字典来存储不同细胞类型的数据
    cells_data = {}
    # 读取细胞系数据
    cells_folder = ['细胞系', '巨噬细胞', 'B细胞', 'T细胞']
    for folder in cells_folder:
        cell_folder = os.path.join(platform_folders, folder)
        cells = pd.DataFrame(columns=header)
        for filename in os.listdir(cell_folder):
            if filename.endswith('.csv'):
                file_path = os.path.join(cell_folder, filename)
                data = pd.read_csv(file_path, index_col=0)
                data = data.astype(float)
                cells = pd.concat([cells, data], axis=0, ignore_index=True)
        cells.fillna(0, inplace=True)
        cells_data[folder] = cells
    # 生成混合样本
    new_samples = []
    for i in range(num_samples):
        # 细胞系数据处理
        cell_sample = cells_data['细胞系'].sample(random_state=seed+i)
        cell_ratio = np.random.uniform(0.2, 0.5)
        combined_values = (cell_sample * cell_ratio).values 
        # 生成3个和为1-cell_ratio的随机数  
        random_ratios = generate_constrained_random_numbers(3, 1-cell_ratio) 
        combined_values += (cells_data['巨噬细胞'].sample(random_state=seed+i) * random_ratios[0]).values
        combined_values += (cells_data['B细胞'].sample(random_state=seed+i) * random_ratios[1]).values
        combined_values += (cells_data['T细胞'].sample(random_state=seed+i) * random_ratios[2]).values
        new_samples.append(combined_values)
    new_samples = np.vstack(new_samples)
    # 对每个样本进行归一化
    new_samples = MinMaxScaler().fit_transform(new_samples.T).T  # 转置后归一化，再转置回来     

    output_file = os.path.join(platform_folders, 'mixed_samples.csv')
    pd.DataFrame(new_samples).to_csv(output_file, index=False)
    print(f"数据已保存至: {output_file}")


if __name__ == '__main__':
    with open('../Dataset/header.txt', 'r') as file:
        # 跳过第一行
        next(file)
        # 读取剩余的行并存储为列表
        header = [line.strip() for line in file]

    generate_mixed_samples('../Dataset/Pseudo data/GPL570', header, num_samples=5)  
    print('-'*20)
    generate_mixed_samples('../Dataset/Pseudo data/GPL20301', header, num_samples=5) 
    print('-'*20) 
    generate_mixed_samples('../Dataset/Pseudo data/GPL24676', header, num_samples=5)  
