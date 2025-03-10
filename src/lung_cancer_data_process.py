import pandas as pd
import os

def load_lung_cancer_data(
        folder_path='../Dataset/Lung_cancer_subtybes/data', 
        gpls=['GPL570', 'GPL20301', 'GPL24676', 'unknown'],
        subtypes=['非小细胞肺癌', '小细胞肺癌', '肺腺癌细胞系']
        ):
    """读取肺癌亚型数据并处理
    Args:
        folder_path: 数据文件夹路径
        gpls: 平台信息标签列表
    Returns:
        DataFrame: 处理后的数据
    """
    # 读取标签文件
    with open('../Dataset/header.txt', 'r') as file:
        next(file)  # 跳过第一行
        header = [line.strip() for line in file]

    # 读取亚型信息
    subtype_info = pd.read_csv('../Dataset/Lung_cancer_subtybes/stat_info.csv',
                               usecols=[0,1,7], encoding='GB2312')
    platform_dict = dict(zip(subtype_info.iloc[:, 0], subtype_info.iloc[:, 1]))  # 第二列为GPL
    label_dict = dict(zip(subtype_info.iloc[:, 0], subtype_info.iloc[:, -1]))    # 倒数第3列为标签
    
    # 获取所有CSV文件
    all_files = [f for f in os.listdir(folder_path) if f.endswith('.csv')]
    
    all_data = pd.DataFrame(columns=header)
    all_labels = []
    all_platforms = []
    
    for file in all_files:
        print('-'*20)
        print(f'处理文件: {file}')
        
        # 读取数据文件
        data_path = os.path.join(folder_path, file)
        data = pd.read_csv(data_path, index_col=0)
        
        # 对齐header，缺失的补0
        data = data.reindex(columns=header, fill_value=0)
        data.fillna(0, inplace=True)

        # 合并数据
        all_data = pd.concat([all_data, data], axis=0, ignore_index=True)
        print(f'数据形状: {data.shape}')
        
        # 获取数据集对应的平台和标签信息
        gse_name = file.replace('.csv', '')
        platform = platform_dict.get(gse_name, 'unknown')
        label = [label_dict.get(gse_name)]
        
        # 添加到列表中
        all_platforms.extend([platform] * len(data))
        all_labels.extend(label * len(data))
        print(f'平台: {platform}, 样本数: {len(data)}')
    
    # 合并所有数据
    result_df = pd.concat([
        all_data, 
        pd.Series(all_platforms, name='GPL'),
        pd.Series(all_labels, name='Subtype')
    ], axis=1)

    # 编码GPL和Subtype列
    platform_encoder = {platform: idx for idx, platform in enumerate(gpls)}
    result_df['GPL'] = result_df['GPL'].map(lambda x: platform_encoder.get(x, 3))
    result_df['Subtype'] = result_df['Subtype'].replace('非小细胞肺癌细胞', '非小细胞肺癌')
    subtype_encoder = {subtype: idx for idx, subtype in enumerate(subtypes)}
    result_df['Subtype'] = result_df['Subtype'].map(subtype_encoder)
    
    print(f'最终数据形状: {result_df.shape}')
    print(result_df.head()) 
    return result_df

if __name__ == '__main__':
    # 加载数据
    result = load_lung_cancer_data()
    
    # 保存处理后的数据
    output_path = '../Dataset/Lung_cancer_subtybes/processed_data.csv'
    result.to_csv(output_path, index=False)
    
    print(f"数据形状: {result.shape}")
    print(f"数据已保存至: {output_path}")