import os
import pandas as pd

def load_cancer_type_data(
        base_path='../Dataset/Cancertype',
        cancers=['liver', 'lung', 'colorectal', 'pancreatic'],
        gpls=['GPL570', 'GPL20301', 'GPL24676', 'unknown']
    ):
    """读取多癌症类型数据并处理
    Args:
        base_path: 基础数据路径
        cancers: 癌症类型列表
        gpls: 平台信息标签列表
    Returns:
        DataFrame: 处理后的数据
    """
    # 读取标签文件
    with open('../Dataset/header.txt', 'r') as file:
        next(file)  # 跳过第一行
        header = [line.strip() for line in file]

    # 读取平台信息
    stat_info = pd.read_csv('../Dataset/Cancertype/stat_info.csv')
    platform_dict = dict(zip(stat_info['Sample'], stat_info['GPL']))
    
    all_data = pd.DataFrame(columns=header)
    all_labels = []
    all_platforms = []
    
    # 处理每个癌症类型文件夹
    for cancer_type in cancers:
        folder_path = os.path.join(base_path, cancer_type)
        print(f'\n处理癌症类型: {cancer_type}')
        
        # 获取所有GSE文件
        gse_files = [f for f in os.listdir(folder_path) 
                    if f.startswith('GSE') and f.endswith('.csv')]
        
        for file in gse_files:
            print('-'*20)
            print(f'处理文件: {file}')
            
            # 读取数据文件
            data_path = os.path.join(folder_path, file)
            data = pd.read_csv(data_path, index_col=0)
            
            # 处理缺失值并对齐header
            data = data.fillna(0)
            data = data.reindex(columns=header, fill_value=0)
            all_data = pd.concat([all_data, data], axis=0, ignore_index=True)
            print(f'数据形状: {data.shape}')
            
            # 获取平台信息
            gse_name = file.replace('.csv', '')
            if 'GPL' in gse_name:
                platform = 'GPL' + gse_name.split('GPL')[1]
            else:
                platform = platform_dict.get(gse_name, 'unknown')
            
            # 添加标签和平台信息
            all_labels.extend([cancer_type] * len(data))
            all_platforms.extend([platform] * len(data))
            print(f'平台: {platform}, 样本数: {len(data)}')
    
    # 合并所有数据
    result_df = pd.concat([
        all_data, 
        pd.Series(all_platforms, name='GPL'),
        pd.Series(all_labels, name='CancerType')
    ], axis=1)
    
    # 编码平台信息
    platform_encoder = {platform: idx for idx, platform in enumerate(gpls)}
    result_df['GPL'] = result_df['GPL'].map(lambda x: platform_encoder.get(x, 3))
    
    # 编码癌症类型
    cancer_encoder = {cancer: idx for idx, cancer in enumerate(cancers)}
    result_df['CancerType'] = result_df['CancerType'].map(cancer_encoder)
    
    print(f'\n最终数据形状: {result_df.shape}')
    print('平台编码:', platform_encoder)
    print('癌症类型编码:', cancer_encoder)
    return result_df

if __name__ == '__main__':
    # 加载数据
    result = load_cancer_type_data()
    
    # 保存处理后的数据
    output_path = '../Dataset/Cancertype/processed_data.csv'
    result.to_csv(output_path, index=False)
    
    print(f"\n数据已保存至: {output_path}")