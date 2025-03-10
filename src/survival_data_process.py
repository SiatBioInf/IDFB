import os
import pandas as pd

def load_survival_data(
        folder_path='../Dataset/Survival_analysis/data', 
        gpls=['GPL570', 'GPL20301', 'GPL24676', 'unknown']
    ):
    """读取生存分析数据并处理
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

    # 读取平台信息
    stat_info = pd.read_csv('../Dataset/Survival_analysis/stat_info.csv')
    platform_dict = dict(zip(stat_info['Sample'], stat_info['GPL']))
    
    # 获取所有不含-os的CSV文件
    all_files = [f for f in os.listdir(folder_path) 
                if f.startswith('GSE') and not f.endswith('_os.csv')]
    
    all_data = pd.DataFrame(columns=header)
    all_labels = []
    all_platforms = []
    
    for file in all_files:
        print('-'*20)
        print(f'处理文件: {file}')
        
        # 读取数据文件
        data_path = os.path.join(folder_path, file)
        data = pd.read_csv(data_path, index_col=0)
        
        # 对齐header并处理缺失值
        data = data.reindex(columns=header, fill_value=0)
        data = data.fillna(0)
        all_data = pd.concat([all_data, data], axis=0, ignore_index=True)
        print(f'数据形状: {data.shape}')
        
        # 读取对应的生存分析标签文件
        os_file = file.replace('.csv', '_os.csv')
        os_path = os.path.join(folder_path, os_file)
        if os.path.exists(os_path):
            os_data = pd.read_csv(os_path, index_col=0)
            os_labels = os_data.iloc[:, 0].tolist()
            all_labels.extend(os_labels)
            print(f'标签数量: {len(os_labels)}')
            
            # 获取数据集对应的平台信息
            gse_name = file.replace('.csv', '')
            if 'GPL' in gse_name:
                platform = 'GPL' + gse_name.split('GPL')[1]
            else:
                platform = platform_dict.get(gse_name, 'unknown')
            print(f'平台: {platform}')
            
            # 添加平台信息
            all_platforms.extend([platform] * len(data))
            print(f'累计平台数: {len(all_platforms)}')
    
    # 合并所有数据
    result_df = pd.concat([
        all_data, 
        pd.Series(all_platforms, name='GPL'),
        pd.Series(all_labels, name='Survival')
    ], axis=1)
    
    # 编码平台信息
    platform_encoder = {platform: idx for idx, platform in enumerate(gpls)}
    result_df['GPL'] = result_df['GPL'].map(lambda x: platform_encoder.get(x, 3))
    
    print(f'最终数据形状: {result_df.shape}')
    print(result_df.head())
    return result_df

if __name__ == '__main__':
    # 加载数据
    result = load_survival_data()
    
    # 保存处理后的数据
    output_path = '../Dataset/Survival_analysis/processed_data.csv'
    result.to_csv(output_path, index=False)
    
    print(f"数据已保存至: {output_path}")