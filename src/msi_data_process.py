import os
import pandas as pd

def load_msi_data(folder_path='../Dataset/MSI/data', 
                  gpls=['GPL570', 'GPL20301', 'GPL24676', 'unknown']):
    # 读取标签文件
    with open('../Dataset/header.txt', 'r') as file:
        # 跳过第一行
        next(file)
        # 读取剩余的行并存储为列表
        header = [line.strip() for line in file]

    # 读取stat_info.csv
    stat_info = pd.read_csv('../Dataset/MSI/stat_info.csv')
    platform_dict = dict(zip(stat_info['Sample'], stat_info['GPL']))
    
    # 获取所有GSE文件
    all_files = os.listdir(folder_path)
    gse_files = [f for f in all_files if f.startswith('GSE') and not f.endswith('-msi.csv')]
    
    all_data = pd.DataFrame(columns=header)
    all_labels = []
    all_platforms = []
    
    for file in gse_files:
        print('-'*20)
        print(file)
        # 读取数据文件
        data_path = os.path.join(folder_path, file)
        data = pd.read_csv(data_path, index_col=0)
        # 确保数据列与header一致，缺失的列补0
        data = data.reindex(columns=header, fill_value=0)
        all_data = pd.concat([all_data, data], axis=0, ignore_index=True)
        print(data.shape)
        # 读取对应的MSI标签文件
        msi_file = file.replace('.csv', '-msi.csv')
        msi_path = os.path.join(folder_path, msi_file)
        if os.path.exists(msi_path):
            msi_data = pd.read_csv(msi_path, index_col=0)
            msi_labels = msi_data.iloc[:, 0].to_list()
            all_labels.extend(msi_labels)
            print(len(all_labels))
                   
            # 获取数据集对应的平台信息
            gse_name = file.replace('.csv', '')
            # 如果gse_name中包含GPL信息，则使用GPL信息，否则使用stat_info中的信息
            if 'GPL' in gse_name:
                gse_platform = 'GPL' + gse_name.split('GPL')[1]
            else:
                gse_platform = platform_dict.get(gse_name, 'unknown')
            print(gse_platform)
            # 添加到列表中
            all_platforms.extend([gse_platform] * len(data))
            print(len(all_platforms))
    
    # 合并所有数据
    result_df = pd.concat([all_data, 
                           pd.Series(all_platforms, name='GPL'), 
                           pd.Series(all_labels, name='MSI')], axis=1)
    print(result_df.shape)
    print(result_df.head())
    
    # 将平台信息转换为数值编码
    platform_encoder = {platform: idx for idx, platform in enumerate(gpls)}
    print(platform_encoder)
    result_df['GPL'] = result_df['GPL'].map(lambda x: platform_encoder.get(x, 3))
      
    return result_df


if __name__ == '__main__':
    # 加载数据
    result = load_msi_data()
    
    # 保存处理后的数据
    output_path = '../Dataset/MSI/processed_data.csv'
    result.to_csv(output_path, index=False)
    
    print(f"数据形状: {result.shape}")
    print(f"数据已保存至: {output_path}")