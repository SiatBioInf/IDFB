import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import silhouette_score
import argparse


def mean_average_precision(x: pd.DataFrame, y: pd.Series, neighbor_frac: float = 0.01) -> float:
    """计算平均精确率均值(MAP)
    Args:
        x: 整合后数据的特征矩阵 (DataFrame)
        y: 真实标签 (Series)
        neighbor_frac: 近邻比例
    Returns:
        float: MAP分数
    """
    # 计算近邻数量k
    n_samples = len(y)
    k = max(int(n_samples * neighbor_frac), 1)
    
    # 获取k近邻索引
    knn = NearestNeighbors(n_neighbors=k+1)
    knn.fit(x)
    
    # 计算每个类别的AP
    class_aps = []
    for label in y.unique():
        # 获取该类别的样本索引
        class_mask = y == label
        class_samples = x[class_mask]
        if len(class_samples) == 0:
            continue
            
        # 获取该类别样本的近邻
        class_neighbors = knn.kneighbors(class_samples, return_distance=False)[:, 1:] # 去除自身
        
        # 计算近邻中同类别样本的比例
        neighbor_labels = y.values[class_neighbors]
        matches = (neighbor_labels == label)
        
        # 计算该类别的AP
        ap = np.mean([np.sum(match) / k for match in matches])
        class_aps.append(ap)
    
    # 返回所有类别AP的平均值
    return np.mean(class_aps)


def avg_silhouette_width(x: pd.DataFrame, y: pd.Series) -> float:
    """计算平均轮廓系数并归一化到[0,1]区间
    Args:
        x: 整合后数据的特征矩阵 (DataFrame)
        y: 真实标签 (Series)
    Returns:
        float: 归一化后的轮廓系数，范围[0,1]
    """   
    # 计算轮廓系数（原始范围为[-1,1]）
    sil_score = silhouette_score(x, y)    
    # 将轮廓系数从[-1,1]映射到[0,1]
    normalized_score = (sil_score + 1) / 2
    
    return normalized_score


def neighborhood_consistency(
        x_single: pd.DataFrame, x_integrated: pd.DataFrame, 
        y: pd.Series, neighbor_frac: float = 0.01
        ) -> float:
    """计算单平台和整合后数据的邻域一致性
    Args:
        x_single: 单平台特征矩阵
        x_integrated: 整合后特征矩阵
        y: 真实样本标签
        neighbor_frac: 邻居比例
    Returns:
        float: 邻域一致性得分 [0,1]
    """
    k = max(int(len(y) * neighbor_frac), 1)
    
    # 构建KNN
    nn_single = NearestNeighbors(n_neighbors=k+1)
    nn_integrated = NearestNeighbors(n_neighbors=k+1)
    
    # 计算每个类别的NC得分
    nc_per_class = []
    for label in y.unique():
        # 获取当前类别的样本
        mask = y == label
        x_s = x_single[mask]
        x_i = x_integrated[mask]
        
        if len(x_s) <= k:
            continue
            
        # 获取邻居关系矩阵
        nn_s = nn_single.fit(x_s).kneighbors_graph(x_s)
        nn_i = nn_integrated.fit(x_i).kneighbors_graph(x_i)
        
        # 移除自身连接
        nn_s.setdiag(0)
        nn_i.setdiag(0)
        
        # 计算交集和并集
        intersection = nn_s.multiply(nn_i).sum(axis=1).A1
        union = (nn_s + nn_i).astype(bool).sum(axis=1).A1
        
        # 计算该类别的NC得分
        nc_score = np.mean(intersection / union)
        nc_per_class.append(nc_score)
    
    return np.mean(nc_per_class)


def seurat_alignment_score(x: pd.DataFrame, p: pd.Series, neighbor_frac: float = 0.01) -> float:
    """计算Seurat对齐分数，评估不同平台间的混合程度
    Args:
        x: 整合后的特征矩阵
        p: 平台标签
        neighbor_frac: 近邻比例
    Returns:
        float: SAS分数 [0,1]，越高表示混合程度越好
    """
    # 计算近邻数
    n_samples = len(p)
    k = max(int(n_samples * neighbor_frac), 1)
    n_platforms = len(p.unique())
    
    # 获取近邻
    nn = NearestNeighbors(n_neighbors=k+1)
    nn.fit(x)
    neighbors = nn.kneighbors(x, return_distance=False)[:, 1:] # 去除自身
    
    # 计算近邻中同平台的比例
    neighbor_platforms = p.values[neighbors]
    same_platform = (neighbor_platforms == p.values[:, np.newaxis])
    same_platform_ratio = same_platform.mean(axis=1)
    
    # 计算SAS分数
    expected_ratio = 1/n_platforms  # 完美混合时的期望比例
    sas = 1 - (same_platform_ratio.mean() - expected_ratio)/(1 - expected_ratio)
    
    return max(0, min(1, sas))


def gpl_asw(x: pd.DataFrame, p: pd.Series) -> float:
    """计算平台混合度分数
    Args:
        x: 整合后的特征矩阵 (DataFrame)
        p: 平台标签 (Series)
    Returns:
        float: 混合度分数，越高表示不同平台混合得越好
    """
    # 计算每个平台的轮廓系数并转换
    gpls_asw = []
    for platform in p.unique():
        # 获取当前平台的样本
        mask = p == platform
        if sum(mask) <= 1:  # 跳过样本数太少的平台
            continue
            
        # 计算该平台的轮廓系数并转换为混合度分数
        try:
            sil = silhouette_score(x, mask)
            mix_score = 1 - sil  # 先进行1减操作
            gpls_asw.append(mix_score)
        except ValueError:  # 处理可能的错误
            continue
    
    # 计算所有平台混合度分数的平均值
    final_score = np.mean(gpls_asw)
    
    return final_score


def graph_connectivity(
        x: pd.DataFrame, y: pd.Series, n_neighbors: int = 15
        ) -> float:
    """计算图连通性分数，评估不同类型样本的混合程度
    Args:
        x: 整合后的特征矩阵 (DataFrame)
        y: 样本类型标签 (Series)
        n_neighbors: 近邻数量，默认15
    Returns:
        float: GC分数 [0,1]，越高表示混合程度越好
    """
    from scipy.sparse.csgraph import connected_components
    
    # 构建KNN图
    nn = NearestNeighbors(n_neighbors=n_neighbors)
    nn.fit(x)
    adj_matrix = nn.kneighbors_graph(x)
    
    # 计算每个类型的连通性比例
    gc_scores = []
    for label in y.unique():
        # 获取当前类型的样本
        mask = y == label
        if sum(mask) <= 1:  # 跳过样本数太少的类型
            continue
            
        # 提取子图并计算连通分量
        sub_matrix = adj_matrix[mask][:, mask]
        n_components, labels = connected_components(sub_matrix, directed=False)
        
        # 计算最大连通分量的大小
        if n_components > 0:
            largest_cc = np.max(np.bincount(labels))
            # 计算该类型的连通性比例并转换为混合度分数
            gc_score = 1 - (largest_cc / sum(mask))  # 修改这里
            gc_scores.append(gc_score)
    
    # 返回所有类型GC分数的平均值
    return np.mean(gc_scores) if gc_scores else 0.0


def biology_conservation(
        x_single: pd.DataFrame,
        x_integrated: pd.DataFrame,
        y: pd.Series,
        neighbor_frac: float = 0.01
    ) -> float:
    """计算生物学保守性分数
    Args:
        x_integrated: 整合后的特征矩阵
        x_single: 单平台特征矩阵
        y: 真实标签
        neighbor_frac: 近邻比例
    Returns:
        float: 生物学保守性分数 [0,1]，越高表示生物学特征保留得越好
    """
    # 计算三个指标
    map_score = mean_average_precision(x_integrated, y, neighbor_frac)
    asw_score = avg_silhouette_width(x_integrated, y)
    nc_score = neighborhood_consistency(x_single, x_integrated, y, neighbor_frac)
    
    # 计算平均值
    bio_conservation = np.mean([map_score, asw_score, nc_score])
    
    return bio_conservation


def gpls_mixing(
        x_integrated: pd.DataFrame,
        p: pd.Series,
        neighbor_frac: float = 0.01,
        n_neighbors: int = 15
    ) -> float:
    """计算平台混合度综合指标
    Args:
        x_integrated: 整合后的特征矩阵
        p: 平台标签
        neighbor_frac: 近邻比例
        n_neighbors: 图连通性的近邻数
    Returns:
        float: 平台混合度分数 [0,1]，越高表示不同平台混合得越好
    """
    # 计算三个混合度指标
    sas_score = seurat_alignment_score(x_integrated, p, neighbor_frac)
    asw_score = gpl_asw(x_integrated, p)
    gc_score = graph_connectivity(x_integrated, p, n_neighbors)
    
    # 计算平均值
    mixing_score = np.mean([sas_score, asw_score, gc_score])
    
    return mixing_score


def main(task):
    """主函数
    Args:
        task: 项目名称，可选值：MSI、Lung_cancer_subtybes、Survival_analysis、Cancertype
    """
    df_single = pd.read_csv(f'../Dataset/{task}/processed_data.csv')
    x_single = df_single.iloc[:, :-2]
    p = df_single.iloc[:, -2]
    y = df_single.iloc[:, -1]   
    x_integrated = pd.read_csv(f'../Dataset/{task}/generated_data.csv')

    bio_conservation = biology_conservation(x_single, x_integrated, y)
    mixing_score = gpls_mixing(x_integrated, p)
    print(f'生物特征保留度: {bio_conservation:.4f}')
    print(f'平台混合度: {mixing_score:.4f}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='计算生成数据的评估指标')
    parser.add_argument('--task', type=str, default='MSI',
                      help='保存generated_data.csv的文件夹名称')
    args = parser.parse_args()
    main(args.task)
