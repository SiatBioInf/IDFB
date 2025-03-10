import pandas as pd
import torch
from models import VAE
import argparse

"""
生成消除平台差异的数据
"""

def load_data(data_path):
    """
    生成数据的函数，根据给定的路径加载数据并进行处理。
    """
    # 加载数据
    df = pd.read_csv(data_path)

    x = torch.tensor(df.iloc[:, :-2].values).float()
    p = torch.tensor(df.iloc[:, -2].values).long()
    y = torch.tensor(df.iloc[:, -1].values).long()

    return x, p, y

def load_vae_model(model_path):
    """加载VAE模型"""
    checkpoint = torch.load(model_path)
    
    # 创建模型实例
    vae = VAE(
        input_dim=checkpoint['input_dim'],
        latent_dim=checkpoint['latent_dim'],
        n_gpls=checkpoint['n_gpls']
    )
    
    # 加载模型参数
    vae.load_state_dict(checkpoint['model_state_dict'])
    return vae

def generate_recon(vae, x, p):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    """使用VAE生成样本"""
    vae.eval()
    vae = vae.to(device)
    
    with torch.no_grad():
        # 将数据移动到设备
        x = x.to(device)
        p = p.to(device)    
        # 生成样本
        generated_samples, _, _ = vae(x, p)
       
    return generated_samples.cpu().numpy()


def main(task):
    """主函数
    Args:
        task: 项目名称，如：MSI、Lung_cancer_subtybes、Survival_analysis、Cancertype
    """
    # 加载模型
    model_path = f'../saved_models/vae_model.pth'
    vae = load_vae_model(model_path)
    # 生成数据
    data_path = f'../Dataset/{task}/processed_data.csv'
    x, p, _ = load_data(data_path)   
    recon = generate_recon(vae, x, p)
    
    print(f"Generated data shape: {recon.shape}")
    # 保存生成的数据
    output_path = f'../Dataset/{task}/generated_data.csv'
    pd.DataFrame(recon).to_csv(output_path, index=False)
    print(f"Generated data saved to: {output_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='生成消除平台差异的数据')
    parser.add_argument('--task', type=str, default='MSI', help='保存processed_data.csv的文件夹名称')
    args = parser.parse_args()
    main(args.task)