import os
import torch
from torch import nn
import torch.nn.functional as F

from models import VAE, Discriminator
from process_data import gen_dataloader


def get_vae_loss(inputs, outputs, z_mean, z_log_var):
    # 计算重构损失
    reconstruction_loss = F.mse_loss(outputs, inputs, reduction='mean')  
    # 计算KL散度损失
    kl_loss = -0.5 * torch.mean(1 + z_log_var - z_mean.pow(2) - z_log_var.exp())
    # 总损失为重构损失和KL散度损失之和
    total_loss = reconstruction_loss + kl_loss
    
    return total_loss


def stop_training(gen_losses, disc_losses, threshold=0, patience=5):
    if len(gen_losses) <= patience:
        return False
    else:
        gen_loss_window = gen_losses[-patience:]
        disc_loss_window = disc_losses[-patience:]
        gen_loss_change = min(gen_loss_window) - gen_loss_window[-1]
        disc_loss_change = min(disc_loss_window) - disc_loss_window[-1]
        if gen_loss_change < threshold and disc_loss_change < threshold:
            return True
        else:
            return False


# training function
def fit(dataloader, vae, disc, n_epochs, lr=0.002, 
        early_stopping=True, device='cuda'):
    criterion = nn.CrossEntropyLoss()
    vae = vae.to(device)
    vae_opt = torch.optim.Adam(vae.parameters(), lr=lr)
    disc = disc.to(device)
    disc_opt = torch.optim.Adam(disc.parameters(), lr=lr)

    disc_losses = []
    gen_losses = []
    for epoch in range(n_epochs):
        disc_running_loss = 0
        gen_running_loss = 0
        for x, p in dataloader:
            p = p.long()
            x, p = x.to(device), p.to(device)
            ### Update discriminator
            # Zero out the discriminator gradients
            disc_opt.zero_grad()
            recon, loc, std = vae(x, p)
            disc_recon_pred = disc(recon.detach(), p) # do not backpropagate through vae
            disc_x_pred = disc(x, p)
            disc_recon_loss = criterion(disc_recon_pred, torch.ones_like(disc_recon_pred)*0.25)
            disc_x_loss = criterion(disc_x_pred, p) 
            disc_loss = (disc_recon_loss + disc_x_loss)/2
            disc_loss.backward(retain_graph=True)
            disc_opt.step()
            # Keep track of the average discriminator loss
            disc_running_loss += disc_loss.item()

            ### Update generator
            # Zero out the generator gradients
            vae_opt.zero_grad()
            vae_loss = get_vae_loss(x, recon, loc, std)
            disc_recon_pred = disc(recon, p)
            gen_loss = (vae_loss + criterion(disc_recon_pred, torch.ones_like(disc_recon_pred)*0.25))/2
            gen_loss.backward()
            vae_opt.step()
            # Keep track of the generator losses
            gen_running_loss += gen_loss.item()

        disc_epoch_loss = disc_running_loss/len(dataloader)
        gen_epoch_loss = gen_running_loss/len(dataloader)

        print(f"Epoch {epoch+1}: "
              f"Discriminator Loss: {disc_epoch_loss:.4f}, "
              f"Generator Loss: {gen_epoch_loss:.4f}")
        
        disc_losses.append(disc_epoch_loss)
        gen_losses.append(gen_epoch_loss)

        if early_stopping:
            if stop_training(gen_losses, disc_losses):
                break

    return vae, disc


def validate(dataloader, vae, disc, device='cuda'):
    vae.eval()
    disc.eval()

    criterion = nn.CrossEntropyLoss()
    disc_running_loss = 0
    gen_running_loss = 0

    with torch.no_grad():

        for x, p in dataloader:
            p = p.long()
            x, p = x.to(device), p.to(device)
            recon, loc, std = vae(x, p)
            disc_recon_pred = disc(recon, p)
            disc_x_pred = disc(x, p)
            disc_recon_loss = criterion(disc_recon_pred, torch.ones_like(disc_recon_pred)*0.25)
            disc_x_loss = criterion(disc_x_pred, p) 
            disc_loss = (disc_recon_loss + disc_x_loss)/2
            disc_running_loss += disc_loss.item()

            vae_loss = get_vae_loss(x, recon, loc, std)
            gen_loss = (vae_loss + criterion(disc_recon_pred, torch.ones_like(disc_recon_pred)*0.25))/2
            gen_running_loss += gen_loss.item()

        disc_val_loss = disc_running_loss/len(dataloader)
        gen_val_loss = gen_running_loss/len(dataloader)

        print(f"Validating discriminator Loss: {disc_val_loss:.4f}, "
              f"Validating generator Loss: {gen_val_loss:.4f}")



def main():
    gpls=['GPL570', 'GPL20301', 'GPL24676']
    train_dataloader, test_dataloader = gen_dataloader(batch_size=64)

    input_dim = 19871
    latent_dim = 512
    n_gpls = len(gpls) + 1 # 增加未在训练集中出现的平台

    vae = VAE(input_dim, latent_dim, n_gpls)
    disc = Discriminator(input_dim, n_gpls)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    vae, disc = fit(train_dataloader, vae, disc, n_epochs=100, lr=0.002, 
        early_stopping=True, device=device)

    validate(test_dataloader, vae, disc, device=device)

    # 保存模型
    model_save_path = '../saved_models'
    # 如果不存在则创建目录
    os.makedirs(model_save_path, exist_ok=True)
    
    # 保存VAE模型
    torch.save({
        'model_state_dict': vae.state_dict(),
        'input_dim': input_dim,
        'latent_dim': latent_dim,
        'n_gpls': n_gpls
    }, os.path.join(model_save_path, 'vae_model.pth'))
    
    # 保存判别器模型
    torch.save({
        'model_state_dict': disc.state_dict(),
        'input_dim': input_dim,
        'n_gpls': n_gpls
    }, os.path.join(model_save_path, 'discriminator_model.pth'))


if __name__ == '__main__':
    main()    


            





