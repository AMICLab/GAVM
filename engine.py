import numpy as np
from torch import nn
from tqdm import tqdm
import torch
from torch.cuda.amp import autocast as autocast
from sklearn.metrics import confusion_matrix
from utils import save_imgs,get_grey1
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
import torch
import numpy as np

def add_gaussian_noise(image, mean=0, std=0.05):
    noise = torch.randn_like(image) * std + mean
    noisy_image = image + noise
    return torch.clamp(noisy_image, 0, 1)  # 确保值在[0,1]范围

class TVLoss(nn.Module):
    def forward(self, x):
        batch_size = x.size(0)
        h_x = x.size(2)
        w_x = x.size(3)
        count_h = (x.size(2)-1) * x.size(3)
        count_w = x.size(2) * (x.size(3)-1)
        h_tv = torch.pow((x[:,:,1:,:] - x[:,:,:h_x-1,:]), 2).sum()
        w_tv = torch.pow((x[:,:,:,1:] - x[:,:,:,:w_x-1]), 2).sum()
        return (h_tv / count_h + w_tv / count_w) / batch_size



def train_one_epoch(train_loader,
                    model,
                    criterion,
                    optimizer,
                    scheduler,
                    epoch,
                    step,
                    logger,
                    config,
                    writer):
    '''
    train model for one epoch
    '''
    # switch to train mode
    model.train()
    loss_list = []

    for iter, data in enumerate(train_loader):
        step += iter
        optimizer.zero_grad()
        images, targets,mri = data
        images, targets,mri = images.cuda(non_blocking=True).float(), targets.cuda(non_blocking=True).float(), mri.cuda(non_blocking=True).float()
        # print(f"Batch {iter + 1}:")
        # print(f"Images shape: {images.shape}")
        # print(f"Targets shape: {targets.shape}")
        out = model(x=images,y=mri)
        # print("Type of output:", type(out))
        # if isinstance(out, (list, tuple)):
        #     print("Output shapes:", [o.shape for o in out])
        # elif isinstance(out, torch.Tensor):
        #     print("Output shape:", out.shape)
        loss = criterion(out, targets)


        loss.backward()
        optimizer.step()

        loss_list.append(loss.item())

        now_lr = optimizer.state_dict()['param_groups'][0]['lr']

        writer.add_scalar('loss', loss, global_step=step)

        if iter % config.print_interval == 0:
            log_info = f'train: epoch {epoch}, iter:{iter}, loss: {np.mean(loss_list):.4f}, lr: {now_lr}'
            print(log_info)
            logger.info(log_info)
    scheduler.step()
    return step


def val_one_epoch(test_loader,
                    model,
                    criterion, 
                    epoch, 
                    logger,
                    config,
                  writer):
    # switch to evaluate mode
    model.eval()
    PSNR = []
    SSIM = []
    Label = []
    # NMSE = []
    loss_list = []
    with torch.no_grad():
        for iter, data in enumerate(test_loader):
            # print(f"Data structure: {type(data)}, Length: {len(data)}")
            img, msk, mri = data
            img, msk ,mri= img.cuda(non_blocking=True).float(), msk.cuda(non_blocking=True).float(), mri.cuda(non_blocking=True).float()
            out = model(x=img, y=mri)
            loss = criterion(out, msk)
            loss_list.append(loss.item())
            outputs_save = out.detach().cpu().numpy()
            GT_save = msk.detach().cpu().numpy()
            image_save = img.detach().cpu().numpy()
            writer.add_scalar('val_ls_in', loss, global_step=iter)
            # outputs_save = np.squeeze(outputs_save)
            # print(f"GT_save shape: {GT_save.shape}")
            # print(f"outputs_save shape: {outputs_save.shape}")
            # if epoch % config.save_img_interval == 0:
            #      save_imgs(image_save, GT_save, outputs_save, config.work_dir + 'outputs/',epoch)

            for j in range(len(outputs_save)):
                # image = get_grey1(np.squeeze(image_save[j,0,:,:]))
                Gt = get_grey1(GT_save[j, 0, :, :])
                out = get_grey1(outputs_save[j, 0, :, :])

                p_out = psnr(Gt, out)
                s_out = ssim(Gt, out)
                # nmse = nmse_cal(Gt, out)

                PSNR.append(p_out)
                SSIM.append(s_out)
                # NMSE.append(nmse)
        writer.add_scalar('val_loss', np.mean(loss_list), global_step=epoch)
        writer.add_scalar('PSNR', np.average(PSNR), global_step=epoch)
        writer.add_scalar('SSIM', np.average(SSIM), global_step=epoch)

        log_info = f'val epoch: {epoch}, loss: {np.mean(loss_list):.4f}, psnr: {np.average(PSNR)},std_psnr:{np.std(PSNR)},ssim: {np.average(SSIM)},std:{np.std(SSIM)}'
        print(log_info)
        logger.info(log_info)
    return np.mean(loss_list)




def test_one_epoch(test_loader,
                    model,
                    criterion,
                    epoch,
                    logger,
                    config):
    # switch to evaluate mode
    model.eval()
    PSNR = []
    SSIM = []
    Label = []
    # NMSE = []
    loss_list = []
    with torch.no_grad():
        for batch_idx, data in enumerate(tqdm(test_loader)):  # 使用 enumerate 获取 batch_idx
            img, msk ,mri= data
            img, msk ,mri= img.cuda(non_blocking=True).float(), msk.cuda(non_blocking=True).float(), mri.cuda(non_blocking=True).float()
            # if config.whether_save_img == False:
            #     if (img == 0).all() or (msk == 0).all():
            #         print("Detected an image or mask with all zero pixels. Skipping this batch.")
            #         continue
            out = model(x=img, y=mri)
            loss = criterion(out, msk)

            loss_list.append(loss.item())
            outputs_save = out.detach().cpu().numpy()
            GT_save = msk.detach().cpu().numpy()
            image_save = img.detach().cpu().numpy()
            # outputs_save = np.squeeze(outputs_save)
            # print(f"GT_save shape: {GT_save.shape}")
            # print(f"outputs_save shape: {outputs_save.shape}")
            if config.whether_save_img == True :
                 save_imgs(image_save, GT_save, outputs_save, config.work_dir + 'outputs/',batch_idx)

            for j in range(len(outputs_save)):
                # image = get_grey1(np.squeeze(image_save[j,0,:,:]))
                Gt = GT_save[j, 0, :, :]
                out =outputs_save[j, 0, :, :]

                p_out = psnr(Gt, out,data_range=1.0)
                s_out = ssim(Gt, out,data_range=1.0)
                # nmse = nmse_cal(Gt, out)

                PSNR.append(p_out)
                SSIM.append(s_out)
                # NMSE.append(nmse)

        log_info = f'val epoch: {epoch}, loss: {np.mean(loss_list):.4f}, psnr: {np.average(PSNR)},std_psnr:{np.std(PSNR)},ssim: {np.average(SSIM)},std:{np.std(SSIM)}'
        print(log_info)
        logger.info(log_info)
    return np.mean(loss_list)
