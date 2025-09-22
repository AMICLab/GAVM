
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from engine import *
import sys
from utils import *
from configs.config_setting_v2 import setting_config
import warnings
warnings.filterwarnings("ignore")
from datasets.base import Dataset
from models.GAVM.GAVM import GAVM

def main(config):

    print('#----------Creating logger----------#')
    sys.path.append(config.work_dir + '/')
    log_dir = os.path.join(config.work_dir, 'log')
    checkpoint_dir = os.path.join(config.work_dir, 'checkpoints')
    resume_model = os.path.join(checkpoint_dir, 'latest.pth')
    outputs = os.path.join(config.work_dir, 'outputs')
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    if not os.path.exists(outputs):
        os.makedirs(outputs)

    global logger
    logger = get_logger('train', log_dir)
    global writer
    writer = SummaryWriter(config.work_dir + 'summary')

    log_config_info(config, logger)





    print('#----------GPU init----------#')
    os.environ["CUDA_VISIBLE_DEVICES"] = config.gpu_id
    set_seed(config.seed)
    torch.cuda.empty_cache()
    print('#----------Preparing dataset----------#')

    '''Use MRI'''
    folder_train = [r'/home/siat/train.txt', r'/home/siat/label/train.txt',r'/home/siat/train.txt']
    folder_val = [r'/home/siat/val.txt', r'/home/siat/label/val.txt', r'/home/siat/mri/val.txt']
    folder_test = [r'/home/siat/test.txt', r'/home/siat/label/test.txt', r'/home/siat/mri/test.txt']
    dataset1 = Dataset(folder_train, 256, augment_flip=True,
                      convert_image_to=None, condition=2, equalizeHist=False,
                      crop_patch=True)
    train_loader = DataLoader(dataset1, batch_size=16, shuffle=True)

    dataset2 = Dataset(folder_val, 256, augment_flip=True,
                      convert_image_to=None, condition=2, equalizeHist=False,
                      crop_patch=True)
    val_loader = DataLoader(dataset2, batch_size=16, shuffle=True)

    dataset3 = Dataset(folder_test, 256, augment_flip=True,
                      convert_image_to=None, condition=2, equalizeHist=False,
                      crop_patch=True)
    test_loader = DataLoader(dataset3, batch_size=16, shuffle=True)

    print('#----------Prepareing Model----------#')
    if config.network == 'vmunet':
        model = GAVM(
            spatial_dims = 2,
            init_filters = 16,
            in_channels=1,
            out_channels=1,
            blocks_down=[1, 1, 1],
            blocks_up=[1, 1, 1],
        )

    else: raise Exception('network in not right!')
    model = model.cuda()

    # cal_params_flops(model, 256, logger)

    print('#----------Prepareing loss, opt, sch and amp----------#')
    criterion = config.criterion
    optimizer = get_optimizer(config, model)
    scheduler = get_scheduler(config, optimizer)

    print('#----------Set other params----------#')
    min_loss = 999
    start_epoch = 1
    min_epoch = 1

    if os.path.exists(resume_model):
        print('#----------Resume Model and Other params----------#')
        checkpoint = torch.load(resume_model, map_location=torch.device('cpu'))
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        saved_epoch = checkpoint['epoch']
        start_epoch += saved_epoch
        min_loss, min_epoch, loss = checkpoint['min_loss'], checkpoint['min_epoch'], checkpoint['loss']

        log_info = f'resuming model from {resume_model}. resume_epoch: {saved_epoch}, min_loss: {min_loss:.4f}, min_epoch: {min_epoch}, loss: {loss:.4f}'
        logger.info(log_info)

    step = 0
    print('#----------Training----------#')
    for epoch in range(start_epoch, config.epochs + 1):

        torch.cuda.empty_cache()

        step = train_one_epoch(
            train_loader,
            model,
            criterion,
            optimizer,
            scheduler,
            epoch,
            step,
            logger,
            config,
            writer
        )
        if epoch >=5:
            loss = val_one_epoch(
                    val_loader,
                    model,
                    criterion,
                    epoch,
                    logger,
                    config,
                    writer
                )

            if loss < min_loss:
                torch.save(model.state_dict(), os.path.join(checkpoint_dir, 'best.pth'))
                min_loss = loss
                min_epoch = epoch

            torch.save(
                {
                    'epoch': epoch,
                    'min_loss': min_loss,
                    'min_epoch': min_epoch,
                    'loss': loss,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                }, os.path.join(checkpoint_dir, f'{epoch}.pth'))

    if os.path.exists(os.path.join(checkpoint_dir, 'best.pth')):
        print('#----------Testing----------#')
        best_weight = torch.load(config.work_dir + 'checkpoints/best.pth', map_location=torch.device('cpu'))
        epoch=1000
        model.load_state_dict(best_weight)
        loss = test_one_epoch(
                test_loader,
                model,
                criterion,
                epoch,
                logger,
                config,
            )
        os.rename(
            os.path.join(checkpoint_dir, 'best.pth'),
            os.path.join(checkpoint_dir, f'best-epoch{min_epoch}-loss{min_loss:.4f}.pth')
        )

if __name__ == '__main__':
    config = setting_config
    main(config)