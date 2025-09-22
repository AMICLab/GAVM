from engine import *
from datasets.base import Dataset
# from models.mamba2_light import MambaIRv2Light
from models.GAVM.GAVM import GAVM
from utils import *
from configs.test_config import config
from torch.utils import data

def main():
    # 选择设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = GAVM(
        spatial_dims=2,
        init_filters=16,
        in_channels=1,
        out_channels=1,
        blocks_down=[1, 1, 1],
        blocks_up=[1, 1, 1],
    ).cuda()

    criterion = config.criterion
    folder = [r'/home/siat/img/test.txt',r'/home/siat/label/test.txt']
    dataset = Dataset(folder, 256, augment_flip=True,
                 convert_image_to=None, condition=2, equalizeHist=False,
                 crop_patch=True)
    val_loader = data.DataLoader(dataset, batch_size=16, shuffle=False)
    global logger
    log_dir = os.path.join(config.work_dir, 'log')
    logger = get_logger('train', log_dir)
    epoch=1000

    print('#----------Testing----------#')
    best_weight = torch.load(config.work_dir + 'checkpoints/150.pth', map_location=torch.device('cuda:0'),weights_only=False)
    # model.load_state_dict(best_weight)
    model.load_state_dict(best_weight['model_state_dict'])
    loss = test_one_epoch(
        val_loader,
        model,
        criterion,
        epoch,
        logger,
        config,
    )

if __name__ == '__main__':
    main()