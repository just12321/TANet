import importlib
import os

from utils.topbreak import branch_based_invasion
os.environ['NO_ALBUMENTATIONS_UPDATE'] = '1'
from utils.losses import IOU_loss
from utils.metrics import Accuracy, Dice, Iou, Recall, mIoU
from types import ModuleType
from tqdm import tqdm
from model.tanet import TANet
from utils.Accumulator import Accumulators
from utils.dataset import CIDataset
from torch.utils.data import DataLoader
from torch.utils.tensorboard.writer import SummaryWriter
import torch
import torch.nn.functional as F
from torchvision.utils import save_image

from utils.utils import overlay_mask, record

ms = {
    'Acc': Accuracy,
    'Iou': Iou,
    'F1': Dice,
    'Recall': Recall,
    'mIoU': mIoU,
}

def metrics(model):
    metrics_acc = Accumulators()
    for idx, (img, mask) in tqdm(enumerate(val_loader)):
        model.clear_pre()
        img, mask = img.to(device), mask.to(device)
        metrics = model.metrics(img, mask, metrics=ms)
        print(idx, metrics['Iou'])
        metrics_acc.update(metrics)
    return metrics_acc.items()

def import_module_from_file(module_name: str, file_path: str) -> ModuleType:
    """
    Import a module given its file path.
    """
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

def traverse_and_import(base_dir: str, filter_func: callable = None) -> dict:
    """
    Traverse the directory and dynamically import model and args from model.py.
    """
    for root, dirs, files in os.walk(base_dir):
        if 'model.py' in files:
            # Extract model name and iteration from the path
            parts = root.split(os.sep)
            model_name = parts[-3]
            iteration = parts[-2]

            if filter_func and not filter_func(model_name, iteration):
                continue


            # Construct the full path to model.py
            model_path = os.path.join(root, 'model.py')
            
            # Dynamically import Name from model.py
            model_py_module = import_module_from_file('model.py', model_path)
            model_name_in_file = model_py_module.modelName            
            args = model_py_module.args
            # Construct the full path to the specific model file
            specific_model_path = os.path.join(root, f'{model_name_in_file}.py')
            
            if os.path.exists(specific_model_path):
                # Dynamically import model and args from the specific model file
                specific_model_module = import_module_from_file(model_name_in_file, specific_model_path)
                model = getattr(specific_model_module, model_name_in_file)
            
                print(f"Model Name: {model_name}, Iteration: {iteration}")
                net = model(**args).to(device)
                net.eval()
                test_accs = Accumulators()
                for i in range(5):
                    checkpoint_path = os.path.join(root, '..', 'checkpoint', f'fold_{i}_best.pt')
                    cp = torch.load(checkpoint_path)
                    print(cp['epoch'])
                    net.load_state_dict(cp['model'])
                    res = metrics(net)
                    test_accs.update(res)
                    print(test_accs.items())
                print('Test metrics:', test_accs.items())
                del net

path = r'./dataset/FSCAD/val'
valset = CIDataset(path, augmentator=None, size=(256, 256), is_train=False, img_type='png', pattern=lambda stem, _:stem)

val_loader = DataLoader(valset, batch_size=1, shuffle=False)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
base_directory = 'exp/KFold'
filter_func = lambda model_name, iteration: 'FSCAD' in iteration

# traverse_and_import(base_directory, filter_func)
# exit(0)

sw = SummaryWriter('exp/vision')
model = TANet(3, 1).to(device)
model.load_state_dict(torch.load(r'exp/KFold/info/TANet/FSCAD_256_256_000/checkpoint/fold_0_best.pt')['model'])
model.eval()

# Visualize the network
print("Waiting for forward...")
cam = {}
with record(model, sw, depth=9, save_cam=cam):
    img, mask = next(iter(val_loader))
    img, mask = img.to(device), mask.float().to(device)
    pred = model(img)
    print("Waiting for backward...")
    loss = IOU_loss(pred, mask)
    loss.backward()
for k, v in cam.items():
    try:
        grad_cam = F.interpolate(v, size=img.shape[2:], mode='bicubic', align_corners=False).view(*img.shape[2:])
        heat_map = overlay_mask(img.squeeze(0).permute(1, 2, 0).cpu(), grad_cam, alpha=0.5)
        sw.add_image(f"heatmap/{k}", heat_map, 0, dataformats='HWC')
    except Exception as e:
        print(f"{k} failed:{e}")
    finally:
        continue
cam.clear()
sw.close()