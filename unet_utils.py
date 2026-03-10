import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
from PIL import Image, ImageFile
import os
from tqdm import tqdm

ImageFile.LOAD_TRUNCATED_IMAGES = True

# --- CONFIGURARE ---
ROOT_PATH = 'E:/ProiectSBC/'
CLASSES_CSV_PATH = os.path.join(ROOT_PATH, 'classes_rgb_values.csv')
INFO_CSV_PATH = os.path.join(ROOT_PATH, 'video_info.csv')
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 6
NUM_CLASSES = 13
LEARNING_RATE = 1e-4

# --- PARSARE CULORI ---
def _parse_rgb(rgb_str):
    c = rgb_str.strip('[]"').replace(' ', '').replace("'", "")
    return tuple(map(int, c.split(',')))

df_classes = pd.read_csv(CLASSES_CSV_PATH)
rgb_to_id = { _parse_rgb(row['rgb_values']): row['index'] for _, row in df_classes.iterrows() if row['index'] < NUM_CLASSES }

# Ponderea claselor pentru Loss
weights_np = 1.0 / (np.array(df_classes[df_classes['index'] < NUM_CLASSES]['relative_percentile_frequency']) / 100.0)
class_weights = torch.from_numpy(weights_np / np.sum(weights_np)).float().to(DEVICE)

# --- DATASET ---
class CarlaSegmentationDataset(Dataset):
    def __init__(self, data_frame, root_dir):
        self.data_frame = data_frame
        self.root_dir = root_dir

    def __len__(self): return len(self.data_frame)

    def __getitem__(self, idx):
        row = self.data_frame.iloc[idx]
        v_id, f_num = int(row['Sequence ID']), int(row['Relative_Frame_Num'])
        
        path_img = os.path.join(self.root_dir, 'images', f"Video_{v_id:03}", f"v{v_id:03}_{f_num:04}.png")
        path_lbl = os.path.join(self.root_dir, 'labels_id', f"Video_{v_id:03}", f"v{v_id:03}_{f_num:04}.png")

        img = np.array(Image.open(path_img).convert('RGB')) / 255.0
        mask = np.array(Image.open(path_lbl)) 

        return torch.from_numpy(img).permute(2,0,1).float(), torch.from_numpy(mask).long()

# --- ARHITECTURA UNET ---
class DoubleConv(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_c, out_c, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_c), nn.ReLU(inplace=True),
            nn.Conv2d(out_c, out_c, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_c), nn.ReLU(inplace=True)
        )
    def forward(self, x): return self.conv(x)

class UNET(nn.Module):
    def __init__(self, in_c=3, out_c=13, features=[64, 128, 256, 512]):
        super().__init__()
        self.downs, self.ups = nn.ModuleList(), nn.ModuleList()
        self.pool = nn.MaxPool2d(2, 2)
        for f in features:
            self.downs.append(DoubleConv(in_c, f)); in_c = f
        for f in reversed(features):
            self.ups.append(nn.ConvTranspose2d(f*2, f, 2, 2))
            self.ups.append(DoubleConv(f*2, f))
        self.bottleneck = DoubleConv(features[-1], features[-1]*2)
        self.final_conv = nn.Conv2d(features[0], out_c, 1)

    def forward(self, x):
        skips = []
        for d in self.downs: x = d(x); skips.append(x); x = self.pool(x)
        x = self.bottleneck(x); skips = skips[::-1]
        for i in range(0, len(self.ups), 2):
            x = self.ups[i](x); skip = skips[i//2]
            if x.shape != skip.shape: x = F.interpolate(x, size=skip.shape[2:])
            x = torch.cat((skip, x), dim=1); x = self.ups[i+1](x)
        return self.final_conv(x)

# --- FUNCTII ANTRENARE ---
def train_epoch(loader, model, opt, loss_fn, dev, scaler):
    model.train()
    loop = tqdm(loader, desc="Antrenare")
    for data, target in loop:
        data, target = data.to(dev, non_blocking=True), target.to(dev, non_blocking=True)
        opt.zero_grad()
        with torch.amp.autocast('cuda'):
            pred = model(data)
            loss = loss_fn(pred, target)
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
        loop.set_postfix(loss=loss.item())

def calculate_metrics(loader, model, dev):
    model.eval()
    matrix = torch.zeros(NUM_CLASSES, NUM_CLASSES, device='cpu')
    with torch.no_grad():
        for x, y in tqdm(loader, desc="Evaluare"):
            x, y = x.to(dev), y.to(dev)
            with torch.amp.autocast('cuda'):
                preds = torch.argmax(model(x), dim=1)
            for t, p in zip(y.view(-1), preds.view(-1)):
                if t < NUM_CLASSES: matrix[t.long(), p.long()] += 1
    tp = matrix.diag(); fp = matrix.sum(0) - tp; fn = matrix.sum(1) - tp
    iou = tp / (tp + fp + fn + 1e-6)
    acc = tp.sum() / matrix.sum()
    return torch.mean(iou).item(), iou.tolist(), acc.item()
