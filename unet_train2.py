import torch
import os
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader
from unet_utils import *


def main():
    # --- 1. CONFIGURARE ---
    CURRENT_BATCH_SIZE = 6
    EPOCHS = 11  # Rulăm până la epoca 10 inclusiv
    print(f"--- Start Antrenare (Batch Size: {CURRENT_BATCH_SIZE}, Workers: 6, AMP: ON) ---")

    # --- 2. SPLIT DATE ---
    df_sequences = pd.read_csv(INFO_CSV_PATH)
    df_sequences['Sequence ID'] = df_sequences['Index']

    df_sequences['Stratum'] = df_sequences['weather'].astype(str) + '_' + df_sequences['driving_scenario'].astype(str)
    strata_counts = df_sequences['Stratum'].value_counts()
    df_safe = df_sequences[~df_sequences['Stratum'].isin(strata_counts[strata_counts < 2].index.tolist())]

    df_safe = df_safe.sample(frac=1, random_state=42).reset_index(drop=True)
    test_ids = df_safe.iloc[:1]['Sequence ID'].values
    val_ids = df_safe.iloc[1:8]['Sequence ID'].values

    all_ids = df_sequences['Sequence ID'].values
    final_train_ids = [idx for idx in all_ids if idx not in test_ids and idx not in val_ids]

    print(f"Set Antrenare: {len(final_train_ids)} video-uri | Test: {len(test_ids)} | Val: {len(val_ids)}")

    # Master DF pentru cadre
    all_data = []
    for _, row in df_sequences.iterrows():
        all_data.extend([(row['Sequence ID'], f) for f in range(row['n_images'])])
    df_master = pd.DataFrame(all_data, columns=['Sequence ID', 'Relative_Frame_Num'])

    # DataFrames pentru Train și Validare
    train_df = df_master[df_master['Sequence ID'].isin(final_train_ids)].reset_index(drop=True)
    val_df = df_master[df_master['Sequence ID'].isin(val_ids)].reset_index(drop=True)

    # DataLoader Antrenare
    train_loader = DataLoader(
        CarlaSegmentationDataset(train_df, ROOT_PATH),
        batch_size=CURRENT_BATCH_SIZE,
        shuffle=True,
        num_workers=6,
        pin_memory=True,
        persistent_workers=True
    )

    # DataLoader Validare (Adăugat pentru metrici)
    val_loader = DataLoader(
        CarlaSegmentationDataset(val_df, ROOT_PATH),
        batch_size=CURRENT_BATCH_SIZE,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    # --- 3. MODEL ȘI ÎNCĂRCARE ---
    model = UNET().to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = torch.nn.CrossEntropyLoss(weight=class_weights)
    scaler = torch.amp.GradScaler('cuda')

    ckpt_path = os.path.join(ROOT_PATH, 'unet_checkpoint.pth.tar')
    best_model_path = os.path.join(ROOT_PATH, 'unet_best_miou.pth.tar')
    old_ckpt_path = os.path.join(ROOT_PATH, 'unet_final_model_peste_noapte.pth.tar')

    start_epoch = 1
    best_miou = 0.0
    target_path = ckpt_path if os.path.exists(ckpt_path) else old_ckpt_path

    if os.path.exists(target_path):
        print(f"🔄 Se încarcă modelul anterior: {target_path}")
        ckpt = torch.load(target_path, map_location=DEVICE)
        if isinstance(ckpt, dict) and 'state_dict' in ckpt:
            model.load_state_dict(ckpt['state_dict'])
            if 'optimizer' in ckpt:
                optimizer.load_state_dict(ckpt['optimizer'])
            if 'epoch' in ckpt:
                start_epoch = ckpt['epoch'] + 1
            if 'miou' in ckpt:
                best_miou = ckpt['miou']
        else:
            model.load_state_dict(ckpt)
            print("⚠️ S-au încărcat doar greutățile.")

    # --- 4. BUCLA DE ANTRENARE ȘI VALIDARE ---
    for epoch in range(start_epoch, EPOCHS):
        print(f"\n🚀 Epoca {epoch}/{EPOCHS - 1}")

        # Pasul 1: Antrenare
        train_epoch(train_loader, model, optimizer, criterion, DEVICE, scaler)

        # Pasul 2: Validare (Apelăm funcția din unet_utils)
        # calculate_metrics returnează: mean_iou, iou_per_class, accuracy
        miou, iou_list, acc = calculate_metrics(val_loader, model, DEVICE)

        print(f"📊 Rezultate Epoca {epoch}: mIoU: {miou:.4f}, Accuracy: {acc:.4f}")

        # Pasul 3: Salvare Checkpoint
        checkpoint = {
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'epoch': epoch,
            'miou': miou,
            'accuracy': acc
        }

        # Salvare periodică
        torch.save(checkpoint, ckpt_path)

        # Salvare "Cel mai bun model"
        if miou > best_miou:
            best_miou = miou
            torch.save(checkpoint, best_model_path)
            print(f"🌟 Model nou salvat ca fiind cel mai bun! (mIoU: {best_miou:.4f})")

        # Backup la fiecare epocă (opțional)
        torch.save(checkpoint, os.path.join(ROOT_PATH, f'model_epoca_{epoch}.pth.tar'))

        print(f"✅ Epoca {epoch} terminată!")


if __name__ == '__main__':
    main()