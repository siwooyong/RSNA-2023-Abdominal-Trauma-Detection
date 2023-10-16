import pandas as pd
import numpy as np

import cv2
import zipfile
import os
import gc
import glob
import shutil
import time
import matplotlib.pyplot as plt
from tqdm import tqdm
from PIL import Image

import pydicom #as dicom
import nibabel as nib

import timm

import torch
import torch.nn as nn
import torch.nn.functional as F

import torchvision
from torchvision.transforms import v2
from concurrent.futures import ThreadPoolExecutor

import albumentations as A

from transformers.optimization import get_cosine_schedule_with_warmup

import numpy as np
import pandas as pd

import math
import pandas.api.types
import sklearn.metrics
from sklearn.metrics import f1_score, accuracy_score, roc_auc_score


def train_function(model,
                   optimizer,
                   scheduler,
                   loss_functions,
                   scaler,
                   loader,
                   device,
                   iters_to_accumulate):
    model.train()

    total_bowel_loss = 0.0
    total_extravasation_loss = 0.0
    total_kidney_loss = 0.0
    total_liver_loss = 0.0
    total_spleen_loss = 0.0
    total_any_injury_loss = 0.0

    total_bowel_weight = 0.0
    total_extravasation_weight = 0.0
    total_kidney_weight = 0.0
    total_liver_weight = 0.0
    total_spleen_weight = 0.0
    total_any_injury_weight = 0.0
    for bi, sample in enumerate(tqdm(loader)):
        sample = [x.to(device) for x in sample]
        video, crop_liver, crop_spleen, crop_kidney, label, bowel, extravasation, kidney, liver, spleen, any_injury, sample_weights = sample


        with torch.cuda.amp.autocast():
            bowel_output, extravasation_output, kidney_output, liver_output, spleen_output, any_injury_output = model(video, crop_liver, crop_spleen, crop_kidney, label, mode = 'test')

        bowel_loss = (loss_functions[0](bowel_output, bowel) * sample_weights[:, 0]).sum()
        extravasation_loss = (loss_functions[0](extravasation_output, extravasation) * sample_weights[:, 1]).sum()
        kidney_loss = (loss_functions[0](kidney_output, kidney) * sample_weights[:, 2]).sum()
        liver_loss = (loss_functions[0](liver_output, liver) * sample_weights[:, 3]).sum()
        spleen_loss = (loss_functions[0](spleen_output, spleen) * sample_weights[:, 4]).sum()
        any_injury_loss = (loss_functions[1](any_injury_output, any_injury) * sample_weights[:, 5]).sum()

        loss = (
            bowel_loss / sample_weights[:, 0].sum()+ \
            extravasation_loss / sample_weights[:, 1].sum()+ \
            kidney_loss / sample_weights[:, 2].sum() + \
            liver_loss / sample_weights[:, 3].sum()+ \
            spleen_loss / sample_weights[:, 4].sum()#+ \
            #any_injury_loss / sample_weights[:, 5].sum()
            )
        loss = loss / 5

        loss = loss / iters_to_accumulate

        scaler.scale(loss).backward()
        if (bi + 1) % iters_to_accumulate == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

            scheduler.step()


        total_bowel_loss += bowel_loss.detach().cpu()
        total_extravasation_loss += extravasation_loss.detach().cpu()
        total_kidney_loss += kidney_loss.detach().cpu()
        total_liver_loss += liver_loss.detach().cpu()
        total_spleen_loss += spleen_loss.detach().cpu()
        total_any_injury_loss += any_injury_loss.detach().cpu()

        total_bowel_weight += sample_weights[:, 0].sum().cpu()
        total_extravasation_weight += sample_weights[:, 1].sum().cpu()
        total_kidney_weight += sample_weights[:, 2].sum().cpu()
        total_liver_weight += sample_weights[:, 3].sum().cpu()
        total_spleen_weight += sample_weights[:, 4].sum().cpu()
        total_any_injury_weight += sample_weights[:, 5].sum().cpu()

    total_bowel_loss = total_bowel_loss / total_bowel_weight
    total_extravasation_loss = total_extravasation_loss / total_extravasation_weight
    total_kidney_loss = total_kidney_loss / total_kidney_weight
    total_liver_loss = total_liver_loss / total_liver_weight
    total_spleen_loss = total_spleen_loss / total_spleen_weight
    total_any_injury_loss  = total_any_injury_loss / total_any_injury_weight

    total_loss = (total_bowel_loss + total_extravasation_loss + total_kidney_loss + total_liver_loss + total_spleen_loss + total_any_injury_loss)/6
    return total_loss


def test_function(model,
                  loader,
                  device,
                  input_df,
                  temperature=1.0):

    test_df = input_df.copy()
    true_df = input_df.copy()
    model.eval()

    # competition metric
    bowel_healthy = []
    bowel_injury = []
    extravasation_healthy = []
    extravasation_injury = []
    kidney_healthy = []
    kidney_low = []
    kidney_high = []
    liver_healthy = []
    liver_low = []
    liver_high = []
    spleen_healthy = []
    spleen_low = []
    spleen_high = []

    # auc
    bowel_preds = []
    extravasation_preds = []
    kidney_preds = []
    liver_preds = []
    spleen_preds = []
    any_injury_preds = []

    bowel_trues = []
    extravasation_trues = []
    kidney_trues = []
    liver_trues = []
    spleen_trues = []
    any_injury_trues = []

    for bi, sample in enumerate(tqdm(loader)):
        sample = [x.to(device) for x in sample]
        video, crop_liver, crop_spleen, crop_kidney, label, bowel, extravasation, kidney, liver, spleen, any_injury, _ = sample

        with torch.no_grad():
            output = model(video, crop_liver, crop_spleen, crop_kidney, label, mode = 'test')

        bowel_output = nn.Softmax(dim=-1)(output[0].cpu()/temperature)
        extravasation_output = nn.Softmax(dim=-1)(output[1].cpu()/temperature)
        kidney_output = nn.Softmax(dim=-1)(output[2].cpu()/temperature)
        liver_output = nn.Softmax(dim=-1)(output[3].cpu()/temperature)
        spleen_output = nn.Softmax(dim=-1)(output[4].cpu()/temperature)
        any_injury_output = output[5].cpu()

        bowel_healthy.extend(bowel_output[:, 0].tolist())
        bowel_injury.extend(bowel_output[:, 1].tolist())
        extravasation_healthy.extend(extravasation_output[:, 0].tolist())
        extravasation_injury.extend(extravasation_output[:, 1].tolist())
        kidney_healthy.extend(kidney_output[:, 0].tolist())
        kidney_low.extend(kidney_output[:, 1].tolist())
        kidney_high.extend(kidney_output[:, 2].tolist())
        liver_healthy.extend(liver_output[:, 0].tolist())
        liver_low.extend(liver_output[:, 1].tolist())
        liver_high.extend(liver_output[:, 2].tolist())
        spleen_healthy.extend(spleen_output[:, 0].tolist())
        spleen_low.extend(spleen_output[:, 1].tolist())
        spleen_high.extend(spleen_output[:, 2].tolist())

        bowel_preds.extend(bowel_output[:, 1].tolist())
        extravasation_preds.extend(extravasation_output[:, 1].tolist())
        kidney_preds.extend(kidney_output.tolist())
        liver_preds.extend(liver_output.tolist())
        spleen_preds.extend(spleen_output.tolist())
        any_injury_preds.extend(any_injury_output.tolist())

        bowel_trues.extend(bowel.tolist())
        extravasation_trues.extend(extravasation.tolist())
        kidney_trues.extend(kidney.tolist())
        liver_trues.extend(liver.tolist())
        spleen_trues.extend(spleen.tolist())
        any_injury_trues.extend(any_injury.tolist())

    test_df['bowel_healthy'] = bowel_healthy
    test_df['bowel_injury'] = bowel_injury
    test_df['extravasation_healthy'] = extravasation_healthy
    test_df['extravasation_injury'] = extravasation_injury
    test_df['kidney_healthy'] = kidney_healthy
    test_df['kidney_low'] = kidney_low
    test_df['kidney_high'] = kidney_high
    test_df['liver_healthy'] = liver_healthy
    test_df['liver_low'] = liver_low
    test_df['liver_high'] = liver_high
    test_df['spleen_healthy'] = spleen_healthy
    test_df['spleen_low'] = spleen_low
    test_df['spleen_high'] = spleen_high

    test_score = score(create_training_solution(true_df), test_df, 'patient_id', reduction='none')

    bowel_auc = roc_auc_score(bowel_trues, bowel_preds)
    extravasation_auc = roc_auc_score(extravasation_trues, extravasation_preds)
    kidney_auc = roc_auc_score(kidney_trues, kidney_preds, multi_class = 'ovr')
    liver_auc = roc_auc_score(liver_trues, liver_preds, multi_class = 'ovr')
    spleen_auc = roc_auc_score(spleen_trues, spleen_preds, multi_class = 'ovr')
    any_injury_auc = roc_auc_score(any_injury_trues, any_injury_preds)

    message = {
        'weighted-log-loss' : {
            'bowel' : round(test_score[0], 4),
            'extravasation' : round(test_score[1], 4),
            'kidney' : round(test_score[2], 4),
            'liver' : round(test_score[3], 4),
            'spleen' : round(test_score[4], 4),
            'any_injury' : round(test_score[5], 4),
            'score' : round(np.mean(test_score), 4)
        },

        'auc' : {
            'bowel' : round(bowel_auc, 4),
            'extravasation' : round(extravasation_auc, 4),
            'kidney' : round(kidney_auc, 4),
            'liver' : round(liver_auc, 4),
            'spleen' : round(spleen_auc, 4),
            'any_injury' : round(any_injury_auc, 4)

        }
    }

    return test_df, torch.tensor(test_score).mean(), message


for k in range(5):
    device = 'cuda'
    epoch = 20
    batch_size = 4
    lr = 2e-4
    wd = 0.01
    warmup_ratio = 0.1
    num_workers = 12
    iters_to_accumulate = 4
    label_smoothing = 0.0
    early_stop_epoch = 15
    dir_name = 'result-channel2-512'

    train, injury_folds, normal_folds = preprocess()

    train_df = pd.concat([injury_folds[k][0]] + [normal_folds[k][0]]).reset_index(drop=True)
    val_df = pd.concat([injury_folds[k][1], normal_folds[k][1]]).reset_index(drop=True)

    train_dataset = CustomDataset(train_df, augmentation=True)
    val_dataset = CustomDataset(val_df)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size = batch_size, num_workers = num_workers, shuffle = True, drop_last = True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size = batch_size, num_workers = num_workers, shuffle = False, drop_last = False)


    model = Custom3DCNN().to(device).float()

    loss_functions = [
        nn.CrossEntropyLoss(label_smoothing = label_smoothing, reduction='none'),
        nn.BCELoss(reduction='none')
    ]

    optimizer = torch.optim.AdamW(params = model.parameters(), lr = lr, weight_decay = wd)
    total_steps = int(len(train_df) * epoch/(batch_size * iters_to_accumulate))
    warmup_steps = int(total_steps * warmup_ratio)
    print('total_steps: ', total_steps)
    print('warmup_steps: ', warmup_steps)

    scheduler = get_cosine_schedule_with_warmup(optimizer,
                                                num_warmup_steps = warmup_steps,
                                                num_training_steps = total_steps)
    scaler = torch.cuda.amp.GradScaler()

    if not os.path.isdir(f'/content/drive/MyDrive/Kaggle/RSNA 2023 Abdominal Trauma Detection/{dir_name}/'):
      os.mkdir(f'/content/drive/MyDrive/Kaggle/RSNA 2023 Abdominal Trauma Detection/{dir_name}/')

    if not os.path.isdir(f'/content/drive/MyDrive/Kaggle/RSNA 2023 Abdominal Trauma Detection/{dir_name}/fold{k+1}/'):
      os.mkdir(f'/content/drive/MyDrive/Kaggle/RSNA 2023 Abdominal Trauma Detection/{dir_name}/fold{k+1}/')

    log_path = f'/content/drive/MyDrive/Kaggle/RSNA 2023 Abdominal Trauma Detection/{dir_name}/fold{k+1}/log.txt'


    for i in range(epoch):
        print(f'{i+1}th epoch training is start...')

        if i==early_stop_epoch:
          break

        # train
        train_loss = train_function(model,
                                    optimizer,
                                    scheduler,
                                    loss_functions,
                                    scaler,
                                    train_loader,
                                    device,
                                    iters_to_accumulate)

        # val
        _, val_loss, message = test_function(model,
                                              val_loader,
                                              device,
                                              val_loader.dataset.df.copy())


        # save
        save_path = f'/content/drive/MyDrive/Kaggle/RSNA 2023 Abdominal Trauma Detection/{dir_name}/fold{k+1}/epoch' + f'{i+1}'.zfill(3) + \
                    f'-trainloss{round(train_loss.tolist(), 4)}' + \
                    f'-valloss{round(val_loss.tolist(), 4)}' + '.bin'
        torch.save(model.state_dict(), save_path)

        _lr = optimizer.param_groups[0]['lr']
        message['log'] = f'epoch : {i+1}, lr : {_lr}, trainloss : {round(train_loss.tolist(), 4)}, valloss : {round(val_loss.tolist(), 4)}'
        print(message)
        with open(log_path, 'a+') as logger:
            logger.write(f'{message}\n')
