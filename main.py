import torch
import hydra
from omegaconf import DictConfig
from torch.utils.data import DataLoader
import random
import numpy as np
from src.models.evflownet import EVFlowNet
from src.datasets import DatasetProvider
from enum import Enum, auto
from src.datasets import train_collate
from tqdm import tqdm
from pathlib import Path
from typing import Dict, Any
import os
import time
from torchvision import transforms
from torch.cuda.amp import GradScaler, autocast
import torch.nn as nn # Import the necessary module
import torch.optim as optim # Import the torch.optim module
import torchvision.transforms.functional as F

class RepresentationType(Enum):
    VOXEL = auto()
    STEPAN = auto()

def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)

def compute_epe_error(pred_flow: torch.Tensor, gt_flow: torch.Tensor):
    '''
    end-point-error (ground truthと予測値の二乗誤差)を計算
    pred_flow: torch.Tensor, Shape: torch.Size([B, 2, 480, 640]) => 予測したオプティカルフローデータ
    gt_flow: torch.Tensor, Shape: torch.Size([B, 2, 480, 640]) => 正解のオプティカルフローデータ
    '''
    epe = torch.mean(torch.mean(torch.norm(pred_flow - gt_flow, p=2, dim=1), dim=(1, 2)), dim=0)
    return epe

def save_optical_flow_to_npy(flow: torch.Tensor, file_name: str):
    '''
    optical flowをnpyファイルに保存
    flow: torch.Tensor, Shape: torch.Size([2, 480, 640]) => オプティカルフローデータ
    file_name: str => ファイル名
    '''
    np.save(f"{file_name}.npy", flow.cpu().numpy())

@hydra.main(version_base=None, config_path="configs", config_name="base")
def main(args: DictConfig):
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    '''
        ディレクトリ構造:

        data
        ├─test
        |  ├─test_city
        |  |    ├─events_left
        |  |    |   ├─events.h5
        |  |    |   └─rectify_map.h5
        |  |    └─forward_timestamps.txt
        └─train
            ├─zurich_city_11_a
            |    ├─events_left
            |    |       ├─ events.h5
            |    |       └─ rectify_map.h5
            |    ├─ flow_forward
            |    |       ├─ 000134.png
            |    |       |.....
            |    └─ forward_timestamps.txt
            ├─zurich_city_11_b
            └─zurich_city_11_c
        '''

    # ------------------
    #    Dataloader
    # ------------------
    
    #EventDataset = EventData(args.data_path, 'train')
    #EventDataLoader = torch.utils.data.DataLoader(dataset=EventDataset, batch_size=args.batch_size, shuffle=True)
    
    loader = DatasetProvider(
        dataset_path=Path(args.dataset_path),
        representation_type=RepresentationType.VOXEL,
        delta_t_ms=100,
        num_bins=4
    )
    train_set = loader.get_train_dataset()
    test_set = loader.get_test_dataset()
    collate_fn = train_collate
    train_data = DataLoader(train_set,
                                 batch_size=args.data_loader.train.batch_size,
                                 shuffle=args.data_loader.train.shuffle,
                                 collate_fn=collate_fn,
                                 drop_last=False)
    test_data = DataLoader(test_set,
                                 batch_size=args.data_loader.test.batch_size,
                                 shuffle=args.data_loader.test.shuffle,
                                 collate_fn=collate_fn,
                                 drop_last=False)

    '''
    train data:
        Type of batch: Dict
        Key: seq_name, Type: list
        Key: event_volume, Type: torch.Tensor, Shape: torch.Size([Batch, 4, 480, 640]) => イベントデータのバッチ
        Key: flow_gt, Type: torch.Tensor, Shape: torch.Size([Batch, 2, 480, 640]) => オプティカルフローデータのバッチ
        Key: flow_gt_valid_mask, Type: torch.Tensor, Shape: torch.Size([Batch, 1, 480, 640]) => オプティカルフローデータのvalid. ベースラインでは使わない

    test data:
        Type of batch: Dict
        Key: seq_name, Type: list
        Key: event_volume, Type: torch.Tensor, Shape: torch.Size([Batch, 4, 480, 640]) => イベントデータのバッチ
    '''
    # ------------------
    #       Model
    # ------------------
    model = EVFlowNet(args.train).to(device)

    # ------------------
    #   optimizer
    # ------------------
    optimizer = torch.optim.Adam(model.parameters(), lr=args.train.initial_learning_rate, weight_decay=args.train.weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, verbose=True) #add

    # ------------------
    #   Start training
    # ------------------
    #model_path = "/content/drive/MyDrive/Colab Notebooks/DLBasics2023_colab/end/EventCamera/checkpoints/model_33_20240706182716.pth"
    #model.load_state_dict(torch.load(model_path, map_location=device))   #add

    print("all epoch: {}".format(args.train.epochs))
    model.train()
    scaler = torch.cuda.amp.GradScaler(init_scale=10000)  #add

    for epoch in range(args.train.epochs):
        total_loss0 = 0
        total_loss1 = 0
        total_loss2 = 0
        loss_batches = 0

        print("on epoch: {}".format(epoch+1))

        transform = transforms.Compose([transforms.RandomHorizontalFlip(p=1.0)]) #add
        #transform2 = transforms.Compose([transforms.RandomVerticalFlip(p=1.0)]) #add
        #transform = transforms.Compose([transforms.RandomErasing(p=0.8, scale=(0.02, 0.33), ratio=(0.3, 3.3)),]) # random erasing

        for i, batch in enumerate(tqdm(train_data)):
            batch: Dict[str, Any]

            #通常画像
            event_image = batch["event_volume"].to(device) # [B, 4, 480, 640]
            ground_truth_flow = batch["flow_gt"].to(device) # [B, 2, 480, 640]
            optimizer.zero_grad()   #add
            
            flow_dict = model(event_image) # [B, 2, 480, 640]

            flow_dict['flow0'] = F.resize(img=flow_dict['flow0'], size=(480,640))
            flow_dict['flow1'] = F.resize(img=flow_dict['flow1'], size=(480,640))            
            flow_dict['flow2'] = F.resize(img=flow_dict['flow2'], size=(480,640))  
            flow = (flow_dict['flow0']*1.2+flow_dict['flow1']*2+flow_dict['flow2']*3+flow_dict['flow3']*4)/10.2 #14.159 
            loss: torch.Tensor = compute_epe_error(flow, ground_truth_flow)

            print(f"batch {i} loss1: {loss.item()}")
            scaler.scale(loss).backward() # 勾配の消失を防ぐために、GradScalerを使用して逆伝播。
            scaler.step(optimizer) # パラメータを更新
            scaler.update() # GradScalerを更新
            total_loss1 += loss.item()
            loss_batches += 1 #add

        scheduler.step(total_loss0/loss_batches) #add
        loss_batches=0

        for i, batch in enumerate(tqdm(train_data)):
            batch: Dict[str, Any]

            # Flipping HorizontalFlip
            batch['event_volume'] = transform(batch['event_volume'])#add
            event_image = batch["event_volume"].to(device) # [B, 4, 480, 640]
            ground_truth_flow = batch["flow_gt"].to(device) # [B, 2, 480, 640]
            optimizer.zero_grad()   #add
            
            flow_dict = model(event_image) # [B, 2, 480, 640]
            
            flow_dict['flow0'] = F.resize(img=flow_dict['flow0'], size=(480,640)) #中間LOSS計算用
            flow_dict['flow1'] = F.resize(img=flow_dict['flow1'], size=(480,640)) #中間LOSS計算用          
            flow_dict['flow2'] = F.resize(img=flow_dict['flow2'], size=(480,640)) #中間LOSS計算用 
            flow = (flow_dict['flow0']*1.2+flow_dict['flow1']*2+flow_dict['flow2']*3+flow_dict['flow3']*4)/10.2 #14.159 #中間LOSS計算用
            loss: torch.Tensor = compute_epe_error(flow, ground_truth_flow)

            print(f"batch {i} loss2: {loss.item()}")
            scaler.scale(loss).backward() # 勾配の消失を防ぐために、GradScalerを使用して逆伝播。
            scaler.step(optimizer) # パラメータを更新
            scaler.update() # GradScalerを更新
            total_loss2 += loss.item()
            loss_batches += 1 #add

        scheduler.step(total_loss0/loss_batches) #0711_0800
        total_loss0 = (total_loss1 + total_loss2) / 2 #0711_0800

        print(f'Epoch {epoch+1}, Loss1: {total_loss1 / len(train_data)}')
        print(f'Epoch {epoch+1}, Loss2: {total_loss2 / len(train_data)}')
        print(f'Epoch {epoch+1}, mean_Loss: {total_loss0 / len(train_data)}')

        # Create the directory if it doesn't exist
        if not os.path.exists('checkpoints'):
            os.makedirs('checkpoints')

        current_time = time.strftime("%Y%m%d%H%M%S")
        model_path = f"checkpoints/model_11_{current_time}.pth"
        torch.save(model.state_dict(), model_path)
        print(f"Model saved to {model_path}")

        # ------------------
        #   Start predicting
        # ------------------
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()
        flow: torch.Tensor = torch.tensor([]).to(device)
        with torch.no_grad():
            print("start test")
            for batch in tqdm(test_data):
                batch: Dict[str, Any]
                event_image = batch["event_volume"].to(device)
                flow_dict2 = model(event_image) # [1, 2, 480, 640]

                flow_dict2['flow0'] = F.resize(img=flow_dict2['flow0'], size=(480,640))
                flow_dict2['flow1'] = F.resize(img=flow_dict2['flow1'], size=(480,640))            
                flow_dict2['flow2'] = F.resize(img=flow_dict2['flow2'], size=(480,640))
                flow_dict3 = (flow_dict2['flow0']*1.2+flow_dict2['flow1']*2+flow_dict2['flow2']*3+flow_dict2['flow3']*4)/10.2                
                
                flow = torch.cat((flow,flow_dict3), dim=0)  # [N, 2, 480, 640]
            print("test done")
        # ------------------
        #  save submission
        # ------------------
        file_name = 'submission_11_'+'{0:03d}'.format(epoch)
        save_optical_flow_to_npy(flow, file_name)

if __name__ == "__main__":
    main()
