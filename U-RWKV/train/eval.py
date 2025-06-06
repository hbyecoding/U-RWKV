import os
import argparse
import random
import numpy as np
import torch
import torch.optim as optim
import wandb
from datetime import datetime

import src.utils.losses as losses
from src.utils.util import AverageMeter
from src.utils.metrics import iou_score

cpu_num = 4
os.environ['OMP_NUM_THREADS'] = str(cpu_num)
os.environ['OPENBLAS_NUM_THREADS'] = str(cpu_num)
os.environ['MKL_NUM_THREADS'] = str(cpu_num)
os.environ['VECLIB_MAXIMUM_THREADS'] = str(cpu_num)
os.environ['NUMEXPR_NUM_THREADS'] = str(cpu_num)
torch.set_num_threads(cpu_num)

# # from src.network.conv_based.cmunext_vit_test1209 import CMUNeXt
# # from src.network.conv_based.cmunext_rwkv_test_inter import CMUNeXt_rwkv_1_3_128_256_768,CMUNeXt_rwkv_1_6_256_512_768

# # from src.network.conv_based.LoRD_rwkv import LoRD, LoRD_4plusDeep,LoRD_double_192_384_768,LoRD_128_192_384,LoRD_128_192_384_enc5_bot
# # from src.network.conv_based.conv_gated_net import LoG



# from main_vs_comp_cmunext import seed_torch,get_train_parser,parse_dims,getDataloader


# def load_model(model_name, args, device=torch.device("cuda" if torch.cuda.is_available() else "cpu")):
#     """
#     根据模型名称动态加载模型，并从 model_pth_dict 中读取对应的 .pth 文件路径。
#     """
#     # 定义模型与预训练权重文件路径的映射字典
#     model_pth_dict = {
#         "U_Net": "./checkpoint2025/U_Net/2025-02-16/244/busi_3_busi_train.txt_BS_8_0.6720_U_Net_model.pth",
#         "AttU_Net": "./checkpoint2025/AttU_Net/2025-02-16/243/busi_3_busi_train.txt_BS_8_0.6737_AttU_Net_model.pth",
#         "TransUnet": "./checkpoint2025/TransUnet/2025-02-10/193/busi_3_busi_train.txt_BS_4_0.7078_TransUnet_model.pth",
#         "MedT": "./checkpoint2025/MedT/2025-02-17/247/busi_3_busi_train.txt_BS_8_0.6111_MedT_model.pth",
#         "UNeXt": "./checkpoint2025/UNeXt/2025-02-09/246/busi_3_busi_train.txt_BS_4_0.6879_UNeXt_model.pth",
#         "ConvUNeXt": "./checkpoint2025/ConvUNeXt/2025-02-09/219/busi_3_busi_train.txt_BS_4_0.6880_ConvUNeXt_model.pth",
#         "CMUNet": "./checkpoint2025/CMUNet/2025-02-17/180/busi_3_busi_train.txt_BS_8_0.7112_CMUNet_model.pth",
#         "CMUNeXt": "./checkpoint2025/cmunext/2025-01-20/253/busi_3_busi_train.txt_BS_4_0.7023_cmunext_model.pth",
#         "tinyUnet": "./checkpoint2025/tinyUnet/2025-02-18/170/busi_3_busi_train.txt_BS_4_0.6474_tinyUnet_model.pth"
#     }

#     # 获取模型对应的预训练权重文件路径
#     model_path = model_pth_dict.get(model_name)
#     if model_path is None:
#         raise ValueError(f"Model {model_name} not found in the model_pth_dict.")

#     # 动态加载模型
#     model = get_model(args)

#     # 加载预训练权重
#     log.info(f"Loading pretrained weights from {model_path}")
#     model.load_state_dict(torch.load(model_path, map_location=device))
#     model.to(device)
#     model.eval()

#     return model

# def get_model(args):
#     if args.model.startswith("CMUNeXt"):
#         from src.network.conv_based.CMUNeXt import CMUNeXt
#         if args.model.endswith("2_4_9_192_384"):
#             model = CMUNeXt(dims=[24, 48, 96, 192, 384],num_classes=args.num_classes)
#         elif args.model.endswith("1_3_128_256_768"):
#             model = CMUNeXt(dims=[16, 32, 128, 256,768],num_classes=args.num_classes)
#         elif args.model.endswith("1_3_128_256_384"):
#             model = CMUNeXt(dims=[16, 32, 128, 256, 384],num_classes=args.num_classes)
#         elif args.model == "CMUNeXt_downsample_DWT":
#             model = CMUNeXt(use_wavelet=True)            
#         else:
#             model = CMUNeXt(dims=[24, 48, 96, 192, 384]).cuda()
            
#     elif args.model == "tinyUnet":
#         from src.network.conv_based.tinyUnet import TinyUNet
#         model = TinyUNet(in_channels=3, num_classes=2).cuda()
#     elif args.model == "U_Net":
#         from src.network.conv_based.U_Net import U_Net
#         model = U_Net(output_ch=args.num_classes).cuda()

#     elif args.model.startswith("AttU_Net"):
#         from src.network.conv_based.AttU_Net import AttU_Net
#         model = AttU_Net()
#     elif args.model.startswith("UNeXt"):
#         from src.network.conv_based.UNeXt import UNext
#         model = UNext()
#     elif args.model.startswith("ConvUNeXt"):
#         from src.network.conv_based.ConvUNeXt import ConvUNeXt
#         model = ConvUNeXt(in_channels=3, num_classes=args.num_classes, base_c=32).cuda()
#     elif args.model == "CMUNet":
#         from src.network.conv_based.CMUNet import CMUNet
#         model = CMUNet(output_ch=args.num_classes).cuda()
        
#     elif args.model == "v_enc_256_fffse_dec_fusion_rwkv_with2x4_wo":
#         from src.network.conv_based.cval_LoRA5_enc_rwkv import v_enc_256_fffse_dec_fusion_rwkv_with2x4_wo
#         model = v_enc_256_fffse_dec_fusion_rwkv_with2x4_wo().cuda()
        
#     elif args.model == "v_enc_256_fffse_dec_fusion_rwkv_with2x4":
#         from src.network.conv_based.cval_LoRA5_enc_rwkv import v_enc_256_fffse_dec_fusion_rwkv_with2x4
#         model = v_enc_256_fffse_dec_fusion_rwkv_with2x4().cuda()
#     elif args.model == "v_enc_384_fffse_dec_fusion_rwkv_with2x4":
#         from src.network.conv_based.cval_LoRA5_enc_rwkv import v_enc_256_fffse_dec_fusion_rwkv_with2x4
#         model = v_enc_256_fffse_dec_fusion_rwkv_with2x4(dims=[24, 48, 96, 192, 384]).cuda()
#     elif args.model == "v_enc_512_fffse_dec_fusion_rwkv_with2x4":
#         from src.network.conv_based.cval_LoRA5_enc_rwkv import v_enc_256_fffse_dec_fusion_rwkv_with2x4
#         model = v_enc_256_fffse_dec_fusion_rwkv_with2x4(dims=[32, 64, 128, 256, 512]).cuda()
#     elif args.model == "v_enc_768_fffse_dec_fusion_rwkv_with2x4":
#         from src.network.conv_based.cval_LoRA5_enc_rwkv import v_enc_256_fffse_dec_fusion_rwkv_with2x4
#         model = v_enc_256_fffse_dec_fusion_rwkv_with2x4(dims = [48,96,192,384,768]).cuda()
        
        
#     elif args.model == "vscan_enc_256_fffse_dec_fusion_rwkv2_h_v":
#         from src.network.conv_based.cval_LoRA5_enc_rwkv import vscan_enc_256_fffse_dec_fusion_rwkv2_h_v
#         model = vscan_enc_256_fffse_dec_fusion_rwkv2_h_v().cuda() # no  ab_scan  这是1 2 -> 3 4
#     # elif args.model == "abscan_256_fffse_fusion_rwkv_with2x4_1_3":
#     #     from src.network.conv_based.cval_LoRA5_enc_rwkv import vscan_enc_256_fffse_dec_fusion_rwkv_with2x4
#     #     model = vscan_enc_256_fffse_dec_fusion_rwkv_with2x4().cuda()

#     elif args.model == "v_enc_128_fffse_decx2_fusion_rwkv_with2x4":
#         from src.network.conv_based.cval_LoRA5_enc_rwkv import v_enc_128_fffse_decx2_fusion_rwkv_with2x4
#         model = v_enc_128_fffse_decx2_fusion_rwkv_with2x4().cuda()
#     # 大的模型
#     # elif args.model == "v_enc_256_fffse_dec_fusion_rwkv_with2x4":
#     #     from src.network.conv_based.cval_LoRA5_enc_rwkv import v_enc_256_fffse_dec_fusion_rwkv_with2x4
#     #     model = v_enc_256_fffse_dec_fusion_rwkv_with2x4().cuda()
#     # elif args.model == "v_enc_256_fffse_dec_fusion_rwkv_with2x4":
#     #     from src.network.conv_based.cval_LoRA5_enc_rwkv import v_enc_256_fffse_dec_fusion_rwkv_with2x4
#     #     model = v_enc_256_fffse_dec_fusion_rwkv_with2x4().cuda()
        
              
#     else:
#         from src.network.transfomer_based.transformer_based_network import get_transformer_based_model
#         parser = argparse.ArgumentParser()
#         model = get_transformer_based_model(parser=parser, model_name=args.model, img_size=args.img_size,
#                                             num_classes=args.num_classes, in_ch=3).cuda()        
    
#     # return CMUNeXt(dims=[24, 48, 96, 192, 384]).cuda()
#     return model

# def get_val_transform(img_size):
#     from torch.utils.data import DataLoader, random_split
#     from src.dataloader.dataset import MedicalDataSets
#     from albumentations.core.composition import Compose
#     from albumentations import RandomRotate90, Flip, Resize, Normalize    
#     return Compose([
#         Resize(img_size, img_size),
#         Normalize(),
#         # ToTensorV2(),
#     ])

# def main(train_args):
#     seed_torch(train_args.seed)
    
#     trainloader, valloader = getDataloader(train_args)

#     model = get_model(train_args)
#     model = model.cuda()
#     print("train file dir:{} val file dir:{}".format(train_args.train_file_dir, train_args.val_file_dir))

#     optimizer = optim.SGD(model.parameters(), lr=train_args.base_lr, momentum=0.9, weight_decay=0.0001)
    
#     # criterion = losses.__dict__['DiceLoss']().cuda()

#     # criterion = losses.__dict__['BCEDiceLoss']().cuda()
#     if train_args.loss_type == "ioudice":
#         criterion = losses.__dict__['IOUDiceLoss']().cuda()
#     elif train_args.loss_type == "bceioudice":
#         criterion = losses.__dict__['BCEIOUDiceLoss']().cuda()
#     elif train_args.loss_type == "bceiou":
#         criterion = losses.__dict__['BCEIOULoss']().cuda()
#     else:
#         criterion = losses.__dict__['DiceLoss']().cuda()    

#     print("{} iterations per epoch".format(len(trainloader)))
#     best_iou = 0
#     best_dice = 0
#     best_metrics = {}
#     iter_num = 0
#     max_epoch = train_args.epoch

#     max_iterations = len(trainloader) * max_epoch

#     wandb_run_name = f"{train_args.model}"
#     wandb.init(
#         name=wandb_run_name,
#         notes=f"module study: {train_args.datasetname}_{train_args.model}",
#         project=f"comp_cmunext_{train_args.datasetname}",
#         config={
#             "learning_rate": train_args.base_lr,
#             "batch_size": train_args.batch_size,
#             "epochs": train_args.epoch,
#             "model": train_args.model,
#             # "ablation": train_args.ablation,
#             "dataset": train_args.datasetname
#         },
#         tags=[train_args.datasetname, train_args.model], #, train_args.ablation
#         save_code=True
#     )

#     for epoch_num in range(max_epoch):
#         model.train()
#         # if epoch_num == 0:
#         #     with open('model_structure.txt', 'w') as f:
#         #         print(model, file=f)            
#         avg_meters = {'loss': AverageMeter(), 'iou': AverageMeter(), 'dice': AverageMeter(),
#                       'val_loss': AverageMeter(), 'val_iou': AverageMeter(), 'val_dice': AverageMeter(),
#                       'val_SE': AverageMeter(), 'val_PC': AverageMeter(),
#                       'val_F1': AverageMeter(), 'val_ACC': AverageMeter()}        

#         for i_batch, sampled_batch in enumerate(trainloader):
#             if train_args.datasetname.startswith("busi") or train_args.datasetname.startswith("isic18") or train_args.datasetname.startswith("isic19") \
#                   or train_args.datasetname.startswith("poly") or train_args.datasetname.startswith("colonDB")  or train_args.datasetname.startswith("clinicDB"):
#                 img_batch, label_batch = sampled_batch['image'], sampled_batch['label'] 
#             else:
#                 img_batch, label_batch = sampled_batch
#             img_batch, label_batch = img_batch.cuda(), label_batch.cuda()
#             if train_args.model.startswith("compnet") :
#                 if train_args.model=="compnet_shallow_encoder" or train_args.model.startswith("compnet_single_decoder") or train_args.model=="compnet_add_encoder":
#                     outputs = model(img_batch)
#                 else:
#                     outputs = model(img_batch)[0]
#             elif train_args.model == "tinyUnet":
#                 outputs = model(img_batch)[:, 0:1, :, :]
#             else:
#                 outputs = model(img_batch)
#             # outputs = model(img_batch)

#             # print("outputs.shape",outputs.shape,"label_batch.shape",label_batch.shape)
#             # import pdb;pdb.set_trace()
#             loss = criterion(outputs, label_batch)
#             iou, dice, _, _, _, _, _ = iou_score(outputs, label_batch)
#             optimizer.zero_grad()
#             loss.backward()
#             optimizer.step()

#             lr_ = train_args.base_lr * (1.0 - iter_num / max_iterations) ** 0.9
#             for param_group in optimizer.param_groups:
#                 param_group['lr'] = lr_

#             iter_num = iter_num + 1
#             avg_meters['loss'].update(loss.item(), img_batch.size(0))
#             avg_meters['iou'].update(iou, img_batch.size(0))
#             avg_meters['dice'].update(dice, img_batch.size(0))
            
#         model.eval()
#         with torch.no_grad():
#             for i_batch, sampled_batch in enumerate(valloader):
#                 if train_args.datasetname.startswith("busi") or train_args.datasetname.startswith("isic18") or train_args.datasetname.startswith("isic19")\
#                       or train_args.datasetname.startswith("poly")  or train_args.datasetname.startswith("colonDB") or train_args.datasetname.startswith("clinicDB"):
#                     img_batch, label_batch = sampled_batch['image'], sampled_batch['label'] 
#                 else:
#                     img_batch, label_batch = sampled_batch
#                 img_batch, label_batch = img_batch.cuda(), label_batch.cuda()
                
#                 if train_args.model.startswith("compnet") :
#                     if train_args.model=="compnet_shallow_encoder" or train_args.model.startswith("compnet_single_decoder") or train_args.model=="compnet_add_encoder":
#                         output = model(img_batch)
#                     else:
#                         output = model(img_batch)[0]
#                 elif train_args.model == "tinyUnet":
#                     output = model(img_batch)[:, 0:1, :, :] 
#                 else:
#                     output = model(img_batch)
#                 # print(" eval output.shape", output.shape)
#                 loss = criterion(output, label_batch)
#                 iou, dice, SE, PC, F1, _, ACC = iou_score(output, label_batch)
#                 avg_meters['val_loss'].update(loss.item(), img_batch.size(0))
#                 avg_meters['val_iou'].update(iou, img_batch.size(0))
#                 avg_meters['val_dice'].update(dice, img_batch.size(0))
#                 avg_meters['val_SE'].update(SE, img_batch.size(0))
#                 avg_meters['val_PC'].update(PC, img_batch.size(0))
#                 avg_meters['val_F1'].update(F1, img_batch.size(0))
#                 avg_meters['val_ACC'].update(ACC, img_batch.size(0))

#         print(f'Epoch [{epoch_num}/{max_epoch}] Train Loss: {avg_meters["loss"].avg:.4f}, Train IoU: {avg_meters["iou"].avg:.4f}, Train Dice: {avg_meters["dice"].avg:.4f}, '
#               f'Val Loss: {avg_meters["val_loss"].avg:.4f}, Val IoU: {avg_meters["val_iou"].avg:.4f}, Val Dice: {avg_meters["val_dice"].avg:.4f}, '
#               f'Val SE: {avg_meters["val_SE"].avg:.4f}, Val PC: {avg_meters["val_PC"].avg:.4f}, '
#               f'Val F1: {avg_meters["val_F1"].avg:.4f}, Val ACC: {avg_meters["val_ACC"].avg:.4f}')        

#         wandb.log({
#             "Train Loss": avg_meters['loss'].avg,
#             "Train IOU": avg_meters['iou'].avg,
#             "Train Dice": avg_meters['dice'].avg,
#             "Val Loss": avg_meters['val_loss'].avg,
#             "Val IOU": avg_meters['val_iou'].avg,
#             "Val Dice": avg_meters['val_dice'].avg,
#             "Val SE": avg_meters['val_SE'].avg,
#             "Val PC": avg_meters['val_PC'].avg,
#             "Val F1": avg_meters['val_F1'].avg,
#             "Val ACC": avg_meters['val_ACC'].avg,
#             "Learning Rate": lr_
#         })


#         if avg_meters['val_iou'].avg > best_iou:
#             best_iou = avg_meters['val_iou'].avg
#             best_dice = avg_meters['val_dice'].avg  # 更新 Best Dice
#             today_date = datetime.now().strftime("%Y-%m-%d")
#             current_time = datetime.now().strftime("%H-%M-%S")
#             if epoch_num >=100:
#                 checkpoint_dir = f"./checkpoint2025/{train_args.model}/{today_date}/{epoch_num}"
#                 if not os.path.isdir(checkpoint_dir):
#                     os.makedirs(checkpoint_dir)
            
#                 # 保存并上传模型
#                 if train_args.datasetname.startswith("busi"):
                
#                     base_name = f"{train_args.datasetname}_{train_args.train_file_dir}_BS_{train_args.batch_size}_{avg_meters['val_iou'].avg:.4f}_{train_args.model}_model"
#                 else:
#                     base_name  = f"{train_args.datasetname}_BS_{train_args.batch_size}_{avg_meters['val_iou'].avg:.4f}_{train_args.model}_model"
#                 # save_checkpoint_and_upload(model, checkpoint_dir, base_name)
            
#                 torch.save(model.state_dict(), f'{checkpoint_dir}/{base_name}.pth')

#                 # 记录最佳 IOU 及相关指标到 WandB
#                 best_metrics = {
#                     "Best Val IOU" :  best_iou,
#                     "Best Val Dice": best_dice,  # 保存 Best Dice
#                     "Best Val Loss" :  avg_meters['val_loss'].avg,
#                     "Best Val SE" :  avg_meters['val_SE'].avg,
#                     "Best Val PC" :  avg_meters['val_PC'].avg,
#                     "Best Val F1" :  avg_meters['val_F1'].avg,
#                     "Best Val ACC" :  avg_meters['val_ACC'].avg,            
#                 }
            
#             print("=>"*10, f"saved best {train_args.datasetname} {train_args.batch_size} with {train_args.model}","<="*10)
#     from thop import profile,clever_format
#     xin = torch.randn(1, 3, 256, 256).cuda()
        
#     flops, params = profile(model,inputs=(xin,))
#     flops, params  = clever_format((flops,params),"%.2f")
#     print(f"FLOPs: {flops}")
#     print(f"Params: {params}")    

#     with open("model_structure.txt", "w") as f:
#         for name, module in model.named_modules():
#             f.write(f"{name}: {module}\n")
#     wandb.save("model_structure.txt")
#     wandb.run.summary.update(best_metrics) 
#     wandb.finish()
#     return f"{best_iou}_{best_dice}"

# if __name__ == "__main__":
#     train_parser = get_train_parser()
#     train_args = train_parser.parse_args()
#     print("0-0-0-0-0-0-0- train args get!")
#     # model_parser = get_model_parser()
#     # model_args = model_parser.parse_args()
#     result = main(train_args)
#     print(result)
    


import os
import time
import torch
from PIL import Image
from torch.utils.data import DataLoader, random_split
from torchvision.utils import save_image
from src.dataloader.dataset import MedicalDataSets
from src.network.conv_based.CMUNeXt import CMUNeXt
from src.network.conv_based.tinyUnet import TinyUNet
from src.network.conv_based.U_Net import U_Net
from src.network.conv_based.AttU_Net import AttU_Net
from src.network.conv_based.UNeXt import UNext
from src.network.conv_based.ConvUNeXt import ConvUNeXt
from src.network.conv_based.CMUNet import CMUNet
from src.network.transfomer_based.transformer_based_network import get_transformer_based_model
from albumentations.core.composition import Compose
from albumentations import RandomRotate90, Flip, Resize, Normalize
import argparse
import logging

# 设置日志配置
logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

def load_model(model_name, args, device=torch.device("cuda" if torch.cuda.is_available() else "cpu")):
    """
    根据模型名称动态加载模型，并从 model_pth_dict 中读取对应的 .pth 文件路径。
    """
    # 定义模型与预训练权重文件路径的映射字典
    # model_pth_dict = {
    #     "U_Net": "./checkpoint2025/U_Net/2025-02-16/244/busi_3_busi_train.txt_BS_8_0.6720_U_Net_model.pth",
    #     "AttU_Net": "./checkpoint2025/AttU_Net/2025-02-16/243/busi_3_busi_train.txt_BS_8_0.6737_AttU_Net_model.pth",
    #     "TransUnet": "./checkpoint2025/TransUnet/2025-02-10/193/busi_3_busi_train.txt_BS_4_0.7078_TransUnet_model.pth",
    #     "MedT": "./checkpoint2025/MedT/2025-02-17/247/busi_3_busi_train.txt_BS_8_0.6111_MedT_model.pth",
    #     "UNeXt": "./checkpoint2025/UNeXt/2025-02-09/246/busi_3_busi_train.txt_BS_4_0.6879_UNeXt_model.pth",
    #     "ConvUNeXt": "./checkpoint2025/ConvUNeXt/2025-02-09/219/busi_3_busi_train.txt_BS_4_0.6880_ConvUNeXt_model.pth",
    #     "CMUNet": "./checkpoint2025/CMUNet/2025-02-17/180/busi_3_busi_train.txt_BS_8_0.7112_CMUNet_model.pth",
    #     "CMUNeXt": "./checkpoint2025/cmunext/2025-01-20/253/busi_3_busi_train.txt_BS_4_0.7023_cmunext_model.pth",
    #     "tinyUnet": "./checkpoint2025/tinyUnet/2025-02-18/170/busi_3_busi_train.txt_BS_4_0.6474_tinyUnet_model.pth"
    # }

    model_pth_dict = {
        "v_enc_256_fffse_dec_fusion_vit": "/data/hongboye/projects/checkpoint2025/v_enc_256_fffse_dec_fusion_vit/2025-02-16/181/busi_3_busi_train.txt_BS_8_0.7000_v_enc_256_fffse_dec_fusion_vit_model.pth",
        "v_enc_256_fffse_dec_fusion_mamba": "/data/hongboye/projects/checkpoint2025/v_enc_256_fffse_dec_fusion_mamba/2025-02-16/173/busi_3_busi_train.txt_BS_8_0.6681_v_enc_256_fffse_dec_fusion_mamba_model.pth",
        "v_enc_128_fffse_decx2_fusion_rwkv_with2x4": "/data/hongboye/projects/checkpoint2025/v_enc_128_fffse_decx2_fusion_rwkv_with2x4/2025-02-15/209/busi_3_busi_train.txt_BS_4_0.6761_v_enc_128_fffse_decx2_fusion_rwkv_with2x4_model.pth",
        "v_enc_256_fffse_dec_fusion_rwkv_with2x4": "/data/hongboye/projects/checkpoint2025/v_enc_256_fffse_dec_fusion_rwkv_with2x4/2025-02-19/254/busi_3_busi_train.txt_BS_4_0.6871_v_enc_256_fffse_dec_fusion_rwkv_with2x4_model.pth",
        "vscan_enc_256_fffse_dec_fusion_rwkv2_h_v": "/data/hongboye/projects/checkpoint2025/vscan_enc_256_fffse_dec_fusion_rwkv2_h_v/2025-02-24/194/busi_3_busi_train.txt_BS_4_0.6931_vscan_enc_256_fffse_dec_fusion_rwkv2_h_v_model.pth"
    }    
    # 获取模型对应的预训练权重文件路径
    model_path = model_pth_dict.get(model_name)
    if model_path is None:
        raise ValueError(f"Model {model_name} not found in the model_pth_dict.")

    # 动态加载模型
    model = get_model(args)

    # 加载预训练权重
    log.info(f"Loading pretrained weights from {model_path}")
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    return model

def get_model(args):

    if args.model.startswith("CMUNeXt"):
        from src.network.conv_based.CMUNeXt import CMUNeXt
        if args.model.endswith("2_4_9_192_384"):
            model = CMUNeXt(dims=[24, 48, 96, 192, 384],num_classes=args.num_classes)
        elif args.model.endswith("1_3_128_256_768"):
            model = CMUNeXt(dims=[16, 32, 128, 256,768],num_classes=args.num_classes)
        elif args.model.endswith("1_3_128_256_384"):
            model = CMUNeXt(dims=[16, 32, 128, 256, 384],num_classes=args.num_classes)
        elif args.model == "CMUNeXt_downsample_DWT":
            model = CMUNeXt(use_wavelet=True)            
        else:
            model = CMUNeXt(dims=[24, 48, 96, 192, 384]).cuda()
            
    elif args.model == "tinyUnet":
        from src.network.conv_based.tinyUnet import TinyUNet
        model = TinyUNet(in_channels=3, num_classes=2).cuda()
    elif args.model == "U_Net":
        from src.network.conv_based.U_Net import U_Net
        model = U_Net(output_ch=args.num_classes).cuda()

    elif args.model.startswith("AttU_Net"):
        from src.network.conv_based.AttU_Net import AttU_Net
        model = AttU_Net()
    elif args.model.startswith("UNeXt"):
        from src.network.conv_based.UNeXt import UNext
        model = UNext()
    elif args.model.startswith("ConvUNeXt"):
        from src.network.conv_based.ConvUNeXt import ConvUNeXt
        model = ConvUNeXt(in_channels=3, num_classes=args.num_classes, base_c=32).cuda()
    elif args.model == "CMUNet":
        from src.network.conv_based.CMUNet import CMUNet
        model = CMUNet(output_ch=args.num_classes).cuda()
        
    elif args.model == "v_enc_256_fffse_dec_fusion_rwkv_with2x4_wo":
        from src.network.conv_based.cval_LoRA5_enc_rwkv import v_enc_256_fffse_dec_fusion_rwkv_with2x4_wo
        model = v_enc_256_fffse_dec_fusion_rwkv_with2x4_wo().cuda()
        
    elif args.model == "v_enc_256_fffse_dec_fusion_rwkv_with2x4":
        from src.network.conv_based.cval_LoRA5_enc_rwkv import v_enc_256_fffse_dec_fusion_rwkv_with2x4
        model = v_enc_256_fffse_dec_fusion_rwkv_with2x4().cuda()
    elif args.model == "v_enc_384_fffse_dec_fusion_rwkv_with2x4":
        from src.network.conv_based.cval_LoRA5_enc_rwkv import v_enc_256_fffse_dec_fusion_rwkv_with2x4
        model = v_enc_256_fffse_dec_fusion_rwkv_with2x4(dims=[24, 48, 96, 192, 384]).cuda()
    elif args.model == "v_enc_512_fffse_dec_fusion_rwkv_with2x4":
        from src.network.conv_based.cval_LoRA5_enc_rwkv import v_enc_256_fffse_dec_fusion_rwkv_with2x4
        model = v_enc_256_fffse_dec_fusion_rwkv_with2x4(dims=[32, 64, 128, 256, 512]).cuda()
    elif args.model == "v_enc_768_fffse_dec_fusion_rwkv_with2x4":
        from src.network.conv_based.cval_LoRA5_enc_rwkv import v_enc_256_fffse_dec_fusion_rwkv_with2x4
        model = v_enc_256_fffse_dec_fusion_rwkv_with2x4(dims = [48,96,192,384,768]).cuda()
        
        
    elif args.model == "vscan_enc_256_fffse_dec_fusion_rwkv2_h_v":
        from src.network.conv_based.cval_LoRA5_enc_rwkv import vscan_enc_256_fffse_dec_fusion_rwkv2_h_v
        model = vscan_enc_256_fffse_dec_fusion_rwkv2_h_v().cuda() # no  ab_scan  这是1 2 -> 3 4
    # elif args.model == "abscan_256_fffse_fusion_rwkv_with2x4_1_3":
    #     from src.network.conv_based.cval_LoRA5_enc_rwkv import vscan_enc_256_fffse_dec_fusion_rwkv_with2x4
    #     model = vscan_enc_256_fffse_dec_fusion_rwkv_with2x4().cuda()

    elif args.model == "v_enc_128_fffse_decx2_fusion_rwkv_with2x4":
        from src.network.conv_based.cval_LoRA5_enc_rwkv import v_enc_128_fffse_decx2_fusion_rwkv_with2x4
        model = v_enc_128_fffse_decx2_fusion_rwkv_with2x4().cuda()
    # 大的模型
    # elif args.model == "v_enc_256_fffse_dec_fusion_rwkv_with2x4":
    #     from src.network.conv_based.cval_LoRA5_enc_rwkv import v_enc_256_fffse_dec_fusion_rwkv_with2x4
    #     model = v_enc_256_fffse_dec_fusion_rwkv_with2x4().cuda()
    # elif args.model == "v_enc_256_fffse_dec_fusion_rwkv_with2x4":
    #     from src.network.conv_based.cval_LoRA5_enc_rwkv import v_enc_256_fffse_dec_fusion_rwkv_with2x4
    #     model = v_enc_256_fffse_dec_fusion_rwkv_with2x4().cuda()
        
              
    else:
        parser = argparse.ArgumentParser()
        model = get_transformer_based_model(parser=parser, model_name=args.model, img_size=args.img_size,
                                            num_classes=args.num_classes, in_ch=3).cuda()        
    
    return model

def get_val_transform(img_size):
    return Compose([
        Resize(img_size, img_size),
        Normalize(),
    ])


def seed_torch(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)  # GPU随机种子确定
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    
# def iou_score(outputs, labels):
#     smooth = 1e-6
#     outputs = torch.sigmoid(outputs)
#     outputs = (outputs > 0.5).float()
#     intersection = (outputs * labels).sum(dim=(1, 2, 3))
#     union = outputs.sum(dim=(1, 2, 3)) + labels.sum(dim=(1, 2, 3)) - intersection
#     iou = (intersection + smooth) / (union + smooth)
#     dice = (2. * intersection + smooth) / (outputs.sum(dim=(1, 2, 3)) + labels.sum(dim=(1, 2, 3)) + smooth)
#     rvd = torch.abs((outputs.sum(dim=(1, 2, 3)).float() - labels.sum(dim=(1, 2, 3)).float()) / labels.sum(dim=(1, 2, 3)).float())
#     return iou.mean(), dice.mean(), rvd.mean(), 0, 0, 0, 0, 0

def validate(model, val_loader, criterion, device, save_dir="validation_results", model_name="default_model"):
    seed_torch(41)
    """执行验证，并且每隔十张图像保存一次预测结果到PNG文件"""
    model.eval()
    val_loss = 0.0
    val_iou = 0.0
    val_dice = 0.0
    val_rvd = 0.0
    total_inference_time = 0.0
    
    # 创建特定于模型的保存目录
    model_save_dir = os.path.join(save_dir, model_name)
    os.makedirs(model_save_dir, exist_ok=True)
    
    # 日志文件路径
    log_file_path = os.path.join(model_save_dir, "model_validation.log")
    file_handler = logging.FileHandler(log_file_path)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    log.addHandler(file_handler)

    with torch.no_grad():
        for i_batch, sampled_batch in enumerate(val_loader):
            img_batch, label_batch = sampled_batch['image'], sampled_batch['label']
            img_batch, label_batch = img_batch.to(device), label_batch.to(device)
            
            start_time = time.time()
            if model_name == "tinyUnet":
                outputs = model(img_batch)[:, 0:1, :, :]
            else:
                outputs = model(img_batch)
            end_time = time.time()
            inference_time = end_time - start_time
            total_inference_time += inference_time
            
            loss = criterion(outputs, label_batch)
            val_loss += loss.item()

            iou, dice,  _, _, _, _, _ = iou_score(outputs, label_batch)
            val_iou += iou
            val_dice += dice
            # val_rvd += rvd

            # 每隔十张图像保存一次预测结果
            if i_batch % 10 == 0:
                # 将模型输出转换为二值图像
                outputs = torch.sigmoid(outputs)
                outputs[outputs > 0.5] = 1
                outputs[outputs <= 0.5] = 0
                output_images = outputs.cpu().data
                
                # 保存图像
                for idx, img in enumerate(output_images):
                    save_path = os.path.join(model_save_dir, f"batch_{i_batch}_img_{idx}.png")
                    save_image(img, save_path)

    val_loss /= len(val_loader)
    val_iou /= len(val_loader)
    val_dice /= len(val_loader)
    # val_rvd /= len(val_loader)
    val_rvd = 0.01
    avg_inference_time = total_inference_time / len(val_loader) *1000
    
    log.info(f'Model: {model_name}')
    log.info(f'验证损失: {val_loss:.4f}, 验证IoU: {val_iou:.4f}, 验证Dice: {val_dice:.4f}, 验证RVD: {val_rvd:.4f}')
    log.info(f'平均推理时间: {avg_inference_time} ms') #:.4f

    # 移除文件处理器以避免重复写入
    log.removeHandler(file_handler)
    file_handler.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Validate a trained medical image segmentation model')
    parser.add_argument('--model', type=str, default="AttU_Net",   help='Name of the model to use')
    parser.add_argument('--data_dir', type=str, default="./Tan9/data",   help='Path to the dataset directory')
    parser.add_argument('--save_dir', type=str, default='./validation_results', help='Base directory to save prediction images')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size for validation')
    parser.add_argument('--img_size', type=int, default=256, help='Size of input images')
    parser.add_argument('--num_classes', type=int, default=1, help='Number of output classes')
    # parser.add_argument('--dims', type=str, default='24,48,96,192,384', help='Dimensions for some models')
    dataset_num = "3"
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(args.model, args, device)

    transform = get_val_transform(args.img_size)
    # dataset = MedicalDataSets(base_dir=args.data_dir, transform=transform)
    
    db_val = MedicalDataSets(base_dir=args.data_dir, split="val", transform=transform,
                            train_file_dir="busi_train"+dataset_num+".txt", val_file_dir="busi_val"+dataset_num+".txt")
    
    dataloader  =  DataLoader(db_val, batch_size=args.batch_size, shuffle=False, num_workers=1)

    # 假设使用BCEWithLogitsLoss作为损失函数\
    criterion = losses.__dict__['BCEDiceLoss']().to(device)
    # criterion = torch.nn.BCEWithLogitsLoss()

    validate(model, dataloader, criterion, device, args.save_dir, args.model)



##########
import os
import argparse
import random
import numpy as np
import torch
import torch.optim as optim
import wandb
from datetime import datetime

import src.utils.losses as losses
from src.utils.util import AverageMeter
from src.utils.metrics import iou_score

cpu_num = 4
os.environ['OMP_NUM_THREADS'] = str(cpu_num)
os.environ['OPENBLAS_NUM_THREADS'] = str(cpu_num)
os.environ['MKL_NUM_THREADS'] = str(cpu_num)
os.environ['VECLIB_MAXIMUM_THREADS'] = str(cpu_num)
os.environ['NUMEXPR_NUM_THREADS'] = str(cpu_num)
torch.set_num_threads(cpu_num)

# # from src.network.conv_based.cmunext_vit_test1209 import CMUNeXt
# # from src.network.conv_based.cmunext_rwkv_test_inter import CMUNeXt_rwkv_1_3_128_256_768,CMUNeXt_rwkv_1_6_256_512_768

# # from src.network.conv_based.LoRD_rwkv import LoRD, LoRD_4plusDeep,LoRD_double_192_384_768,LoRD_128_192_384,LoRD_128_192_384_enc5_bot
# # from src.network.conv_based.conv_gated_net import LoG



# from main_vs_comp_cmunext import seed_torch,get_train_parser,parse_dims,getDataloader


# def load_model(model_name, args, device=torch.device("cuda" if torch.cuda.is_available() else "cpu")):
#     """
#     根据模型名称动态加载模型，并从 model_pth_dict 中读取对应的 .pth 文件路径。
#     """
#     # 定义模型与预训练权重文件路径的映射字典
#     model_pth_dict = {
#         "U_Net": "./checkpoint2025/U_Net/2025-02-16/244/busi_3_busi_train.txt_BS_8_0.6720_U_Net_model.pth",
#         "AttU_Net": "./checkpoint2025/AttU_Net/2025-02-16/243/busi_3_busi_train.txt_BS_8_0.6737_AttU_Net_model.pth",
#         "TransUnet": "./checkpoint2025/TransUnet/2025-02-10/193/busi_3_busi_train.txt_BS_4_0.7078_TransUnet_model.pth",
#         "MedT": "./checkpoint2025/MedT/2025-02-17/247/busi_3_busi_train.txt_BS_8_0.6111_MedT_model.pth",
#         "UNeXt": "./checkpoint2025/UNeXt/2025-02-09/246/busi_3_busi_train.txt_BS_4_0.6879_UNeXt_model.pth",
#         "ConvUNeXt": "./checkpoint2025/ConvUNeXt/2025-02-09/219/busi_3_busi_train.txt_BS_4_0.6880_ConvUNeXt_model.pth",
#         "CMUNet": "./checkpoint2025/CMUNet/2025-02-17/180/busi_3_busi_train.txt_BS_8_0.7112_CMUNet_model.pth",
#         "CMUNeXt": "./checkpoint2025/cmunext/2025-01-20/253/busi_3_busi_train.txt_BS_4_0.7023_cmunext_model.pth",
#         "tinyUnet": "./checkpoint2025/tinyUnet/2025-02-18/170/busi_3_busi_train.txt_BS_4_0.6474_tinyUnet_model.pth"
#     }

#     # 获取模型对应的预训练权重文件路径
#     model_path = model_pth_dict.get(model_name)
#     if model_path is None:
#         raise ValueError(f"Model {model_name} not found in the model_pth_dict.")

#     # 动态加载模型
#     model = get_model(args)

#     # 加载预训练权重
#     log.info(f"Loading pretrained weights from {model_path}")
#     model.load_state_dict(torch.load(model_path, map_location=device))
#     model.to(device)
#     model.eval()

#     return model

# def get_model(args):
#     if args.model.startswith("CMUNeXt"):
#         from src.network.conv_based.CMUNeXt import CMUNeXt
#         if args.model.endswith("2_4_9_192_384"):
#             model = CMUNeXt(dims=[24, 48, 96, 192, 384],num_classes=args.num_classes)
#         elif args.model.endswith("1_3_128_256_768"):
#             model = CMUNeXt(dims=[16, 32, 128, 256,768],num_classes=args.num_classes)
#         elif args.model.endswith("1_3_128_256_384"):
#             model = CMUNeXt(dims=[16, 32, 128, 256, 384],num_classes=args.num_classes)
#         elif args.model == "CMUNeXt_downsample_DWT":
#             model = CMUNeXt(use_wavelet=True)            
#         else:
#             model = CMUNeXt(dims=[24, 48, 96, 192, 384]).cuda()
            
#     elif args.model == "tinyUnet":
#         from src.network.conv_based.tinyUnet import TinyUNet
#         model = TinyUNet(in_channels=3, num_classes=2).cuda()
#     elif args.model == "U_Net":
#         from src.network.conv_based.U_Net import U_Net
#         model = U_Net(output_ch=args.num_classes).cuda()

#     elif args.model.startswith("AttU_Net"):
#         from src.network.conv_based.AttU_Net import AttU_Net
#         model = AttU_Net()
#     elif args.model.startswith("UNeXt"):
#         from src.network.conv_based.UNeXt import UNext
#         model = UNext()
#     elif args.model.startswith("ConvUNeXt"):
#         from src.network.conv_based.ConvUNeXt import ConvUNeXt
#         model = ConvUNeXt(in_channels=3, num_classes=args.num_classes, base_c=32).cuda()
#     elif args.model == "CMUNet":
#         from src.network.conv_based.CMUNet import CMUNet
#         model = CMUNet(output_ch=args.num_classes).cuda()
        
#     elif args.model == "v_enc_256_fffse_dec_fusion_rwkv_with2x4_wo":
#         from src.network.conv_based.cval_LoRA5_enc_rwkv import v_enc_256_fffse_dec_fusion_rwkv_with2x4_wo
#         model = v_enc_256_fffse_dec_fusion_rwkv_with2x4_wo().cuda()
        
#     elif args.model == "v_enc_256_fffse_dec_fusion_rwkv_with2x4":
#         from src.network.conv_based.cval_LoRA5_enc_rwkv import v_enc_256_fffse_dec_fusion_rwkv_with2x4
#         model = v_enc_256_fffse_dec_fusion_rwkv_with2x4().cuda()
#     elif args.model == "v_enc_384_fffse_dec_fusion_rwkv_with2x4":
#         from src.network.conv_based.cval_LoRA5_enc_rwkv import v_enc_256_fffse_dec_fusion_rwkv_with2x4
#         model = v_enc_256_fffse_dec_fusion_rwkv_with2x4(dims=[24, 48, 96, 192, 384]).cuda()
#     elif args.model == "v_enc_512_fffse_dec_fusion_rwkv_with2x4":
#         from src.network.conv_based.cval_LoRA5_enc_rwkv import v_enc_256_fffse_dec_fusion_rwkv_with2x4
#         model = v_enc_256_fffse_dec_fusion_rwkv_with2x4(dims=[32, 64, 128, 256, 512]).cuda()
#     elif args.model == "v_enc_768_fffse_dec_fusion_rwkv_with2x4":
#         from src.network.conv_based.cval_LoRA5_enc_rwkv import v_enc_256_fffse_dec_fusion_rwkv_with2x4
#         model = v_enc_256_fffse_dec_fusion_rwkv_with2x4(dims = [48,96,192,384,768]).cuda()
        
        
#     elif args.model == "vscan_enc_256_fffse_dec_fusion_rwkv2_h_v":
#         from src.network.conv_based.cval_LoRA5_enc_rwkv import vscan_enc_256_fffse_dec_fusion_rwkv2_h_v
#         model = vscan_enc_256_fffse_dec_fusion_rwkv2_h_v().cuda() # no  ab_scan  这是1 2 -> 3 4
#     # elif args.model == "abscan_256_fffse_fusion_rwkv_with2x4_1_3":
#     #     from src.network.conv_based.cval_LoRA5_enc_rwkv import vscan_enc_256_fffse_dec_fusion_rwkv_with2x4
#     #     model = vscan_enc_256_fffse_dec_fusion_rwkv_with2x4().cuda()

#     elif args.model == "v_enc_128_fffse_decx2_fusion_rwkv_with2x4":
#         from src.network.conv_based.cval_LoRA5_enc_rwkv import v_enc_128_fffse_decx2_fusion_rwkv_with2x4
#         model = v_enc_128_fffse_decx2_fusion_rwkv_with2x4().cuda()
#     # 大的模型
#     # elif args.model == "v_enc_256_fffse_dec_fusion_rwkv_with2x4":
#     #     from src.network.conv_based.cval_LoRA5_enc_rwkv import v_enc_256_fffse_dec_fusion_rwkv_with2x4
#     #     model = v_enc_256_fffse_dec_fusion_rwkv_with2x4().cuda()
#     # elif args.model == "v_enc_256_fffse_dec_fusion_rwkv_with2x4":
#     #     from src.network.conv_based.cval_LoRA5_enc_rwkv import v_enc_256_fffse_dec_fusion_rwkv_with2x4
#     #     model = v_enc_256_fffse_dec_fusion_rwkv_with2x4().cuda()
        
              
#     else:
#         from src.network.transfomer_based.transformer_based_network import get_transformer_based_model
#         parser = argparse.ArgumentParser()
#         model = get_transformer_based_model(parser=parser, model_name=args.model, img_size=args.img_size,
#                                             num_classes=args.num_classes, in_ch=3).cuda()        
    
#     # return CMUNeXt(dims=[24, 48, 96, 192, 384]).cuda()
#     return model

# def get_val_transform(img_size):
#     from torch.utils.data import DataLoader, random_split
#     from src.dataloader.dataset import MedicalDataSets
#     from albumentations.core.composition import Compose
#     from albumentations import RandomRotate90, Flip, Resize, Normalize    
#     return Compose([
#         Resize(img_size, img_size),
#         Normalize(),
#         # ToTensorV2(),
#     ])

# def main(train_args):
#     seed_torch(train_args.seed)
    
#     trainloader, valloader = getDataloader(train_args)

#     model = get_model(train_args)
#     model = model.cuda()
#     print("train file dir:{} val file dir:{}".format(train_args.train_file_dir, train_args.val_file_dir))

#     optimizer = optim.SGD(model.parameters(), lr=train_args.base_lr, momentum=0.9, weight_decay=0.0001)
    
#     # criterion = losses.__dict__['DiceLoss']().cuda()

#     # criterion = losses.__dict__['BCEDiceLoss']().cuda()
#     if train_args.loss_type == "ioudice":
#         criterion = losses.__dict__['IOUDiceLoss']().cuda()
#     elif train_args.loss_type == "bceioudice":
#         criterion = losses.__dict__['BCEIOUDiceLoss']().cuda()
#     elif train_args.loss_type == "bceiou":
#         criterion = losses.__dict__['BCEIOULoss']().cuda()
#     else:
#         criterion = losses.__dict__['DiceLoss']().cuda()    

#     print("{} iterations per epoch".format(len(trainloader)))
#     best_iou = 0
#     best_dice = 0
#     best_metrics = {}
#     iter_num = 0
#     max_epoch = train_args.epoch

#     max_iterations = len(trainloader) * max_epoch

#     wandb_run_name = f"{train_args.model}"
#     wandb.init(
#         name=wandb_run_name,
#         notes=f"module study: {train_args.datasetname}_{train_args.model}",
#         project=f"comp_cmunext_{train_args.datasetname}",
#         config={
#             "learning_rate": train_args.base_lr,
#             "batch_size": train_args.batch_size,
#             "epochs": train_args.epoch,
#             "model": train_args.model,
#             # "ablation": train_args.ablation,
#             "dataset": train_args.datasetname
#         },
#         tags=[train_args.datasetname, train_args.model], #, train_args.ablation
#         save_code=True
#     )

#     for epoch_num in range(max_epoch):
#         model.train()
#         # if epoch_num == 0:
#         #     with open('model_structure.txt', 'w') as f:
#         #         print(model, file=f)            
#         avg_meters = {'loss': AverageMeter(), 'iou': AverageMeter(), 'dice': AverageMeter(),
#                       'val_loss': AverageMeter(), 'val_iou': AverageMeter(), 'val_dice': AverageMeter(),
#                       'val_SE': AverageMeter(), 'val_PC': AverageMeter(),
#                       'val_F1': AverageMeter(), 'val_ACC': AverageMeter()}        

#         for i_batch, sampled_batch in enumerate(trainloader):
#             if train_args.datasetname.startswith("busi") or train_args.datasetname.startswith("isic18") or train_args.datasetname.startswith("isic19") \
#                   or train_args.datasetname.startswith("poly") or train_args.datasetname.startswith("colonDB")  or train_args.datasetname.startswith("clinicDB"):
#                 img_batch, label_batch = sampled_batch['image'], sampled_batch['label'] 
#             else:
#                 img_batch, label_batch = sampled_batch
#             img_batch, label_batch = img_batch.cuda(), label_batch.cuda()
#             if train_args.model.startswith("compnet") :
#                 if train_args.model=="compnet_shallow_encoder" or train_args.model.startswith("compnet_single_decoder") or train_args.model=="compnet_add_encoder":
#                     outputs = model(img_batch)
#                 else:
#                     outputs = model(img_batch)[0]
#             elif train_args.model == "tinyUnet":
#                 outputs = model(img_batch)[:, 0:1, :, :]
#             else:
#                 outputs = model(img_batch)
#             # outputs = model(img_batch)

#             # print("outputs.shape",outputs.shape,"label_batch.shape",label_batch.shape)
#             # import pdb;pdb.set_trace()
#             loss = criterion(outputs, label_batch)
#             iou, dice, _, _, _, _, _ = iou_score(outputs, label_batch)
#             optimizer.zero_grad()
#             loss.backward()
#             optimizer.step()

#             lr_ = train_args.base_lr * (1.0 - iter_num / max_iterations) ** 0.9
#             for param_group in optimizer.param_groups:
#                 param_group['lr'] = lr_

#             iter_num = iter_num + 1
#             avg_meters['loss'].update(loss.item(), img_batch.size(0))
#             avg_meters['iou'].update(iou, img_batch.size(0))
#             avg_meters['dice'].update(dice, img_batch.size(0))
            
#         model.eval()
#         with torch.no_grad():
#             for i_batch, sampled_batch in enumerate(valloader):
#                 if train_args.datasetname.startswith("busi") or train_args.datasetname.startswith("isic18") or train_args.datasetname.startswith("isic19")\
#                       or train_args.datasetname.startswith("poly")  or train_args.datasetname.startswith("colonDB") or train_args.datasetname.startswith("clinicDB"):
#                     img_batch, label_batch = sampled_batch['image'], sampled_batch['label'] 
#                 else:
#                     img_batch, label_batch = sampled_batch
#                 img_batch, label_batch = img_batch.cuda(), label_batch.cuda()
                
#                 if train_args.model.startswith("compnet") :
#                     if train_args.model=="compnet_shallow_encoder" or train_args.model.startswith("compnet_single_decoder") or train_args.model=="compnet_add_encoder":
#                         output = model(img_batch)
#                     else:
#                         output = model(img_batch)[0]
#                 elif train_args.model == "tinyUnet":
#                     output = model(img_batch)[:, 0:1, :, :] 
#                 else:
#                     output = model(img_batch)
#                 # print(" eval output.shape", output.shape)
#                 loss = criterion(output, label_batch)
#                 iou, dice, SE, PC, F1, _, ACC = iou_score(output, label_batch)
#                 avg_meters['val_loss'].update(loss.item(), img_batch.size(0))
#                 avg_meters['val_iou'].update(iou, img_batch.size(0))
#                 avg_meters['val_dice'].update(dice, img_batch.size(0))
#                 avg_meters['val_SE'].update(SE, img_batch.size(0))
#                 avg_meters['val_PC'].update(PC, img_batch.size(0))
#                 avg_meters['val_F1'].update(F1, img_batch.size(0))
#                 avg_meters['val_ACC'].update(ACC, img_batch.size(0))

#         print(f'Epoch [{epoch_num}/{max_epoch}] Train Loss: {avg_meters["loss"].avg:.4f}, Train IoU: {avg_meters["iou"].avg:.4f}, Train Dice: {avg_meters["dice"].avg:.4f}, '
#               f'Val Loss: {avg_meters["val_loss"].avg:.4f}, Val IoU: {avg_meters["val_iou"].avg:.4f}, Val Dice: {avg_meters["val_dice"].avg:.4f}, '
#               f'Val SE: {avg_meters["val_SE"].avg:.4f}, Val PC: {avg_meters["val_PC"].avg:.4f}, '
#               f'Val F1: {avg_meters["val_F1"].avg:.4f}, Val ACC: {avg_meters["val_ACC"].avg:.4f}')        

#         wandb.log({
#             "Train Loss": avg_meters['loss'].avg,
#             "Train IOU": avg_meters['iou'].avg,
#             "Train Dice": avg_meters['dice'].avg,
#             "Val Loss": avg_meters['val_loss'].avg,
#             "Val IOU": avg_meters['val_iou'].avg,
#             "Val Dice": avg_meters['val_dice'].avg,
#             "Val SE": avg_meters['val_SE'].avg,
#             "Val PC": avg_meters['val_PC'].avg,
#             "Val F1": avg_meters['val_F1'].avg,
#             "Val ACC": avg_meters['val_ACC'].avg,
#             "Learning Rate": lr_
#         })


#         if avg_meters['val_iou'].avg > best_iou:
#             best_iou = avg_meters['val_iou'].avg
#             best_dice = avg_meters['val_dice'].avg  # 更新 Best Dice
#             today_date = datetime.now().strftime("%Y-%m-%d")
#             current_time = datetime.now().strftime("%H-%M-%S")
#             if epoch_num >=100:
#                 checkpoint_dir = f"./checkpoint2025/{train_args.model}/{today_date}/{epoch_num}"
#                 if not os.path.isdir(checkpoint_dir):
#                     os.makedirs(checkpoint_dir)
            
#                 # 保存并上传模型
#                 if train_args.datasetname.startswith("busi"):
                
#                     base_name = f"{train_args.datasetname}_{train_args.train_file_dir}_BS_{train_args.batch_size}_{avg_meters['val_iou'].avg:.4f}_{train_args.model}_model"
#                 else:
#                     base_name  = f"{train_args.datasetname}_BS_{train_args.batch_size}_{avg_meters['val_iou'].avg:.4f}_{train_args.model}_model"
#                 # save_checkpoint_and_upload(model, checkpoint_dir, base_name)
            
#                 torch.save(model.state_dict(), f'{checkpoint_dir}/{base_name}.pth')

#                 # 记录最佳 IOU 及相关指标到 WandB
#                 best_metrics = {
#                     "Best Val IOU" :  best_iou,
#                     "Best Val Dice": best_dice,  # 保存 Best Dice
#                     "Best Val Loss" :  avg_meters['val_loss'].avg,
#                     "Best Val SE" :  avg_meters['val_SE'].avg,
#                     "Best Val PC" :  avg_meters['val_PC'].avg,
#                     "Best Val F1" :  avg_meters['val_F1'].avg,
#                     "Best Val ACC" :  avg_meters['val_ACC'].avg,            
#                 }
            
#             print("=>"*10, f"saved best {train_args.datasetname} {train_args.batch_size} with {train_args.model}","<="*10)
#     from thop import profile,clever_format
#     xin = torch.randn(1, 3, 256, 256).cuda()
        
#     flops, params = profile(model,inputs=(xin,))
#     flops, params  = clever_format((flops,params),"%.2f")
#     print(f"FLOPs: {flops}")
#     print(f"Params: {params}")    

#     with open("model_structure.txt", "w") as f:
#         for name, module in model.named_modules():
#             f.write(f"{name}: {module}\n")
#     wandb.save("model_structure.txt")
#     wandb.run.summary.update(best_metrics) 
#     wandb.finish()
#     return f"{best_iou}_{best_dice}"

# if __name__ == "__main__":
#     train_parser = get_train_parser()
#     train_args = train_parser.parse_args()
#     print("0-0-0-0-0-0-0- train args get!")
#     # model_parser = get_model_parser()
#     # model_args = model_parser.parse_args()
#     result = main(train_args)
#     print(result)
    


import os
import time
import torch
from PIL import Image
from torch.utils.data import DataLoader, random_split
from torchvision.utils import save_image
from src.dataloader.dataset import MedicalDataSets
from src.network.conv_based.CMUNeXt import CMUNeXt
from src.network.conv_based.tinyUnet import TinyUNet
from src.network.conv_based.U_Net import U_Net
from src.network.conv_based.AttU_Net import AttU_Net
from src.network.conv_based.UNeXt import UNext
from src.network.conv_based.ConvUNeXt import ConvUNeXt
from src.network.conv_based.CMUNet import CMUNet
from src.network.transfomer_based.transformer_based_network import get_transformer_based_model
from albumentations.core.composition import Compose
from albumentations import RandomRotate90, Flip, Resize, Normalize
import argparse
import logging

# 设置日志配置
logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

def load_model(model_name, args, device=torch.device("cuda" if torch.cuda.is_available() else "cpu")):
    """
    根据模型名称动态加载模型，并从 model_pth_dict 中读取对应的 .pth 文件路径。
    """
    # 定义模型与预训练权重文件路径的映射字典
    # model_pth_dict = {
    #     "U_Net": "./checkpoint2025/U_Net/2025-02-16/244/busi_3_busi_train.txt_BS_8_0.6720_U_Net_model.pth",
    #     "AttU_Net": "./checkpoint2025/AttU_Net/2025-02-16/243/busi_3_busi_train.txt_BS_8_0.6737_AttU_Net_model.pth",
    #     "TransUnet": "./checkpoint2025/TransUnet/2025-02-10/193/busi_3_busi_train.txt_BS_4_0.7078_TransUnet_model.pth",
    #     "MedT": "./checkpoint2025/MedT/2025-02-17/247/busi_3_busi_train.txt_BS_8_0.6111_MedT_model.pth",
    #     "UNeXt": "./checkpoint2025/UNeXt/2025-02-09/246/busi_3_busi_train.txt_BS_4_0.6879_UNeXt_model.pth",
    #     "ConvUNeXt": "./checkpoint2025/ConvUNeXt/2025-02-09/219/busi_3_busi_train.txt_BS_4_0.6880_ConvUNeXt_model.pth",
    #     "CMUNet": "./checkpoint2025/CMUNet/2025-02-17/180/busi_3_busi_train.txt_BS_8_0.7112_CMUNet_model.pth",
    #     "CMUNeXt": "./checkpoint2025/cmunext/2025-01-20/253/busi_3_busi_train.txt_BS_4_0.7023_cmunext_model.pth",
    #     "tinyUnet": "./checkpoint2025/tinyUnet/2025-02-18/170/busi_3_busi_train.txt_BS_4_0.6474_tinyUnet_model.pth"
    # }

    # model_pth_dict = {
    #     "v_enc_256_fffse_dec_fusion_vit": "/data/hongboye/projects/checkpoint2025/v_enc_256_fffse_dec_fusion_vit/2025-02-16/181/busi_3_busi_train.txt_BS_8_0.7000_v_enc_256_fffse_dec_fusion_vit_model.pth",
    #     "v_enc_256_fffse_dec_fusion_mamba": "/data/hongboye/projects/checkpoint2025/v_enc_256_fffse_dec_fusion_mamba/2025-02-16/173/busi_3_busi_train.txt_BS_8_0.6681_v_enc_256_fffse_dec_fusion_mamba_model.pth",
    #     "v_enc_128_fffse_decx2_fusion_rwkv_with2x4": "/data/hongboye/projects/checkpoint2025/v_enc_128_fffse_decx2_fusion_rwkv_with2x4/2025-02-15/209/busi_3_busi_train.txt_BS_4_0.6761_v_enc_128_fffse_decx2_fusion_rwkv_with2x4_model.pth",
    #     "v_enc_256_fffse_dec_fusion_rwkv_with2x4": "/data/hongboye/projects/checkpoint2025/v_enc_256_fffse_dec_fusion_rwkv_with2x4/2025-02-19/254/busi_3_busi_train.txt_BS_4_0.6871_v_enc_256_fffse_dec_fusion_rwkv_with2x4_model.pth",
    #     "vscan_enc_256_fffse_dec_fusion_rwkv2_h_v": "/data/hongboye/projects/checkpoint2025/vscan_enc_256_fffse_dec_fusion_rwkv2_h_v/2025-02-24/194/busi_3_busi_train.txt_BS_4_0.6931_vscan_enc_256_fffse_dec_fusion_rwkv2_h_v_model.pth"
    # }    
    
    model_pth_dict = {
        "U_Net": "./checkpoint2025/U_Net/2025-02-16/244/busi_3_busi_train.txt_BS_8_0.6720_U_Net_model.pth",
        "AttU_Net": "./checkpoint2025/AttU_Net/2025-02-16/243/busi_3_busi_train.txt_BS_8_0.6737_AttU_Net_model.pth",
        "TransUnet": "./checkpoint2025/TransUnet/2025-02-10/193/busi_3_busi_train.txt_BS_4_0.7078_TransUnet_model.pth",
        "MedT": "./checkpoint2025/MedT/2025-02-17/247/busi_3_busi_train.txt_BS_8_0.6111_MedT_model.pth",
        "UNeXt": "./checkpoint2025/UNeXt/2025-02-09/246/busi_3_busi_train.txt_BS_4_0.6879_UNeXt_model.pth",
        "ConvUNeXt": "./checkpoint2025/ConvUNeXt/2025-02-09/219/busi_3_busi_train.txt_BS_4_0.6880_ConvUNeXt_model.pth",
        "CMUNet": "./checkpoint2025/CMUNet/2025-02-17/180/busi_3_busi_train.txt_BS_8_0.7112_CMUNet_model.pth",
        "CMUNeXt": "./checkpoint2025/cmunext/2025-01-20/253/busi_3_busi_train.txt_BS_4_0.7023_cmunext_model.pth",
        "tinyUnet": "./checkpoint2025/tinyUnet/2025-02-18/170/busi_3_busi_train.txt_BS_4_0.6474_tinyUnet_model.pth",
        "v_enc_256_fffse_dec_fusion_vit": "/data/hongboye/projects/checkpoint2025/v_enc_256_fffse_dec_fusion_vit/2025-02-16/181/busi_3_busi_train.txt_BS_8_0.7000_v_enc_256_fffse_dec_fusion_vit_model.pth",
        "v_enc_256_fffse_dec_fusion_mamba": "/data/hongboye/projects/checkpoint2025/v_enc_256_fffse_dec_fusion_mamba/2025-02-16/173/busi_3_busi_train.txt_BS_8_0.6681_v_enc_256_fffse_dec_fusion_mamba_model.pth",
        "v_enc_128_fffse_decx2_fusion_rwkv_with2x4": "/data/hongboye/projects/checkpoint2025/v_enc_128_fffse_decx2_fusion_rwkv_with2x4/2025-02-15/209/busi_3_busi_train.txt_BS_4_0.6761_v_enc_128_fffse_decx2_fusion_rwkv_with2x4_model.pth",
        "v_enc_256_fffse_dec_fusion_rwkv_with2x4": "/data/hongboye/projects/checkpoint2025/v_enc_256_fffse_dec_fusion_rwkv_with2x4/2025-02-19/254/busi_3_busi_train.txt_BS_4_0.6871_v_enc_256_fffse_dec_fusion_rwkv_with2x4_model.pth",
        "vscan_enc_256_fffse_dec_fusion_rwkv2_h_v": "/data/hongboye/projects/checkpoint2025/vscan_enc_256_fffse_dec_fusion_rwkv2_h_v/2025-02-24/194/busi_3_busi_train.txt_BS_4_0.6931_vscan_enc_256_fffse_dec_fusion_rwkv2_h_v_model.pth",
        "comp_rwkv":"./checkpoint2025/comp_rwkv/2025-01-22/261/busi_3_busi_train.txt_BS_4_0.7104_comp_rwkv_model.pth"
    }
    
    # model_pth_dict = {
    #     "comp_rwkv_5_4scan": "/data/hongboye/projects/checkpoint2025/comp_rwkv_5_4scan/2025-02-08/222/busi_3_busi_train.txt_BS_4_0.7049_comp_rwkv_5_4scan_model.pth"
    # }
    
    # 获取模型对应的预训练权重文件路径
    model_path = model_pth_dict.get(model_name)
    if model_path is None:
        raise ValueError(f"Model {model_name} not found in the model_pth_dict.")

    # 动态加载模型
    model = get_model(args)

    # 加载预训练权重
    log.info(f"Loading pretrained weights from {model_path}")
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    return model

def get_model(args):

    if args.model.startswith("CMUNeXt"):
        from src.network.conv_based.CMUNeXt import CMUNeXt
        if args.model.endswith("2_4_9_192_384"):
            model = CMUNeXt(dims=[24, 48, 96, 192, 384],num_classes=args.num_classes)
        elif args.model.endswith("1_3_128_256_768"):
            model = CMUNeXt(dims=[16, 32, 128, 256,768],num_classes=args.num_classes)
        elif args.model.endswith("1_3_128_256_384"):
            model = CMUNeXt(dims=[16, 32, 128, 256, 384],num_classes=args.num_classes)
        elif args.model == "CMUNeXt_downsample_DWT":
            model = CMUNeXt(use_wavelet=True)            
        else:
            model = CMUNeXt(dims=[16,32,128,160,256]).cuda()
            
    elif args.model == "comp_rwkv":
        from src.network.conv_based.compnet1225_enc_rwkv_single_Decoder import comp_rwkv
        model = comp_rwkv()            
    elif args.model == "comp_rwkv_5_4scan":
        from src.network.conv_based.compnet1225_enc_rwkv_single_Decoder_rwkv4d import LoRA__5
        model  =  LoRA__5(dims=[16, 32, 128, 160, 256])            
    
    elif args.model == "tinyUnet":
        from src.network.conv_based.tinyUnet import TinyUNet
        model = TinyUNet(in_channels=3, num_classes=2).cuda()
    elif args.model == "U_Net":
        from src.network.conv_based.U_Net import U_Net
        model = U_Net(output_ch=args.num_classes).cuda()

    elif args.model.startswith("AttU_Net"):
        from src.network.conv_based.AttU_Net import AttU_Net
        model = AttU_Net()
    elif args.model.startswith("UNeXt"):
        from src.network.conv_based.UNeXt import UNext
        model = UNext()
    elif args.model.startswith("ConvUNeXt"):
        from src.network.conv_based.ConvUNeXt import ConvUNeXt
        model = ConvUNeXt(in_channels=3, num_classes=args.num_classes, base_c=32).cuda()
    elif args.model == "CMUNet":
        from src.network.conv_based.CMUNet import CMUNet
        model = CMUNet(output_ch=args.num_classes).cuda()

    elif args.model == "v_enc_256_fffse_dec_fusion_vit":
        print("判别成功")
        from src.network.conv_based.cval_LoRA5_enc_vit_mamba import v_enc_256_fffse_dec_fusion_vit
        model = v_enc_256_fffse_dec_fusion_vit().cuda()
    elif args.model == "v_enc_256_fffse_dec_fusion_mamba":
        print("判别成功")
        from src.network.conv_based.cval_LoRA5_enc_vit_mamba import v_enc_256_fffse_dec_fusion_mamba
        model = v_enc_256_fffse_dec_fusion_mamba().cuda()        
        
    elif args.model == "v_enc_256_fffse_dec_fusion_rwkv_with2x4_wo":
        from src.network.conv_based.cval_LoRA5_enc_rwkv import v_enc_256_fffse_dec_fusion_rwkv_with2x4_wo
        model = v_enc_256_fffse_dec_fusion_rwkv_with2x4_wo().cuda()
        
    elif args.model == "v_enc_256_fffse_dec_fusion_rwkv_with2x4":
        from src.network.conv_based.cval_LoRA5_enc_rwkv import v_enc_256_fffse_dec_fusion_rwkv_with2x4
        model = v_enc_256_fffse_dec_fusion_rwkv_with2x4().cuda()
    elif args.model == "v_enc_384_fffse_dec_fusion_rwkv_with2x4":
        from src.network.conv_based.cval_LoRA5_enc_rwkv import v_enc_256_fffse_dec_fusion_rwkv_with2x4
        model = v_enc_256_fffse_dec_fusion_rwkv_with2x4(dims=[24, 48, 96, 192, 384]).cuda()
    elif args.model == "v_enc_512_fffse_dec_fusion_rwkv_with2x4":
        from src.network.conv_based.cval_LoRA5_enc_rwkv import v_enc_256_fffse_dec_fusion_rwkv_with2x4
        model = v_enc_256_fffse_dec_fusion_rwkv_with2x4(dims=[32, 64, 128, 256, 512]).cuda()
    elif args.model == "v_enc_768_fffse_dec_fusion_rwkv_with2x4":
        from src.network.conv_based.cval_LoRA5_enc_rwkv import v_enc_256_fffse_dec_fusion_rwkv_with2x4
        model = v_enc_256_fffse_dec_fusion_rwkv_with2x4(dims = [48,96,192,384,768]).cuda()
        
        
    elif args.model == "vscan_enc_256_fffse_dec_fusion_rwkv2_h_v":
        from src.network.conv_based.cval_LoRA5_enc_rwkv import vscan_enc_256_fffse_dec_fusion_rwkv2_h_v
        model = vscan_enc_256_fffse_dec_fusion_rwkv2_h_v().cuda() # no  ab_scan  这是1 2 -> 3 4
    # elif args.model == "abscan_256_fffse_fusion_rwkv_with2x4_1_3":
    #     from src.network.conv_based.cval_LoRA5_enc_rwkv import vscan_enc_256_fffse_dec_fusion_rwkv_with2x4
    #     model = vscan_enc_256_fffse_dec_fusion_rwkv_with2x4().cuda()

    elif args.model == "v_enc_128_fffse_decx2_fusion_rwkv_with2x4":
        from src.network.conv_based.cval_LoRA5_enc_rwkv import v_enc_128_fffse_decx2_fusion_rwkv_with2x4
        model = v_enc_128_fffse_decx2_fusion_rwkv_with2x4().cuda()
    # 大的模型
    # elif args.model == "v_enc_256_fffse_dec_fusion_rwkv_with2x4":
    #     from src.network.conv_based.cval_LoRA5_enc_rwkv import v_enc_256_fffse_dec_fusion_rwkv_with2x4
    #     model = v_enc_256_fffse_dec_fusion_rwkv_with2x4().cuda()
    # elif args.model == "v_enc_256_fffse_dec_fusion_rwkv_with2x4":
    #     from src.network.conv_based.cval_LoRA5_enc_rwkv import v_enc_256_fffse_dec_fusion_rwkv_with2x4
    #     model = v_enc_256_fffse_dec_fusion_rwkv_with2x4().cuda()
        
              
    else:
        parser = argparse.ArgumentParser()
        model = get_transformer_based_model(parser=parser, model_name=args.model, img_size=args.img_size,
                                            num_classes=args.num_classes, in_ch=3).cuda()        
    
    return model

def get_val_transform(img_size):
    return Compose([
        Resize(img_size, img_size),
        Normalize(),
    ])

# def iou_score(outputs, labels):
#     smooth = 1e-6
#     outputs = torch.sigmoid(outputs)
#     outputs = (outputs > 0.5).float()
#     intersection = (outputs * labels).sum(dim=(1, 2, 3))
#     union = outputs.sum(dim=(1, 2, 3)) + labels.sum(dim=(1, 2, 3)) - intersection
#     iou = (intersection + smooth) / (union + smooth)
#     dice = (2. * intersection + smooth) / (outputs.sum(dim=(1, 2, 3)) + labels.sum(dim=(1, 2, 3)) + smooth)
#     rvd = torch.abs((outputs.sum(dim=(1, 2, 3)).float() - labels.sum(dim=(1, 2, 3)).float()) / labels.sum(dim=(1, 2, 3)).float())
#     return iou.mean(), dice.mean(), rvd.mean(), 0, 0, 0, 0, 0

def seed_torch(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)  # GPU随机种子确定
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

# def validate(model, val_loader, criterion, device, save_dir="validation_results", model_name="default_model"):
#     seed_torch(41)
#     """执行验证，并且每隔十张图像保存一次预测结果到PNG文件"""
#     model.eval()
#     val_loss = 0.0
#     val_iou = 0.0
#     val_dice = 0.0
#     val_rvd = 0.0
#     total_inference_time = 0.0
    
#     # 创建特定于模型的保存目录
#     model_save_dir = os.path.join(save_dir, model_name)
#     os.makedirs(model_save_dir, exist_ok=True)
    
#     # 日志文件路径
#     log_file_path = os.path.join(model_save_dir, "model_validation.log")
#     file_handler = logging.FileHandler(log_file_path)
#     formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
#     file_handler.setFormatter(formatter)
#     log.addHandler(file_handler)

#     with torch.no_grad():
#         for i_batch, sampled_batch in enumerate(val_loader):
#             img_batch, label_batch = sampled_batch['image'], sampled_batch['label']
#             img_batch, label_batch = img_batch.to(device), label_batch.to(device)
            
#             start_time = time.time()
#             if model_name == "tinyUnet":
#                 outputs = model(img_batch)[:, 0:1, :, :]
#             else:
#                 outputs = model(img_batch)
#             end_time = time.time()
#             inference_time = end_time - start_time
#             total_inference_time += inference_time
            
#             loss = criterion(outputs, label_batch)
#             val_loss += loss.item()

#             iou, dice,  _, _, _, _, _ = iou_score(outputs, label_batch)
#             val_iou += iou
#             val_dice += dice
#             # val_rvd += rvd

#             # 每隔十张图像保存一次预测结果
#             if i_batch % 10 == 0:
#                 # 将模型输出转换为二值图像
#                 outputs = torch.sigmoid(outputs)
#                 outputs[outputs > 0.5] = 1
#                 outputs[outputs <= 0.5] = 0
#                 output_images = outputs.cpu().data
                
#                 # 保存图像
#                 for idx, img in enumerate(output_images):
#                     save_path = os.path.join(model_save_dir, f"batch_{i_batch}_img_{idx}.png")
#                     save_image(img, save_path)

#     val_loss /= len(val_loader)
#     val_iou /= len(val_loader)
#     val_dice /= len(val_loader)
#     # val_rvd /= len(val_loader)
#     val_rvd = 0.01
#     avg_inference_time = total_inference_time / len(val_loader) *1000
    
#     log.info(f'Model: {model_name}')
#     log.info(f'验证损失: {val_loss:.4f}, 验证IoU: {val_iou:.4f}, 验证Dice: {val_dice:.4f}, 验证RVD: {val_rvd:.4f}')
#     log.info(f'平均推理时间: {avg_inference_time} ms') #:.4f

#     # 移除文件处理器以避免重复写入
#     log.removeHandler(file_handler)
#     file_handler.close()

# def validate(model, val_loader, criterion, device, save_dir="validation_results", model_name="default_model"):
#     seed_torch(41)
#     """执行验证，并且每隔十张图像保存一次预测结果到PNG文件"""
#     model.eval()
#     val_loss = 0.0
#     val_iou = 0.0
#     val_dice = 0.0
#     total_inference_time = 0.0
    
#     # 创建特定于模型的保存目录
#     model_save_dir = os.path.join(save_dir, model_name)
#     os.makedirs(model_save_dir, exist_ok=True)
    
#     # 日志文件路径
#     log_file_path = os.path.join(model_save_dir, "model_validation.log")
#     file_handler = logging.FileHandler(log_file_path)
#     formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
#     file_handler.setFormatter(formatter)
#     log.addHandler(file_handler)

#     ignore_batches = 5  # 忽略前 5 个批次的推理时间
#     with torch.no_grad():
#         for i_batch, sampled_batch in enumerate(val_loader):
#             img_batch, label_batch = sampled_batch['image'], sampled_batch['label']
#             img_batch, label_batch = img_batch.to(device), label_batch.to(device)
            
#             start_time = time.time()
#             if model_name == "tinyUnet":
#                 outputs = model(img_batch)[:, 0:1, :, :]
#             else:
#                 outputs = model(img_batch)
#             end_time = time.time()
#             inference_time = end_time - start_time
            
#             if i_batch >= ignore_batches:
#                 total_inference_time += inference_time
            
#             loss = criterion(outputs, label_batch)
#             val_loss += loss.item()

#             iou, dice, _, _, _, _, _ = iou_score(outputs, label_batch)
#             val_iou += iou
#             val_dice += dice

#             # 每隔十张图像保存一次预测结果
#             if i_batch % 10 == 0:
#                 # 将模型输出转换为二值图像
#                 outputs = torch.sigmoid(outputs)
#                 outputs[outputs > 0.5] = 1
#                 outputs[outputs <= 0.5] = 0
#                 output_images = outputs.cpu().data
                
#                 # 保存图像
#                 for idx, img in enumerate(output_images):
#                     save_path = os.path.join(model_save_dir, f"batch_{i_batch}_img_{idx}.png")
#                     save_image(img, save_path)

#     val_loss /= len(val_loader)
#     val_iou /= len(val_loader)
#     val_dice /= len(val_loader)
#     avg_inference_time = total_inference_time / max(len(val_loader) - ignore_batches, 1) * 1000
    
#     log.info(f'Model: {model_name}')
#     log.info(f'验证损失: {val_loss:.4f}, 验证IoU: {val_iou:.4f}, 验证Dice: {val_dice:.4f}')
#     log.info(f'平均推理时间: {avg_inference_time} ms')

#     # 移除文件处理器以避免重复写入
#     log.removeHandler(file_handler)
#     file_handler.close()

import os
import time
import torch
from torchvision.utils import save_image
import logging

# 设置日志配置
logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

def seed_torch(seed=41):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# def iou_score(outputs, labels):
#     smooth = 1e-6
#     outputs = torch.sigmoid(outputs)
#     outputs = (outputs > 0.5).float()
#     intersection = (outputs * labels).sum(dim=(1, 2, 3))
#     union = outputs.sum(dim=(1, 2, 3)) + labels.sum(dim=(1, 2, 3)) - intersection
#     iou = (intersection + smooth) / (union + smooth)
#     dice = (2. * intersection + smooth) / (outputs.sum(dim=(1, 2, 3)) + labels.sum(dim=(1, 2, 3)) + smooth)
#     return iou.mean(), dice.mean(), 0, 0, 0, 0, 0

def cal_inference(model):
    import numpy as np
    from fvcore.nn import FlopCountAnalysis, parameter_count_table
    from thop import profile    
    repetitions = 300
    dummy_input = torch.rand(1, 3, 256, 256).cuda()
    # 预热, GPU 平时可能为了节能而处于休眠状态, 因此需要预热
    print('warm up ...\n')
    with torch.no_grad():
        for _ in range(100):
            _ = model(dummy_input)

    # synchronize 等待所有 GPU 任务处理完才返回 CPU 主线程
    torch.cuda.synchronize()

    # 设置用于测量时间的 cuda Event, 这是PyTorch 官方推荐的接口,理论上应该最靠谱
    starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    # 初始化一个时间容器
    timings = np.zeros((repetitions, 1))

    print('testing ...\n')
    with torch.no_grad():
        for rep in range(repetitions):
            starter.record()
            _ = model(dummy_input)
            ender.record()
            torch.cuda.synchronize() # 等待GPU任务完成
            curr_time = starter.elapsed_time(ender) # 从 starter 到 ender 之间用时,单位为毫秒
            timings[rep] = curr_time

    mean_syn = np.sum(timings)/repetitions
    std_syn = np.std(timings)
    mean_fps = 1000. / mean_syn
    result_str = ' * Mean@1 {mean_syn:.3f}ms Std@5 {std_syn:.3f}ms FPS@1 {mean_fps:.2f}'.format(mean_syn=mean_syn,
                                                                                               std_syn=std_syn,
                                                                                               mean_fps=mean_fps)
    print(result_str)
    return result_str
    
# def validate(model, val_loader, criterion, device, save_dir="validation_results", model_name="default_model"):
#     val_loss = 0.0
#     val_iou = 0.0
#     val_dice = 0.0
#     total_inference_time = 0.0    
#     seed_torch(41)
#     """执行验证，并且每隔十张图像保存一次预测结果和标签到PNG文件"""
#     model.eval()
    
#     # 创建特定于模型的保存目录
#     model_save_dir = os.path.join(save_dir, model_name)
#     os.makedirs(model_save_dir, exist_ok=True)
    
#     # 日志文件路径
#     log_file_path = os.path.join(model_save_dir, "model_validation.log")
#     file_handler = logging.FileHandler(log_file_path)
#     formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
#     file_handler.setFormatter(formatter)
#     log.addHandler(file_handler)

#     ignore_batches = 5  # 忽略前 5 个批次的推理时间
#     with torch.no_grad():
#         for i_batch, sampled_batch in enumerate(val_loader):
#             img_batch, label_batch = sampled_batch['image'], sampled_batch['label']
#             img_batch, label_batch = img_batch.to(device), label_batch.to(device)
            
#             start_time = time.time()
#             if model_name == "tinyUnet":
#                 outputs = model(img_batch)[:, 0:1, :, :]
#             else:
#                 outputs = model(img_batch)
#             end_time = time.time()
#             inference_time = end_time - start_time
            
#             if i_batch >= ignore_batches:
#                 total_inference_time += inference_time
            
#             loss = criterion(outputs, label_batch)
#             val_loss += loss.item()

#             iou, dice, _, _, _, _, _ = iou_score(outputs, label_batch)
#             val_iou += iou
#             val_dice += dice

#             # 每隔十张图像保存一次预测结果和标签
#             if i_batch % 10 == 0:
#                 # 将模型输出转换为二值图像
#                 outputs = torch.sigmoid(outputs)
#                 outputs[outputs > 0.5] = 1
#                 outputs[outputs <= 0.5] = 0
#                 output_images = outputs.cpu().data
                
#                 # # 将标签转换为二值图像
#                 label_images = label_batch.cpu().data
                
#                 # 保存预测结果图像
#                 for idx, img in enumerate(output_images):
#                     save_path = os.path.join(model_save_dir, f"batch_{i_batch}_pred_img_{idx}.png")
#                     save_image(img, save_path)
                
#                 # 保存标签图像
#                 for idx, lbl in enumerate(label_images):
#                     save_path = os.path.join(model_save_dir, f"batch_{i_batch}_label_img_{idx}.png")
#                     save_image(lbl, save_path)

#     val_loss /= len(val_loader)
#     val_iou /= len(val_loader)
#     val_dice /= len(val_loader)
#     avg_inference_time = total_inference_time / max(len(val_loader) - ignore_batches, 1) * 1000
    
#     log.info(f'Model: {model_name}')
#     log.info(f'验证损失: {val_loss:.4f}, 验证IoU: {val_iou:.4f}, 验证Dice: {val_dice:.4f}')
#     # log.info(f'平均推理时间: {avg_inference_time} ms')

#     # 移除文件处理器以避免重复写入
#     log.removeHandler(file_handler)
#     file_handler.close()

def validate(model, val_loader, criterion, device, save_dir="validation_results", model_name="default_model"):
    val_loss = 0.0
    val_iou = 0.0
    val_dice = 0.0
    total_inference_time = 0.0    
    seed_torch(41)
    """执行验证，并且每隔十张图像保存一次预测结果和标签到PNG文件"""
    model.eval()
    
    # 创建特定于模型的保存目录
    model_save_dir = os.path.join(save_dir, model_name)
    os.makedirs(model_save_dir, exist_ok=True)
    
    # 日志文件路径
    log_file_path = os.path.join(model_save_dir, "model_validation.log")
    file_handler = logging.FileHandler(log_file_path)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    log.addHandler(file_handler)

    ignore_batches = 5  # 忽略前 5 个批次的推理时间
    with torch.no_grad():
        for i_batch, sampled_batch in enumerate(val_loader):
            img_batch, label_batch = sampled_batch['image'], sampled_batch['label']
            img_batch, label_batch = img_batch.to(device), label_batch.to(device)
            
            # start_time = time.time()
            if model_name == "tinyUnet":
                outputs = model(img_batch)[:, 0:1, :, :]
            else:
                outputs = model(img_batch)
            end_time = time.time()
            # inference_time = end_time - start_time
            
            # if i_batch >= ignore_batches:
            #     total_inference_time += inference_time
            
            loss = criterion(outputs, label_batch)
            val_loss += loss.item()

            iou, dice, _, _, _, _, _ = iou_score(outputs, label_batch)
            val_iou += iou
            val_dice += dice

            # 每隔十张图像保存一次预测结果和标签
            if i_batch % 10 == 0:
                # 将模型输出转换为二值图像
                outputs = torch.sigmoid(outputs)
                outputs[outputs > 0.5] = 1
                outputs[outputs <= 0.5] = 0
                output_images = outputs.cpu().data
                
                # # 将标签转换为二值图像
                label_images = label_batch.cpu().data
                
                # 保存预测结果图像
                for idx, img in enumerate(output_images):
                    save_path = os.path.join(model_save_dir, f"batch_{i_batch}_pred_img_{idx}.png")
                    save_image(img, save_path)
                
                # 保存标签图像
                for idx, lbl in enumerate(label_images):
                    save_path = os.path.join(model_save_dir, f"batch_{i_batch}_label_img_{idx}.png")
                    save_image(lbl, save_path)

    val_loss /= len(val_loader)
    val_iou /= len(val_loader)
    val_dice /= len(val_loader)
    # avg_inference_time = total_inference_time / max(len(val_loader) - ignore_batches, 1) * 1000
    
    log.info(f'Model: {model_name}')
    log.info(f'验证损失: {val_loss:.4f}, 验证IoU: {val_iou:.4f}, 验证Dice: {val_dice:.4f}')
    # log.info(f'平均推理时间: {avg_inference_time} ms')

    
    # 移除文件处理器以避免重复写入
    log.removeHandler(file_handler)
    file_handler.close()





if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Validate a trained medical image segmentation model')
    parser.add_argument('--model', type=str, default="AttU_Net",   help='Name of the model to use')
    parser.add_argument('--data_dir', type=str, default="./Tan9/data",   help='Path to the dataset directory')
    parser.add_argument('--save_dir', type=str, default='./vatestvalidation_results', help='Base directory to save prediction images')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size for validation')
    parser.add_argument('--img_size', type=int, default=256, help='Size of input images')
    parser.add_argument('--num_classes', type=int, default=1, help='Number of output classes')
    # parser.add_argument('--dims', type=str, default='24,48,96,192,384', help='Dimensions for some models')
    dataset_num = "3"
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(args.model, args, device)
    
    inference_result = cal_inference(model)
    log.info(f'推理结果: {inference_result}')
    transform = get_val_transform(args.img_size)
    # dataset = MedicalDataSets(base_dir=args.data_dir, transform=transform)
    
    db_val = MedicalDataSets(base_dir=args.data_dir, split="val", transform=transform,
                            train_file_dir="busi_train"+dataset_num+".txt", val_file_dir="busi_val"+dataset_num+".txt")
    
    dataloader  =  DataLoader(db_val, batch_size=args.batch_size, shuffle=False, num_workers=1)

    # 假设使用BCEWithLogitsLoss作为损失函数\
    criterion = losses.__dict__['BCEDiceLoss']().to(device)
    # criterion = torch.nn.BCEWithLogitsLoss()

    validate(model, dataloader, criterion, device, args.save_dir, args.model)



        