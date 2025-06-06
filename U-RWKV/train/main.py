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

# from src.network.conv_based.cmunext_vit_test1209 import CMUNeXt
# from src.network.conv_based.cmunext_rwkv_test_inter import CMUNeXt_rwkv_1_3_128_256_768,CMUNeXt_rwkv_1_6_256_512_768
from src.network.conv_based.cmunext_vit_test1209 import CMUNeXt_Vit_SelfAttn

# from src.network.conv_based.LoRD_rwkv import LoRD, LoRD_4plusDeep,LoRD_double_192_384_768,LoRD_128_192_384,LoRD_128_192_384_enc5_bot
# from src.network.conv_based.conv_gated_net import LoG

cpu_num = 4
os.environ['OMP_NUM_THREADS'] = str(cpu_num)
os.environ['OPENBLAS_NUM_THREADS'] = str(cpu_num)
os.environ['MKL_NUM_THREADS'] = str(cpu_num)
os.environ['VECLIB_MAXIMUM_THREADS'] = str(cpu_num)
os.environ['NUMEXPR_NUM_THREADS'] = str(cpu_num)
torch.set_num_threads(cpu_num)

from main_vs_comp_cmunext import seed_torch,get_train_parser,parse_dims,getDataloader


def get_model(args):
    dims = [24,48,96,192,384]
    dims = parse_dims(args.dims)  # 解析 dims 参数
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
    elif args.model.startswith("DWConvFusionNet"):
        from src.network.conv_based.DWConvFusionNet_0114 import DWConvFusionNet
        model = DWConvFusionNet(dims=dims, num_classes=args.num_classes,use_gs=True,use_wt=True)   
    elif args.model == "U_Net":
        from src.network.conv_based.U_Net import U_Net
        model = U_Net(output_ch=args.num_classes).cuda()
    elif args.model == "U_Net1632256":
        from src.network.conv_based.U_Net import U_Net1632256
        model = U_Net1632256(output_ch=args.num_classes).cuda()            
                        
    elif args.model.startswith("compnet"):
        if args.model == ("compnet_with_dw1227"):
            from src.network.conv_based.compnet_wo_dw1227 import CompNet_with_DW
            model = CompNet_with_DW()
        elif args.model ==  ("compnet_wo_dw1227"):
            from src.network.conv_based.compnet_wo_dw1227 import CompNet
            model = CompNet()
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
            
    elif args.model == "v_enc_sym256384":
        from src.network.conv_based.cval_LoRA5_enc_rwkv import v_enc_sym_l_384
        model = v_enc_sym_l_384().cuda()
    elif args.model == "v_enc_128_fffse_dec_resi2_rwkv_withbirwkv":
        from src.network.conv_based.cval_LoRA5_enc_rwkv import v_enc_128_fffse_dec_resi2_rwkv_withbirwkv
        model = v_enc_128_fffse_dec_resi2_rwkv_withbirwkv().cuda()
    elif args.model == "v_enc_128_fffse_dec_resi2_rwkv_with2x4":
        from src.network.conv_based.cval_LoRA5_enc_rwkv import v_enc_256_fffse_dec_resi2_rwkv_with2x4
        model = v_enc_256_fffse_dec_resi2_rwkv_with2x4(dims=[8, 16, 32, 64, 128]).cuda()
    elif args.model == "v_enc_128_fffse_dec_resi2_rwkv_with2":
        from src.network.conv_based.cval_LoRA5_enc_rwkv import v_enc_128_fffse_dec_resi2_rwkv_with2
        model = v_enc_128_fffse_dec_resi2_rwkv_with2().cuda()
    elif args.model == "v_enc_256_fffse_dec_resi2_rwkv_withbirwkv":
        from src.network.conv_based.cval_LoRA5_enc_rwkv import v_enc_256_fffse_dec_resi2_rwkv_withbirwkv
        model = v_enc_256_fffse_dec_resi2_rwkv_withbirwkv().cuda()
    elif args.model == "v_enc_256_fffse_dec_resi2_rwkv_with2x4":
        from src.network.conv_based.cval_LoRA5_enc_rwkv import v_enc_256_fffse_dec_resi2_rwkv_with2x4
        model = v_enc_256_fffse_dec_resi2_rwkv_with2x4().cuda()
        
    elif args.model == "v_enc_512_fffse_dec_resi2_rwkv_with2x4":
        from src.network.conv_based.cval_LoRA5_enc_rwkv import v_enc_256_fffse_dec_resi2_rwkv_with2x4
        model = v_enc_256_fffse_dec_resi2_rwkv_with2x4(dims=[32, 64, 128, 256, 512]).cuda()
    elif args.model == "v_enc_384_fffse_dec_resi2_rwkv_with2x4":
        from src.network.conv_based.cval_LoRA5_enc_rwkv import v_enc_256_fffse_dec_resi2_rwkv_with2x4
        model = v_enc_256_fffse_dec_resi2_rwkv_with2x4(dims=[24, 48, 96, 192, 384]).cuda()
        
    elif args.model == "v_enc_256_fffse_dec_fusion_rwkv_with2x4_wo":
        from src.network.conv_based.cval_LoRA5_enc_rwkv import v_enc_256_fffse_dec_fusion_rwkv_with2x4_wo
        model = v_enc_256_fffse_dec_fusion_rwkv_with2x4_wo().cuda()
    elif args.model == "v_enc_256_fffse_dec_fusion_rwkv_with2x4_single":
        from src.network.conv_based.cval_LoRA5_enc_rwkv import v_enc_256_fffse_dec_fusion_rwkv_with2x4_single
        model = v_enc_256_fffse_dec_fusion_rwkv_with2x4_single().cuda()
        
    elif args.model == "v_enc_256_fffse_dec_fusion_rwkv_with2x4_allx4":
        from src.network.conv_based.cval_LoRA5_enc_rwkv import v_enc_256_fffse_dec_fusion_rwkv_with2x4_allx4
        model = v_enc_256_fffse_dec_fusion_rwkv_with2x4_allx4().cuda()
    elif args.model == "v_enc_256_fffse_dec_fusion_rwkv_with2x4_simpchinchout":
        from src.network.conv_based.cval_LoRA5_enc_rwkv import v_enc_256_fffse_dec_fusion_rwkv_with2x4_simpchinchout
        model = v_enc_256_fffse_dec_fusion_rwkv_with2x4_simpchinchout().cuda()
        
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
        
    elif args.model == "abscan_256_fffse_fusion_rwkv_with2x4_1":
        from src.network.conv_based.cval_LoRA5_enc_rwkv import vscan_enc_256_fffse_dec_fusion_rwkv_with2x4
        model = vscan_enc_256_fffse_dec_fusion_rwkv_with2x4(ab_scan='1').cuda()
    elif args.model == "abscan_256_fffse_fusion_rwkv_with2x4_2":
        from src.network.conv_based.cval_LoRA5_enc_rwkv import vscan_enc_256_fffse_dec_fusion_rwkv_with2x4
        model = vscan_enc_256_fffse_dec_fusion_rwkv_with2x4(ab_scan='2').cuda()
    elif args.model == "abscan_256_fffse_fusion_rwkv_with2x4_1_2":
        from src.network.conv_based.cval_LoRA5_enc_rwkv import vscan_enc_256_fffse_dec_fusion_rwkv_with2x4
        model = vscan_enc_256_fffse_dec_fusion_rwkv_with2x4(ab_scan='1_2').cuda()
    elif args.model == "abscan_256_fffse_fusion_rwkv_with2x4_1_3":
        from src.network.conv_based.cval_LoRA5_enc_rwkv import vscan_enc_256_fffse_dec_fusion_rwkv_with2x4
        model = vscan_enc_256_fffse_dec_fusion_rwkv_with2x4(ab_scan='1_3').cuda()
    elif args.model == "abscan_256_fffse_fusion_rwkv_with2x4_2_4":
        from src.network.conv_based.cval_LoRA5_enc_rwkv import vscan_enc_256_fffse_dec_fusion_rwkv_with2x4
        model = vscan_enc_256_fffse_dec_fusion_rwkv_with2x4(ab_scan='2_4').cuda()
    elif args.model == "abscan_256_fffse_fusion_rwkv_with2x4_3_4":
        from src.network.conv_based.cval_LoRA5_enc_rwkv import vscan_enc_256_fffse_dec_fusion_rwkv_with2x4
        model = vscan_enc_256_fffse_dec_fusion_rwkv_with2x4(ab_scan='3_4').cuda()
    elif args.model == "abscan_256_fffse_fusion_rwkv_with2x4_4":
        from src.network.conv_based.cval_LoRA5_enc_rwkv import vscan_enc_256_fffse_dec_fusion_rwkv_with2x4
        model = vscan_enc_256_fffse_dec_fusion_rwkv_with2x4(ab_scan='4').cuda()
    elif args.model == "abscan_256_fffse_fusion_rwkv_with2x4_3":
        from src.network.conv_based.cval_LoRA5_enc_rwkv import vscan_enc_256_fffse_dec_fusion_rwkv_with2x4
        model = vscan_enc_256_fffse_dec_fusion_rwkv_with2x4(ab_scan='3').cuda()
        
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
        
    elif args.model == "testLoRA__5_160256":
        from src.network.conv_based.compnet1225_enc_rwkv import LoRA__5
        model = LoRA__5(dims=[16, 32, 128, 160, 256],input_channel=3, num_classes=2).cuda()        
    elif args.model == "tinyUnet":
        from src.network.conv_based.tinyUnet import TinyUNet
        model = TinyUNet(in_channels=3, num_classes=2).cuda()
        
    elif args.model == "v_enc_256_fffse_dec_simple2":
        from src.network.conv_based.cval_LoRA5_enc import v_enc_256_fffse_dec_simple2
        model = v_enc_256_fffse_dec_simple2().cuda()
    elif args.model == "v_enc_256_fffse_dec_resi2":
        from src.network.conv_based.cval_LoRA5_enc import v_enc_256_fffse_dec_resi2
        model = v_enc_256_fffse_dec_resi2().cuda()                    
    elif args.model == "v_enc_128_fffse_dec_resi2":
        from src.network.conv_based.cval_LoRA5_enc import v_enc_256_fffse_dec_resi2
        model = v_enc_256_fffse_dec_resi2(dims=[8, 16, 32, 64, 128]).cuda()                    
    elif args.model == "v_enc_base384_fffse_dec_resi2":
        from src.network.conv_based.cval_LoRA5_enc import v_enc_256_fffse_dec_resi2
        model = v_enc_256_fffse_dec_resi2(dims=[24, 48, 96, 192, 384]).cuda()                    
    elif args.model == "v_enc_512_fffse_dec_resi2":
        from src.network.conv_based.cval_LoRA5_enc import v_enc_256_fffse_dec_resi2
        model = v_enc_256_fffse_dec_resi2(dims=[32, 64, 128, 256, 512]).cuda()                    
    else:
        from src.network.transfomer_based.transformer_based_network import get_transformer_based_model
        parser = argparse.ArgumentParser()
        model = get_transformer_based_model(parser=parser, model_name=args.model, img_size=args.img_size,
                                            num_classes=args.num_classes, in_ch=3).cuda()        
    
    # return CMUNeXt(dims=[24, 48, 96, 192, 384]).cuda()
    return model

def main(train_args):
    seed_torch(train_args.seed)
    
    trainloader, valloader = getDataloader(train_args)

    model = get_model(train_args)
    model = model.cuda()
    print("train file dir:{} val file dir:{}".format(train_args.train_file_dir, train_args.val_file_dir))

    optimizer = optim.SGD(model.parameters(), lr=train_args.base_lr, momentum=0.9, weight_decay=0.0001)
    
    # criterion = losses.__dict__['DiceLoss']().cuda()

    # criterion = losses.__dict__['BCEDiceLoss']().cuda()
    if train_args.loss_type == "ioudice":
        criterion = losses.__dict__['IOUDiceLoss']().cuda()
    elif train_args.loss_type == "bceioudice":
        criterion = losses.__dict__['BCEIOUDiceLoss']().cuda()
    elif train_args.loss_type == "bceiou":
        criterion = losses.__dict__['BCEIOULoss']().cuda()
    else:
        criterion = losses.__dict__['DiceLoss']().cuda()    

    print("{} iterations per epoch".format(len(trainloader)))
    best_iou = 0
    best_dice = 0
    best_metrics = {}
    iter_num = 0
    max_epoch = train_args.epoch

    max_iterations = len(trainloader) * max_epoch

    wandb_run_name = f"{train_args.model}"
    wandb.init(
        name=wandb_run_name,
        notes=f"module study: {train_args.datasetname}_{train_args.model}",
        project=f"comp_cmunext_{train_args.datasetname}",
        config={
            "learning_rate": train_args.base_lr,
            "batch_size": train_args.batch_size,
            "epochs": train_args.epoch,
            "model": train_args.model,
            # "ablation": train_args.ablation,
            "dataset": train_args.datasetname
        },
        tags=[train_args.datasetname, train_args.model], #, train_args.ablation
        save_code=True
    )

    for epoch_num in range(max_epoch):
        model.train()
        # if epoch_num == 0:
        #     with open('model_structure.txt', 'w') as f:
        #         print(model, file=f)            
        avg_meters = {'loss': AverageMeter(), 'iou': AverageMeter(), 'dice': AverageMeter(),
                      'val_loss': AverageMeter(), 'val_iou': AverageMeter(), 'val_dice': AverageMeter(),
                      'val_SE': AverageMeter(), 'val_PC': AverageMeter(),
                      'val_F1': AverageMeter(), 'val_ACC': AverageMeter()}        

        for i_batch, sampled_batch in enumerate(trainloader):
            if train_args.datasetname.startswith("busi") or train_args.datasetname.startswith("isic18") or train_args.datasetname.startswith("isic19") \
                  or train_args.datasetname.startswith("poly") or train_args.datasetname.startswith("colonDB")  or train_args.datasetname.startswith("clinicDB"):
                img_batch, label_batch = sampled_batch['image'], sampled_batch['label'] 
            else:
                img_batch, label_batch = sampled_batch
            img_batch, label_batch = img_batch.cuda(), label_batch.cuda()
            if train_args.model.startswith("compnet") :
                if train_args.model=="compnet_shallow_encoder" or train_args.model.startswith("compnet_single_decoder") or train_args.model=="compnet_add_encoder":
                    outputs = model(img_batch)
                else:
                    outputs = model(img_batch)[0]
            elif train_args.model == "tinyUnet":
                outputs = model(img_batch)[:, 0:1, :, :]
            else:
                outputs = model(img_batch)
            # outputs = model(img_batch)

            # print("outputs.shape",outputs.shape,"label_batch.shape",label_batch.shape)
            # import pdb;pdb.set_trace()
            loss = criterion(outputs, label_batch)
            iou, dice, _, _, _, _, _ = iou_score(outputs, label_batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            lr_ = train_args.base_lr * (1.0 - iter_num / max_iterations) ** 0.9
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr_

            iter_num = iter_num + 1
            avg_meters['loss'].update(loss.item(), img_batch.size(0))
            avg_meters['iou'].update(iou, img_batch.size(0))
            avg_meters['dice'].update(dice, img_batch.size(0))
            
        model.eval()
        with torch.no_grad():
            for i_batch, sampled_batch in enumerate(valloader):
                if train_args.datasetname.startswith("busi") or train_args.datasetname.startswith("isic18") or train_args.datasetname.startswith("isic19")\
                      or train_args.datasetname.startswith("poly")  or train_args.datasetname.startswith("colonDB") or train_args.datasetname.startswith("clinicDB"):
                    img_batch, label_batch = sampled_batch['image'], sampled_batch['label'] 
                else:
                    img_batch, label_batch = sampled_batch
                img_batch, label_batch = img_batch.cuda(), label_batch.cuda()
                
                if train_args.model.startswith("compnet") :
                    if train_args.model=="compnet_shallow_encoder" or train_args.model.startswith("compnet_single_decoder") or train_args.model=="compnet_add_encoder":
                        output = model(img_batch)
                    else:
                        output = model(img_batch)[0]
                elif train_args.model == "tinyUnet":
                    output = model(img_batch)[:, 0:1, :, :] 
                else:
                    output = model(img_batch)
                # print(" eval output.shape", output.shape)
                loss = criterion(output, label_batch)
                iou, dice, SE, PC, F1, _, ACC = iou_score(output, label_batch)
                avg_meters['val_loss'].update(loss.item(), img_batch.size(0))
                avg_meters['val_iou'].update(iou, img_batch.size(0))
                avg_meters['val_dice'].update(dice, img_batch.size(0))
                avg_meters['val_SE'].update(SE, img_batch.size(0))
                avg_meters['val_PC'].update(PC, img_batch.size(0))
                avg_meters['val_F1'].update(F1, img_batch.size(0))
                avg_meters['val_ACC'].update(ACC, img_batch.size(0))

        print(f'Epoch [{epoch_num}/{max_epoch}] Train Loss: {avg_meters["loss"].avg:.4f}, Train IoU: {avg_meters["iou"].avg:.4f}, Train Dice: {avg_meters["dice"].avg:.4f}, '
              f'Val Loss: {avg_meters["val_loss"].avg:.4f}, Val IoU: {avg_meters["val_iou"].avg:.4f}, Val Dice: {avg_meters["val_dice"].avg:.4f}, '
              f'Val SE: {avg_meters["val_SE"].avg:.4f}, Val PC: {avg_meters["val_PC"].avg:.4f}, '
              f'Val F1: {avg_meters["val_F1"].avg:.4f}, Val ACC: {avg_meters["val_ACC"].avg:.4f}')        

        wandb.log({
            "Train Loss": avg_meters['loss'].avg,
            "Train IOU": avg_meters['iou'].avg,
            "Train Dice": avg_meters['dice'].avg,
            "Val Loss": avg_meters['val_loss'].avg,
            "Val IOU": avg_meters['val_iou'].avg,
            "Val Dice": avg_meters['val_dice'].avg,
            "Val SE": avg_meters['val_SE'].avg,
            "Val PC": avg_meters['val_PC'].avg,
            "Val F1": avg_meters['val_F1'].avg,
            "Val ACC": avg_meters['val_ACC'].avg,
            "Learning Rate": lr_
        })


        if avg_meters['val_iou'].avg > best_iou:
            best_iou = avg_meters['val_iou'].avg
            best_dice = avg_meters['val_dice'].avg  # 更新 Best Dice
            today_date = datetime.now().strftime("%Y-%m-%d")
            current_time = datetime.now().strftime("%H-%M-%S")
            if epoch_num >=0:
                checkpoint_dir = f"./checkpoint2025/{train_args.model}/{today_date}/{epoch_num}"
                if not os.path.isdir(checkpoint_dir):
                    os.makedirs(checkpoint_dir)
            
                # 保存并上传模型
                if train_args.datasetname.startswith("busi"):
                
                    base_name = f"{train_args.datasetname}_{train_args.train_file_dir}_BS_{train_args.batch_size}_{avg_meters['val_iou'].avg:.4f}_{train_args.model}_model"
                else:
                    base_name  = f"{train_args.datasetname}_BS_{train_args.batch_size}_{avg_meters['val_iou'].avg:.4f}_{train_args.model}_model"
                # save_checkpoint_and_upload(model, checkpoint_dir, base_name)
            
                torch.save(model.state_dict(), f'{checkpoint_dir}/{base_name}.pth')

                # 记录最佳 IOU 及相关指标到 WandB
                best_metrics = {
                    "Best Val IOU" :  best_iou,
                    "Best Val Dice": best_dice,  # 保存 Best Dice
                    "Best Val Loss" :  avg_meters['val_loss'].avg,
                    "Best Val SE" :  avg_meters['val_SE'].avg,
                    "Best Val PC" :  avg_meters['val_PC'].avg,
                    "Best Val F1" :  avg_meters['val_F1'].avg,
                    "Best Val ACC" :  avg_meters['val_ACC'].avg,            
                }
            
            print("=>"*10, f"saved best {train_args.datasetname} {train_args.batch_size} with {train_args.model}","<="*10)
    from thop import profile,clever_format
    xin = torch.randn(1, 3, 256, 256).cuda()
        
    flops, params = profile(model,inputs=(xin,))
    flops, params  = clever_format((flops,params),"%.2f")
    print(f"FLOPs: {flops}")
    print(f"Params: {params}")    

    with open("model_structure.txt", "w") as f:
        for name, module in model.named_modules():
            f.write(f"{name}: {module}\n")
    wandb.save("model_structure.txt")
    wandb.run.summary.update(best_metrics) 
    wandb.finish()
    return f"{best_iou}_{best_dice}"

if __name__ == "__main__":
    train_parser = get_train_parser()
    train_args = train_parser.parse_args()
    print("0-0-0-0-0-0-0- train args get!")
    # model_parser = get_model_parser()
    # model_args = model_parser.parse_args()
    result = main(train_args)
    print(result)
    
    