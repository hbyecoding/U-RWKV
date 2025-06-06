import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import wandb
from datetime import datetime
import os

# 数据预处理
transform = transforms.Compose([
    transforms.Resize((256, 256)),  # 调整图像大小
    transforms.ToTensor(),  # 转换为 Tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # 标准化
])

# 加载 ImageNet-1k 数据集
train_dataset = datasets.ImageFolder(root='/data/hongboye/dataset/IN1k/train', transform=transform)
val_dataset = datasets.ImageFolder(root='/data/hongboye/dataset/IN1k/val', transform=transform)

# 创建 DataLoader
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4)

# 定义模型
from Tan9.src.network.conv_based.compnet1225_enc_rwkv import LoRA__5_Classification
model = LoRA__5_Classification(input_channel=3, num_classes=1000)  # ImageNet-1k 有 1000 个类别
model = model.to(device)  # 将模型移动到 GPU（如果可用）

# 定义损失函数和优化器
criterion = torch.nn.CrossEntropyLoss()  # 交叉熵损失
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)  # Adam 优化器

# 初始化 WandB
wandb.init(
    name=f"{train_args.model}",  # 实验名称 基于数据集
    notes=f"baseline {train_args.datasetname}_{train_args.model}.",  # 实验备注
    project=f"test_{train_args.datasetname}",  # 项目名称 基于数据集
    config={
        "learning_rate": train_args.base_lr,
        "batch_size": train_args.batch_size,
        "epochs": 280
    },  # 实验配置
    tags=["ablation", f"{train_args.datasetname}"],  # 实验标签
    save_code=True  # 保存代码
)

# 训练循环
def train(model, train_loader, criterion, optimizer, device):
    model.train()  # 设置模型为训练模式
    running_loss = 0.0
    correct_top1 = 0
    correct_top5 = 0
    total = 0

    for batch_idx, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.to(device), targets.to(device)  # 将数据移动到设备

        # 前向传播
        outputs = model(inputs)
        loss = criterion(outputs, targets)

        # 反向传播和优化
        optimizer.zero_grad()  # 清空梯度
        loss.backward()  # 反向传播
        optimizer.step()  # 更新参数

        # 统计损失和准确率
        running_loss += loss.item()
        _, predicted = outputs.topk(5, 1, True, True)
        total += targets.size(0)
        correct_top1 += predicted[:, 0].eq(targets).sum().item()
        correct_top5 += predicted.eq(targets.view(-1, 1)).sum().item()

        # 打印训练信息
        if batch_idx % 100 == 0:
            print(f"Train Batch: {batch_idx}/{len(train_loader)}, Loss: {loss.item():.4f}, Top-1 Acc: {100.*correct_top1/total:.2f}%, Top-5 Acc: {100.*correct_top5/total:.2f}%")

    train_loss = running_loss / len(train_loader)
    train_acc_top1 = 100. * correct_top1 / total
    train_acc_top5 = 100. * correct_top5 / total
    return train_loss, train_acc_top1, train_acc_top5

# 验证循环
def validate(model, val_loader, criterion, device):
    model.eval()  # 设置模型为评估模式
    running_loss = 0.0
    correct_top1 = 0
    correct_top5 = 0
    total = 0

    with torch.no_grad():  # 不计算梯度
        for batch_idx, (inputs, targets) in enumerate(val_loader):
            inputs, targets = inputs.to(device), targets.to(device)  # 将数据移动到设备

            # 前向传播
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            # 统计损失和准确率
            running_loss += loss.item()
            _, predicted = outputs.topk(5, 1, True, True)
            total += targets.size(0)
            correct_top1 += predicted[:, 0].eq(targets).sum().item()
            correct_top5 += predicted.eq(targets.view(-1, 1)).sum().item()

    val_loss = running_loss / len(val_loader)
    val_acc_top1 = 100. * correct_top1 / total
    val_acc_top5 = 100. * correct_top5 / total
    return val_loss, val_acc_top1, val_acc_top5

# 完整训练代码
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 设备
num_epochs = 10  # 训练轮数
best_acc_top1 = 0.0
best_acc_top5 = 0.0

for epoch in range(num_epochs):
    print(f"Epoch {epoch+1}/{num_epochs}")
    
    # 训练
    train_loss, train_acc_top1, train_acc_top5 = train(model, train_loader, criterion, optimizer, device)
    print(f"Train Loss: {train_loss:.4f}, Train Top-1 Acc: {train_acc_top1:.2f}%, Train Top-5 Acc: {train_acc_top5:.2f}%")
    
    # 验证
    val_loss, val_acc_top1, val_acc_top5 = validate(model, val_loader, criterion, device)
    print(f"Val Loss: {val_loss:.4f}, Val Top-1 Acc: {val_acc_top1:.2f}%, Val Top-5 Acc: {val_acc_top5:.2f}%")

    # 记录到 WandB
    wandb.log({
        "Train Loss": train_loss,
        "Train Top-1 Acc": train_acc_top1,
        "Train Top-5 Acc": train_acc_top5,
        "Val Loss": val_loss,
        "Val Top-1 Acc": val_acc_top1,
        "Val Top-5 Acc": val_acc_top5,
    })

    # 保存最佳模型
    if val_acc_top1 > best_acc_top1:
        best_acc_top1 = val_acc_top1
        best_acc_top5 = val_acc_top5
        today_date = datetime.now().strftime("%Y-%m-%d")
        checkpoint_dir = f"./checkpoint/{train_args.model}/{today_date}/{epoch+1}"
        if not os.path.isdir(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        
        # 保存模型
        torch.save(model.state_dict(), f'{checkpoint_dir}/best_model.pth')
        
        # 记录最佳指标到 WandB
        wandb.run.summary.update({
            "Best Val Top-1 Acc": best_acc_top1,
            "Best Val Top-5 Acc": best_acc_top5,
        })

# 完成训练
wandb.finish()
print("Training Finished!")