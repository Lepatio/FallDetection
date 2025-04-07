import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import time
import sys

# 修复导入路径
sys.path.append(os.path.abspath("."))
from src.data_preprocessing import prepare_dataloaders
from model.TexFilter import Model

import argparse
from tqdm import tqdm

# 命令行参数解析
def parse_args():
    parser = argparse.ArgumentParser(description="Fall Detection Training")
    parser.add_argument("--batch_size", type=int, default=64, help="batch size")
    parser.add_argument("--epochs", type=int, default=50, help="number of epochs")
    parser.add_argument("--lr", type=float, default=0.001, help="learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-5, help="weight decay")
    parser.add_argument("--model_dir", type=str, default="checkpoints", help="directory to save model")
    parser.add_argument("--data_dir", type=str, default="data", help="data directory")
    parser.add_argument("--output_dir", type=str, default="output", help="output directory")
    parser.add_argument("--seq_len", type=int, default=256, help="sequence length")
    parser.add_argument("--embed_size", type=int, default=128, help="embedding dimension")
    parser.add_argument("--hidden_size", type=int, default=256, help="hidden dimension")
    parser.add_argument("--dropout", type=float, default=0.1, help="dropout rate")
    parser.add_argument("--no_augmentation", action="store_true", help="disable data augmentation")
    parser.add_argument("--no_scaling", action="store_true", help="disable data scaling")
    
    return parser.parse_args()

# 进度条
class ProgressBar:
    def __init__(self, total, prefix='', length=30, fill='█', print_end='\r'):
        self.total = total
        self.prefix = prefix
        self.length = length
        self.fill = fill
        self.print_end = print_end
        self.start_time = time.time()
        self.current = 0
        
    def update(self, current=None, suffix=''):
        if current is not None:
            self.current = current
        else:
            self.current += 1
            
        percent = ("{0:.1f}").format(100 * (self.current / float(self.total)))
        filled_length = int(self.length * self.current // self.total)
        bar = self.fill * filled_length + '-' * (self.length - filled_length)
        
        elapsed_time = time.time() - self.start_time
        if self.current > 0:
            eta = elapsed_time * (self.total / self.current - 1)
            time_info = f" | ETA: {self._format_time(eta)} | Elapsed: {self._format_time(elapsed_time)}"
        else:
            time_info = ""
            
        print(f'\r{self.prefix} |{bar}| {percent}% {suffix}{time_info}', end=self.print_end)
        
        if self.current == self.total:
            print()
    
    def reset(self):
        """重置进度条"""
        self.current = 0
        self.start_time = time.time()
            
    def _format_time(self, seconds):
        m, s = divmod(int(seconds), 60)
        h, m = divmod(m, 60)
        return f"{h:02d}:{m:02d}:{s:02d}"

# 配置类，用于传递参数给模型
class Config:
    def __init__(self, args):
        self.seq_len = args.seq_len
        self.pred_len = 3  # 3分类预测
        self.enc_in = 6  # 6个特征
        self.embed_size = args.embed_size
        self.d_model = args.embed_size  # 添加d_model属性
        self.hidden_size = args.hidden_size
        self.dropout = args.dropout
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.epochs = args.epochs  # 添加epochs属性
        self.model_name = f"model_{self.seq_len}_{self.embed_size}_{self.hidden_size}_{self.dropout}.pth"

# 训练函数
def train(model, train_loader, val_loader=None, criterion=None, optimizer=None, scheduler=None, config=None, save_dir=None):
    """
    训练模型
    
    Args:
        model: 模型
        train_loader: 训练数据加载器
        val_loader: 验证数据加载器 (可选)
        criterion: 损失函数
        optimizer: 优化器
        scheduler: 学习率调度器 (可选)
        config: 配置 (可选)
        save_dir: 保存目录 (可选)
    
    Returns:
        如果提供了完整参数: 返回训练历史记录
        如果只提供了基本参数: 返回当前epoch的训练损失和准确率
    """
    # 兼容旧的函数签名
    if val_loader is None and isinstance(criterion, torch.nn.Module) and isinstance(optimizer, torch.optim.Optimizer) and isinstance(scheduler, (int, str)) and isinstance(config, object):
        # 旧的函数签名: train(model, train_loader, criterion, optimizer, device, epoch, args)
        device = optimizer  # 在旧签名中，device是第4个参数
        epoch = scheduler   # 在旧签名中，epoch是第5个参数
        args = config       # 在旧签名中，args是第6个参数
        
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        progress = ProgressBar(len(train_loader), prefix=f'Epoch {epoch}/{args.epochs} [Train]')
        
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            
            # 确保输出和目标的形状正确
            if outputs.dim() > 2:  # 如果输出是三维的 [batch, seq, classes]
                # 将输出调整为 [batch, classes] 通过对序列维度平均
                outputs = outputs.mean(dim=1)
            
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
            progress.update(batch_idx + 1, suffix=f'Loss: {total_loss/(batch_idx+1):.4f} | Acc: {100.*correct/total:.2f}%')
        
        return total_loss / len(train_loader), correct / total
    
    # 新的函数实现，完整的训练过程
    device = getattr(config, 'device', torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    model.to(device)
    
    # 记录训练历史
    history = {
        'train_loss': [],
        'val_loss': [],
        'train_acc': [],
        'val_acc': []
    }
    
    best_val_loss = float('inf')
    patience = getattr(config, 'patience', 10)  # 提前停止的耐心值
    early_stopping_counter = 0
    
    # 创建进度条
    train_progress = ProgressBar(
        total=len(train_loader),
        prefix='Training'
    )
    
    # 确保config有epochs属性，如果没有就使用默认值
    epochs = getattr(config, 'epochs', 50)
    print(f"Training for {epochs} epochs")
    
    # 保存完整路径
    # 确保config有model_name属性，如果没有就创建一个
    if not hasattr(config, 'model_name'):
        config.model_name = f"model_{time.strftime('%Y%m%d_%H%M%S')}.pth"
    
    # 如果save_dir是None，使用默认值
    if save_dir is None:
        save_dir = "checkpoints"
    
    model_save_path = os.path.join(save_dir, config.model_name)
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    
    # 训练循环
    for epoch in range(epochs):
        # 训练阶段
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0
        
        # 重新初始化进度条前缀，然后重置进度条
        train_progress.prefix = f'Epoch {epoch+1}/{epochs} [Train]'
        train_progress.reset()
        
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            
            # 确保输出和目标的形状正确
            if outputs.dim() > 2:  # 如果输出是三维的 [batch, seq, classes]
                # 将输出调整为 [batch, classes] 通过对序列维度平均
                outputs = outputs.mean(dim=1)
            
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            train_total += targets.size(0)
            train_correct += predicted.eq(targets).sum().item()
            
            # 更新进度条
            train_progress.update(batch_idx + 1, suffix=f'Epoch: {epoch+1}/{epochs} | Loss: {train_loss/(batch_idx+1):.4f} | Acc: {100.*train_correct/train_total:.2f}%')
        
        # 计算训练指标
        epoch_train_loss = train_loss / len(train_loader)
        epoch_train_acc = train_correct / train_total
        
        # 验证阶段
        if val_loader:
            model.eval()
            val_loss = 0
            val_correct = 0
            val_total = 0
            
            with torch.no_grad():
                for batch_idx, (inputs, targets) in enumerate(val_loader):
                    inputs, targets = inputs.to(device), targets.to(device)
                    outputs = model(inputs)
                    
                    # 确保输出和目标的形状正确
                    if outputs.dim() > 2:  # 如果输出是三维的 [batch, seq, classes]
                        # 将输出调整为 [batch, classes] 通过对序列维度平均
                        outputs = outputs.mean(dim=1)
                    
                    loss = criterion(outputs, targets)
                    
                    val_loss += loss.item()
                    _, predicted = outputs.max(1)
                    val_total += targets.size(0)
                    val_correct += predicted.eq(targets).sum().item()
            
            # 计算验证指标
            val_loss = val_loss / len(val_loader)
            val_acc = val_correct / val_total
            
            # 更新学习率调度器，传递验证损失作为指标
            if scheduler:
                scheduler.step(val_loss)
            
            # 保存历史记录
            history['train_loss'].append(epoch_train_loss)
            history['val_loss'].append(val_loss)
            history['train_acc'].append(epoch_train_acc)
            history['val_acc'].append(val_acc)
            
            # 打印结果
            print(f"\nEpoch: {epoch+1}/{epochs}")
            print(f"Train Loss: {epoch_train_loss:.4f}, Train Acc: {epoch_train_acc*100:.2f}%")
            print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc*100:.2f}%")
            
            # 保存最佳模型
            if val_loss < best_val_loss:
                print(f"Validation loss decreased ({best_val_loss:.4f} --> {val_loss:.4f}). Saving model...")
                best_val_loss = val_loss
                torch.save(model.state_dict(), model_save_path)
                early_stopping_counter = 0  # 重置早停计数器
            else:
                early_stopping_counter += 1
                print(f"Validation loss did not improve. Early stopping counter: {early_stopping_counter}/{patience}")
                
                # 如果连续patience个epoch验证损失没有改善，则提前停止
                if early_stopping_counter >= patience:
                    print(f"Early stopping triggered after {epoch+1} epochs")
                    break
        else:
            # 无验证集，仅保存训练历史
            history['train_loss'].append(epoch_train_loss)
            history['train_acc'].append(epoch_train_acc)
            
            # 打印结果
            print(f"\nEpoch: {epoch+1}/{epochs}")
            print(f"Train Loss: {epoch_train_loss:.4f}, Train Acc: {epoch_train_acc*100:.2f}%")
            
            # 每个epoch保存一次模型
            torch.save(model.state_dict(), model_save_path)
    
    # 返回最后一个epoch的训练损失和准确率
    return history['train_loss'][-1], history['train_acc'][-1]

# 验证/测试函数
def evaluate(model, data_loader, criterion, device, prefix="Validation"):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    all_targets = []
    all_predicted = []
    
    progress = ProgressBar(len(data_loader), prefix=f'[{prefix}]')
    
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(data_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            
            # 直接使用模型进行预测
            outputs = model(inputs)
            
            # 确保输出的形状正确
            if outputs.dim() > 2:  # 如果输出是三维的，取平均
                outputs = outputs.mean(dim=1)
            
            loss = criterion(outputs, targets)
            
            total_loss += loss.item()
            
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
            all_targets.extend(targets.cpu().numpy())
            all_predicted.extend(predicted.cpu().numpy())
            
            progress.update(batch_idx + 1, suffix=f'Loss: {total_loss/(batch_idx+1):.4f} | Acc: {100.*correct/total:.2f}%')
    
    return total_loss / len(data_loader), correct / total, all_targets, all_predicted

# 绘制训练曲线
def plot_training_curves(train_losses, train_accs, val_losses, val_accs, args):
    plt.figure(figsize=(15, 6))
    
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.title('Loss Curves')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.plot(train_accs, label='Train Accuracy')
    plt.plot(val_accs, label='Validation Accuracy')
    plt.title('Accuracy Curves')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, 'training_curves.png'))
    plt.close()

# 绘制混淆矩阵
def plot_confusion_matrix(y_true, y_pred, args, mode='test'):
    class_names = ['BKG', 'ALERT', 'FALL']
    cm = confusion_matrix(y_true, y_pred)
    
    # 计算百分比
    cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
    
    plt.figure(figsize=(8, 8))
    ax = sns.heatmap(
        cm_percent, 
        annot=True, 
        fmt=".2f",
        cmap='Blues',
        cbar=True,
        square=True,
        annot_kws={'size': 12, 'weight': 'bold'},
        xticklabels=class_names,
        yticklabels=class_names,
        linewidths=0.5,
        linecolor='lightgray'
    )
    
    # 修正中文标签
    ax.set_xlabel('Predicted Label', fontsize=14, labelpad=15)  # 预测标签
    ax.set_ylabel('True Label', fontsize=14, labelpad=15)       # 真实标签
    ax.set_title('Classification Accuracy (%)', fontsize=16, pad=20)  # 分类准确率
    
    # 调整颜色条标签
    cbar = ax.collections[0].colorbar
    cbar.set_label('Accuracy (%)', fontsize=12, labelpad=10)
    
    # 设置刻度标签
    ax.set_xticklabels(class_names, fontsize=12, rotation=0)
    ax.set_yticklabels(class_names, fontsize=12, rotation=0)
    
    # 保存文件
    plt.savefig(os.path.join(args.output_dir, f'{mode}_confusion_matrix.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 打印分类报告（修正中文为英文）
    report = classification_report(y_true, y_pred, target_names=class_names, digits=4)
    print("\nClassification Report:")  # 修改为英文
    print(report)
    
    # 保存分类报告到文件（文件名保持英文）
    with open(os.path.join(args.output_dir, f'{mode}_classification_report.txt'), 'w') as f:
        f.write(report)

# 主函数
def main():
    # 解析参数
    args = parse_args()
    
    # 创建输出和模型保存目录
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.model_dir, exist_ok=True)
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 准备数据加载器
    train_loader, val_loader, test_loader = prepare_dataloaders(
        args.data_dir,
        batch_size=args.batch_size,
        apply_augmentation=not args.no_augmentation,
        apply_scaling=not args.no_scaling
    )
    
    # 创建模型配置和模型
    config = Config(args)
    model = Model(config).to(device)
    
    # 打印模型结构
    print(model)
    
    # 设置损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5)
    
    # 训练循环
    train_losses = []
    train_accs = []
    val_losses = []
    val_accs = []
    best_val_acc = 0
    
    print("\n开始训练...")
    for epoch in range(1, args.epochs + 1):
        # 训练
        train_loss, train_acc = train(model, train_loader, val_loader, criterion, optimizer, scheduler, config, args.model_dir)
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        
        # 验证
        val_loss, val_acc, _, _ = evaluate(model, val_loader, criterion, device)
        val_losses.append(val_loss)
        val_accs.append(val_acc)
        
        # 保存最佳模型
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), os.path.join(args.model_dir, 'best_model.pth'))
            print(f"保存最佳模型，验证准确率: {val_acc*100:.2f}%")
        
        # 每10个epoch保存一次训练曲线
        if epoch % 10 == 0 or epoch == args.epochs:
            plot_training_curves(train_losses, train_accs, val_losses, val_accs, args)
    
    # 绘制最终训练曲线
    plot_training_curves(train_losses, train_accs, val_losses, val_accs, args)
    
    # 加载最佳模型
    model.load_state_dict(torch.load(os.path.join(args.model_dir, 'best_model.pth')))
    
    # 在测试集上评估模型
    print("\n在测试集上评估最佳模型...")
    test_loss, test_acc, y_true, y_pred = evaluate(model, test_loader, criterion, device, prefix="Test")
    print(f"\n测试准确率: {test_acc*100:.2f}%")
    
    # 绘制并保存混淆矩阵
    plot_confusion_matrix(y_true, y_pred, args, mode='test')
    
    print(f"\n训练完成! 最佳验证准确率: {best_val_acc*100:.2f}%, 测试准确率: {test_acc*100:.2f}%")
    print(f"结果保存在: {args.output_dir}")
    print(f"模型保存在: {args.model_dir}")

if __name__ == "__main__":
    main()
