import os
import sys
import torch
import argparse
import numpy as np
import matplotlib.pyplot as plt
import logging

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("train.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("FallDetection")

# 添加当前目录到路径
sys.path.append(os.path.abspath("."))

# 导入自定义模块
from src.data_preprocessing import prepare_dataloaders, visualize_sample
from src.trainer import train, evaluate, plot_confusion_matrix
from model.TexFilter import Model

class Config:
    def __init__(self, args):
        # 必要的模型参数
        self.seq_len = args.seq_len  # 序列长度
        self.pred_len = 3  # 3分类预测
        self.enc_in = 6  # 6个特征通道
        
        # 嵌入和隐藏层参数
        self.embed_size = args.embed_size  # 嵌入维度
        self.d_model = args.embed_size  # d_model与embed_size相同
        self.hidden_size = args.hidden_size  # 隐藏层维度
        self.dropout = args.dropout  # dropout比率
        
        # Transformer参数
        self.n_heads = args.n_heads  # 多头注意力头数
        self.d_ff = args.hidden_size * 2  # 前馈网络维度
        self.e_layers = 3  # 编码器层数
        
        # 训练参数
        self.epochs = args.epochs  # 训练轮数
        self.patience = args.patience  # 提前停止耐心值
        self.device = torch.device(args.device)  # 训练设备
        
        # 模型文件名
        self.model_name = f"model_{self.seq_len}_{self.embed_size}_{self.hidden_size}_{self.dropout}.pth"

def parse_args():
    parser = argparse.ArgumentParser(description="跌倒检测系统")
    parser.add_argument("--mode", type=str, choices=["train", "test", "visualize", "all"], default="all",
                        help="运行模式: 训练(train), 测试(test), 可视化(visualize), 全部(all)")
    parser.add_argument("--batch_size", type=int, default=64, help="批处理大小")
    parser.add_argument("--epochs", type=int, default=50, help="训练轮数")
    parser.add_argument("--patience", type=int, default=10, help="提前停止耐心值，连续多少个epoch无改善则停止")
    parser.add_argument("--lr", type=float, default=0.001, help="学习率")
    parser.add_argument("--weight_decay", type=float, default=1e-5, help="权重衰减")
    parser.add_argument("--model_dir", type=str, default="checkpoints", help="模型保存目录")
    parser.add_argument("--data_dir", type=str, default="data", help="数据目录")
    parser.add_argument("--output_dir", type=str, default="output", help="输出目录")
    parser.add_argument("--seq_len", type=int, default=256, help="序列长度")
    parser.add_argument("--embed_size", type=int, default=128, help="嵌入维度")
    parser.add_argument("--hidden_size", type=int, default=256, help="隐藏层维度")
    parser.add_argument("--dropout", type=float, default=0.1, help="dropout比率")
    parser.add_argument("--n_heads", type=int, default=4, help="注意力头数")
    parser.add_argument("--no_augmentation", action="store_true", help="禁用数据增强")
    parser.add_argument("--no_scaling", action="store_true", help="禁用数据标准化")
    parser.add_argument("--model_path", type=str, default=None, help="测试模式下的模型路径")
    parser.add_argument("--sample_id", type=int, default=0, help="可视化模式下要显示的样本ID")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", 
                        help="设备类型 (cuda/cpu)")
    parser.add_argument("--no_cache", action="store_true", help="禁用数据缓存")
    parser.add_argument("--processed_cache_dir", type=str, default=None, 
                        help="处理后数据的缓存目录，默认为data同级的processed_data目录")
    
    return parser.parse_args()

def train_model(args):
    """训练模型"""
    # 创建输出和模型保存目录
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.model_dir, exist_ok=True)
    
    # 设置设备
    device = torch.device(args.device)
    print(f"Using device: {device}")
    
    # 打印缓存使用情况
    if args.no_cache:
        print("Cache is disabled, data will be loaded from original files")
    else:
        print(f"Cache is enabled, processed data will be saved to '{args.processed_cache_dir or os.path.join(os.path.dirname(args.data_dir), 'processed_data')}'")
    
    # 准备数据加载器
    train_loader, val_loader, test_loader = prepare_dataloaders(
        args.data_dir,
        batch_size=args.batch_size,
        apply_augmentation=not args.no_augmentation,
        apply_scaling=not args.no_scaling,
        use_cache=not args.no_cache,
        processed_cache_dir=args.processed_cache_dir
    )
    
    # 创建模型配置和模型
    config = Config(args)
    model = Model(config).to(device)
    
    # 打印模型结构
    print(model)
    
    # 设置损失函数和优化器
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5)
    
    # 训练循环
    train_losses = []
    train_accs = []
    val_losses = []
    val_accs = []
    best_val_acc = 0
    
    print("\nStarting training...")
    for epoch in range(1, args.epochs + 1):
        # 训练 - 使用新的调用方式，提供所有必要参数
        train_loss, train_acc = train(
            model, 
            train_loader, 
            val_loader=val_loader, 
            criterion=criterion, 
            optimizer=optimizer, 
            scheduler=scheduler, 
            config=config, 
            save_dir=args.model_dir
        )
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        
        # 验证
        val_loss, val_acc, _, _ = evaluate(model, val_loader, criterion, device)
        val_losses.append(val_loss)
        val_accs.append(val_acc)
        
        # 更新学习率
        scheduler.step(val_loss)
        
        # 保存最佳模型
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), os.path.join(args.model_dir, 'best_model.pth'))
            print(f"Saved best model, validation accuracy: {val_acc*100:.2f}%")
        
        # 每10个epoch保存一次训练曲线
        if epoch % 10 == 0 or epoch == args.epochs:
            plot_training_curves(train_losses, train_accs, val_losses, val_accs, args)
    
    # 绘制最终训练曲线
    plot_training_curves(train_losses, train_accs, val_losses, val_accs, args)
    
    # 加载最佳模型
    model.load_state_dict(torch.load(os.path.join(args.model_dir, 'best_model.pth')))
    
    # 在测试集上评估模型
    print("\nEvaluating best model on test set...")
    test_loss, test_acc, y_true, y_pred = evaluate(model, test_loader, criterion, device, prefix="Test")
    print(f"\nTest accuracy: {test_acc*100:.2f}%")
    
    # 绘制并保存混淆矩阵
    plot_confusion_matrix(y_true, y_pred, args)
    
    print(f"\nTraining completed! Best validation accuracy: {best_val_acc*100:.2f}%, Test accuracy: {test_acc*100:.2f}%")
    print(f"Results saved to: {args.output_dir}")
    print(f"Model saved to: {args.model_dir}")
    
    return model

def test_model(args, model=None):
    """测试模型"""
    # 设置设备
    device = torch.device(args.device)
    print(f"Using device: {device}")
    
    # 如果没有传入模型，就根据路径加载
    if model is None:
        # 确定模型路径
        if args.model_path:
            model_path = args.model_path
        else:
            model_path = os.path.join(args.model_dir, 'best_model.pth')
        
        if not os.path.exists(model_path):
            print(f"Error: Model file {model_path} does not exist!")
            return None
        
        # 创建模型配置和模型
        config = Config(args)
        model = Model(config).to(device)
        
        # 加载模型权重
        model.load_state_dict(torch.load(model_path, map_location=device))
        print(f"Model loaded from: {model_path}")
    else:
        # 如果传入了模型，确保它在正确的设备上
        if next(model.parameters()).device != device:
            model = model.to(device)
        print("Using provided model for testing")
    
    # 准备数据加载器
    _, _, test_loader = prepare_dataloaders(
        args.data_dir,
        batch_size=args.batch_size,
        apply_augmentation=False,  # 测试时不需要数据增强
        apply_scaling=not args.no_scaling,
        use_cache=not args.no_cache,
        processed_cache_dir=args.processed_cache_dir
    )
    
    # 设置损失函数
    criterion = torch.nn.CrossEntropyLoss()
    
    # 在测试集上评估模型
    print("Evaluating model on test set...")
    test_loss, test_acc, y_true, y_pred = evaluate(model, test_loader, criterion, device, prefix="Test")
    print(f"\nTest accuracy: {test_acc*100:.2f}%")
    
    # 绘制并保存混淆矩阵
    os.makedirs(args.output_dir, exist_ok=True)
    plot_confusion_matrix(y_true, y_pred, args)
    
    # 计算每个类别的精确度和召回率
    class_names = ['BKG', 'ALERT', 'FALL']
    from sklearn.metrics import classification_report
    report = classification_report(y_true, y_pred, target_names=class_names)
    print("\nClassification Report:")
    print(report)
    
    # 保存分类报告到文件
    with open(os.path.join(args.output_dir, 'test_classification_report.txt'), 'w') as f:
        f.write(report)
    
    return model

def visualize_data(args):
    """可视化样本数据"""
    # 打印缓存使用情况
    if args.no_cache:
        print("Cache is disabled, data will be loaded from original files")
    else:
        print(f"Cache is enabled, data will be loaded from cache if available")
    
    # 准备数据加载器
    train_loader, val_loader, test_loader = prepare_dataloaders(
        args.data_dir,
        batch_size=args.batch_size,
        apply_augmentation=False,  # 可视化时不需要数据增强
        apply_scaling=not args.no_scaling,
        use_cache=not args.no_cache,
        processed_cache_dir=args.processed_cache_dir
    )
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 从训练集、验证集和测试集中各获取一个批次
    train_batch = next(iter(train_loader))
    val_batch = next(iter(val_loader))
    test_batch = next(iter(test_loader))
    
    # 可视化训练集样本
    sample_id = min(args.sample_id, len(train_batch[0])-1)
    x_train_sample = train_batch[0][sample_id].numpy()
    y_train_sample = train_batch[1][sample_id].item()
    
    fig_train = visualize_sample(x_train_sample, y_train_sample, "Training Sample")
    plt.savefig(os.path.join(args.output_dir, "train_sample.png"))
    plt.close(fig_train)
    
    # 可视化验证集样本
    sample_id = min(args.sample_id, len(val_batch[0])-1)
    x_val_sample = val_batch[0][sample_id].numpy()
    y_val_sample = val_batch[1][sample_id].item()
    
    fig_val = visualize_sample(x_val_sample, y_val_sample, "Validation Sample")
    plt.savefig(os.path.join(args.output_dir, "val_sample.png"))
    plt.close(fig_val)
    
    # 可视化测试集样本
    sample_id = min(args.sample_id, len(test_batch[0])-1)
    x_test_sample = test_batch[0][sample_id].numpy()
    y_test_sample = test_batch[1][sample_id].item()
    
    fig_test = visualize_sample(x_test_sample, y_test_sample, "Test Sample")
    plt.savefig(os.path.join(args.output_dir, "test_sample.png"))
    plt.close(fig_test)
    
    print(f"Sample visualizations saved to directory: {args.output_dir}")
    
    # 可视化类别分布
    visualize_class_distribution(train_loader, val_loader, test_loader, args)

def visualize_class_distribution(train_loader, val_loader, test_loader, args):
    """可视化类别分布"""
    # 获取训练集标签
    train_labels = []
    for _, y in train_loader:
        train_labels.extend(y.numpy())
    
    # 获取验证集标签
    val_labels = []
    for _, y in val_loader:
        val_labels.extend(y.numpy())
    
    # 获取测试集标签
    test_labels = []
    for _, y in test_loader:
        test_labels.extend(y.numpy())
    
    # 统计各类别样本数
    train_counts = np.bincount(train_labels)
    val_counts = np.bincount(val_labels)
    test_counts = np.bincount(test_labels)
    
    # 绘制类别分布条形图
    class_names = ['BKG', 'ALERT', 'FALL']
    
    plt.figure(figsize=(12, 6))
    
    # 设置宽度和位置
    width = 0.25
    x = np.arange(len(class_names))
    
    # 绘制条形图
    plt.bar(x - width, train_counts, width, label='Training')
    plt.bar(x, val_counts, width, label='Validation')
    plt.bar(x + width, test_counts, width, label='Test')
    
    # 添加标签和标题
    plt.xlabel('Class')
    plt.ylabel('Number of Samples')
    plt.title('Class Distribution in Datasets')
    plt.xticks(x, class_names)
    plt.legend()
    
    # 在条形上方添加数值标签
    for i, count in enumerate(train_counts):
        plt.text(i - width, count + 100, str(count), ha='center')
    
    for i, count in enumerate(val_counts):
        plt.text(i, count + 100, str(count), ha='center')
    
    for i, count in enumerate(test_counts):
        plt.text(i + width, count + 100, str(count), ha='center')
    
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, "class_distribution.png"))
    plt.close()
    
    print(f"Class distribution visualization saved to: {os.path.join(args.output_dir, 'class_distribution.png')}")

def plot_training_curves(train_losses, train_accs, val_losses, val_accs, args):
    """绘制训练曲线"""
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

def run_all(args):
    """执行完整的训练-测试-可视化流程"""
    print("="*50)
    print("Starting Complete Fall Detection Workflow")
    print("="*50)
    
    # 1. 先进行数据可视化
    print("\nStep 1: Data Visualization")
    print("-"*50)
    visualize_data(args)
    
    # 2. 训练模型
    print("\nStep 2: Model Training")
    print("-"*50)
    model = train_model(args)
    
    # 3. 测试模型 - 使用已有模型，不重新训练
    print("\nStep 3: Model Testing and Evaluation")
    print("-"*50)
    # 传入已训练好的模型，防止重新训练
    test_model(args, model)
    
    print("\n="*50)
    print("Fall Detection Workflow Completed!")
    print("="*50)

if __name__ == "__main__":
    # 解析命令行参数
    args = parse_args()
    
    try:
        # 根据模式执行相应操作
        if args.mode == "train":
            train_model(args)
        elif args.mode == "test":
            test_model(args)
        elif args.mode == "visualize":
            visualize_data(args)
        elif args.mode == "all":
            run_all(args)
        else:
            print(f"错误: 未知的运行模式 {args.mode}")
    except FileNotFoundError as e:
        logger.error(f"文件不存在错误: {e}")
        print(f"\n错误: {e}")
        print("请确保数据文件存在并且路径正确。")
    except RuntimeError as e:
        logger.error(f"运行时错误: {e}")
        print(f"\n错误: {e}")
        print("程序运行过程中发生错误，请检查日志获取更多信息。")
    except Exception as e:
        logger.error(f"未预期的错误: {e}", exc_info=True)
        print(f"\n未预期的错误: {e}")
        print("程序运行过程中发生未预期的错误，请检查日志获取更多信息。") 