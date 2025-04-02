import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import random
from typing import Tuple, List, Optional
import pickle
import time

class FallDetectionDataset(Dataset):
    """跌倒检测数据集"""
    
    def __init__(self, features, labels):
        """
        初始化数据集
        
        Args:
            features: 特征数据，形状为 (n_samples, seq_len, n_features)
            labels: 标签数据，形状为 (n_samples, n_classes)
        """
        self.features = features
        self.labels = labels
    
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        # 将数据转换为PyTorch张量
        feature = torch.FloatTensor(self.features[idx])
        
        # 将one-hot编码的标签转换为类别索引
        if self.labels[idx].shape[0] > 1:  # 如果是one-hot编码
            label_idx = np.argmax(self.labels[idx]).item()
        else:
            # 如果已经是类别索引
            label_idx = int(self.labels[idx])
        
        # 创建标量长整型张量
        label = torch.tensor(label_idx, dtype=torch.long)
        
        return feature, label

def load_data(data_dir, dataset_type, use_cache=True, cache_dir="processed_data"):
    """
    加载数据并缓存处理后的数据
    
    Args:
        data_dir: 数据所在目录
        dataset_type: 数据集类型 ('train', 'val', 'test')
        use_cache: 是否使用缓存
        cache_dir: 缓存目录
    
    Returns:
        features, labels: 加载的特征和标签数据
    """
    # 如果使用缓存，检查是否存在缓存文件
    cache_path = None
    if use_cache:
        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir)
        cache_path = os.path.join(cache_dir, f"{dataset_type}_data.pkl")
        if os.path.exists(cache_path):
            print(f"加载缓存数据: {cache_path}")
            try:
                start_time = time.time()
                with open(cache_path, 'rb') as f:
                    data = pickle.load(f)
                print(f"加载完成，耗时: {time.time() - start_time:.2f}秒")
                return data['features'], data['labels']
            except Exception as e:
                print(f"加载缓存失败: {e}，将从原始文件加载数据")
    
    # 如果没有缓存或者不使用缓存，加载原始数据
    print(f"加载原始数据: {dataset_type}")
    start_time = time.time()
    
    # 构建数据文件路径
    # 检查是否已经指向Three Classes目录
    if os.path.basename(data_dir) == "Three Classes":
        three_classes_dir = data_dir
    else:
        three_classes_dir = os.path.join(data_dir, "Three Classes")
    
    # 检查目录是否存在
    if not os.path.exists(three_classes_dir):
        raise FileNotFoundError(f"数据目录不存在: {three_classes_dir}")
    
    # 加载二进制文件
    features_path = os.path.join(three_classes_dir, f"x_{dataset_type}")
    labels_path = os.path.join(three_classes_dir, f"y_{dataset_type}")
    
    print(f"加载特征文件: {features_path}")
    print(f"加载标签文件: {labels_path}")
    
    # 检查文件是否存在
    if not os.path.exists(features_path):
        raise FileNotFoundError(f"特征文件不存在: {features_path}")
    if not os.path.exists(labels_path):
        raise FileNotFoundError(f"标签文件不存在: {labels_path}")
    
    try:
        # 从二进制文件加载数据
        features = np.fromfile(features_path, dtype=np.float32)
        labels = np.fromfile(labels_path, dtype=np.int8)
        
        # 重新调整特征形状
        n_features = 6  # 假设有6个特征通道
        seq_len = 256   # 假设序列长度为256
        features = features.reshape(-1, seq_len, n_features)
        
        # 重新调整标签形状
        n_classes = 3   # 假设有3个类别
        labels = labels.reshape(-1, n_classes)
        
        print(f"加载完成，耗时: {time.time() - start_time:.2f}秒")
        print(f"特征形状: {features.shape}, 标签形状: {labels.shape}")
        
        # 如果使用缓存，保存处理后的数据
        if use_cache and cache_path:
            try:
                print(f"缓存处理后的数据: {cache_path}")
                with open(cache_path, 'wb') as f:
                    pickle.dump({'features': features, 'labels': labels}, f)
            except Exception as e:
                print(f"保存缓存失败: {e}，但不影响后续处理")
        
        return features, labels
    except Exception as e:
        raise RuntimeError(f"加载数据失败: {e}")

def apply_standardization(x_train: np.ndarray, x_val: np.ndarray, x_test: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    对数据进行标准化处理
    
    Args:
        x_train: 训练集特征
        x_val: 验证集特征
        x_test: 测试集特征
        
    Returns:
        标准化后的 x_train, x_val, x_test
    """
    # 获取数据维度
    n_samples, seq_len, n_features = x_train.shape
    
    # 将数据重塑为2D以进行标准化 (n_samples*seq_len, n_features)
    x_train_reshaped = x_train.reshape(-1, n_features)
    x_val_reshaped = x_val.reshape(-1, n_features)
    x_test_reshaped = x_test.reshape(-1, n_features)
    
    # 初始化标准化器
    scaler = StandardScaler()
    
    # 只用训练集拟合标准化器
    scaler.fit(x_train_reshaped)
    
    # 对所有数据集应用标准化
    x_train_scaled = scaler.transform(x_train_reshaped).reshape(x_train.shape)
    x_val_scaled = scaler.transform(x_val_reshaped).reshape(x_val.shape)
    x_test_scaled = scaler.transform(x_test_reshaped).reshape(x_test.shape)
    
    print(f"数据标准化完成")
    
    return x_train_scaled, x_val_scaled, x_test_scaled

def augment_data(x_train: np.ndarray, y_train: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    对训练数据进行增强
    
    Args:
        x_train: 训练集特征
        y_train: 训练集标签
        
    Returns:
        增强后的 x_train, y_train
    """
    # 获取数据维度
    n_samples, seq_len, n_features = x_train.shape
    
    # 获取类别索引
    class_indices = np.argmax(y_train, axis=1)
    
    # 统计各类别样本数
    class_counts = np.bincount(class_indices)
    print(f"增强前类别分布: {class_counts}")
    
    # 找出样本最多的类别
    max_class = np.argmax(class_counts)
    max_count = class_counts[max_class]
    
    augmented_features = []
    augmented_labels = []
    
    # 对除了样本最多的类别外的其他类别进行增强
    for class_idx in range(len(class_counts)):
        if class_idx == max_class:
            continue
            
        # 找出当前类别的所有样本
        class_samples_idx = np.where(class_indices == class_idx)[0]
        
        # 计算需要增强的样本数量
        n_augment = max_count - class_counts[class_idx]
        
        # 如果需要增强的样本数量太多，则限制为原样本数量的5倍
        if n_augment > 5 * class_counts[class_idx]:
            n_augment = 5 * class_counts[class_idx]
            
        print(f"对类别 {class_idx} 增强 {n_augment} 个样本")
        
        for _ in range(n_augment):
            # 随机选择一个样本
            sample_idx = random.choice(class_samples_idx)
            sample = x_train[sample_idx].copy()
            
            # 应用随机增强技术
            aug_type = random.randint(0, 3)
            
            if aug_type == 0:
                # 添加随机噪声
                noise = np.random.normal(0, 0.01, sample.shape)
                sample = sample + noise
            elif aug_type == 1:
                # 时间扭曲
                warp_factor = random.uniform(0.9, 1.1)
                indices = np.arange(seq_len)
                warped_indices = np.clip(np.round(indices * warp_factor), 0, seq_len - 1).astype(int)
                sample = sample[warped_indices]
            elif aug_type == 2:
                # 尺度变换
                scale_factor = random.uniform(0.9, 1.1)
                sample = sample * scale_factor
            elif aug_type == 3:
                # 随机翻转部分通道
                channels_to_flip = random.sample(range(n_features), random.randint(1, n_features // 2))
                for channel in channels_to_flip:
                    sample[:, channel] = -sample[:, channel]
            
            # 添加到增强数据中
            augmented_features.append(sample)
            augmented_labels.append(y_train[sample_idx])
    
    if augmented_features:
        # 将增强后的数据添加到原始数据中
        x_train_augmented = np.vstack([x_train, np.array(augmented_features)])
        y_train_augmented = np.vstack([y_train, np.array(augmented_labels)])
        
        # 洗牌
        indices = np.arange(len(x_train_augmented))
        np.random.shuffle(indices)
        x_train_augmented = x_train_augmented[indices]
        y_train_augmented = y_train_augmented[indices]
        
        # 统计增强后的类别分布
        augmented_class_indices = np.argmax(y_train_augmented, axis=1)
        augmented_class_counts = np.bincount(augmented_class_indices)
        print(f"增强后类别分布: {augmented_class_counts}")
        
        return x_train_augmented, y_train_augmented
    else:
        return x_train, y_train

def create_dataloaders(
    x_train: np.ndarray, 
    y_train: np.ndarray, 
    x_val: np.ndarray, 
    y_val: np.ndarray, 
    x_test: np.ndarray, 
    y_test: np.ndarray, 
    batch_size: int = 64
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    创建PyTorch数据加载器
    
    Args:
        x_train: 训练集特征
        y_train: 训练集标签
        x_val: 验证集特征
        y_val: 验证集标签
        x_test: 测试集特征
        y_test: 测试集标签
        batch_size: 批处理大小
        
    Returns:
        train_loader, val_loader, test_loader
    """
    # 创建数据集
    train_dataset = FallDetectionDataset(x_train, y_train)
    val_dataset = FallDetectionDataset(x_val, y_val)
    test_dataset = FallDetectionDataset(x_test, y_test)
    
    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    
    print(f"数据加载器创建完成:")
    print(f"训练集: {len(train_dataset)} 样本, {len(train_loader)} 批次")
    print(f"验证集: {len(val_dataset)} 样本, {len(val_loader)} 批次")
    print(f"测试集: {len(test_dataset)} 样本, {len(test_loader)} 批次")
    
    return train_loader, val_loader, test_loader

def prepare_dataloaders(
    data_dir: str, 
    batch_size: int = 64, 
    apply_augmentation: bool = True, 
    apply_scaling: bool = True,
    use_cache: bool = True,
    processed_cache_dir: str = None
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    加载、预处理数据并创建数据加载器
    
    Args:
        data_dir: 数据目录路径
        batch_size: 批处理大小
        apply_augmentation: 是否应用数据增强
        apply_scaling: 是否应用数据标准化
        use_cache: 是否使用缓存数据
        processed_cache_dir: 处理后数据的缓存目录
        
    Returns:
        train_loader, val_loader, test_loader
    """
    # 确定缓存目录
    if processed_cache_dir is None:
        processed_cache_dir = os.path.join(os.path.dirname(data_dir), "processed_data")
    os.makedirs(processed_cache_dir, exist_ok=True)
    
    # 构建缓存文件名，包含处理参数信息
    cache_filename = f"processed_data_aug_{apply_augmentation}_scale_{apply_scaling}.pkl"
    cache_file = os.path.join(processed_cache_dir, cache_filename)
    
    # 尝试从缓存加载处理后的数据
    if use_cache and os.path.exists(cache_file):
        print(f"发现处理后的缓存文件: {cache_file}，尝试加载...")
        try:
            start_time = time.time()
            with open(cache_file, 'rb') as f:
                data = pickle.load(f)
            
            x_train = data['x_train']
            y_train = data['y_train']
            x_val = data['x_val']
            y_val = data['y_val']
            x_test = data['x_test']
            y_test = data['y_test']
            
            load_time = time.time() - start_time
            print(f"成功从缓存加载处理后的数据！耗时: {load_time:.2f}秒")
            
            # 打印类别分布情况
            train_class_indices = np.argmax(y_train, axis=1)
            val_class_indices = np.argmax(y_val, axis=1)
            test_class_indices = np.argmax(y_test, axis=1)
            
            train_class_counts = np.bincount(train_class_indices)
            val_class_counts = np.bincount(val_class_indices)
            test_class_counts = np.bincount(test_class_indices)
            
            print("\n类别分布情况:")
            print(f"训练集: BKG={train_class_counts[0]}, ALERT={train_class_counts[1]}, FALL={train_class_counts[2]}")
            print(f"验证集: BKG={val_class_counts[0]}, ALERT={val_class_counts[1]}, FALL={val_class_counts[2]}")
            print(f"测试集: BKG={test_class_counts[0]}, ALERT={test_class_counts[1]}, FALL={test_class_counts[2]}")
            
            # 创建数据加载器
            return create_dataloaders(x_train, y_train, x_val, y_val, x_test, y_test, batch_size)
        except Exception as e:
            print(f"加载缓存失败: {e}，将重新处理数据")
    
    # 如果没有缓存或加载失败，从原始文件开始处理
    # 加载数据
    x_train, y_train = load_data(data_dir, 'train', use_cache=use_cache)
    x_val, y_val = load_data(data_dir, 'val', use_cache=use_cache)
    x_test, y_test = load_data(data_dir, 'test', use_cache=use_cache)
    
    # 应用标准化
    if apply_scaling:
        x_train, x_val, x_test = apply_standardization(x_train, x_val, x_test)
    
    # 应用数据增强
    if apply_augmentation:
        print("应用数据增强...")
        x_train, y_train = augment_data(x_train, y_train)
        print(f"增强后 x_train: {x_train.shape}, y_train: {y_train.shape}")
    
    # 保存处理后的数据到缓存
    try:
        print(f"正在将处理后的数据保存到缓存...")
        data = {
            'x_train': x_train,
            'y_train': y_train,
            'x_val': x_val,
            'y_val': y_val,
            'x_test': x_test,
            'y_test': y_test
        }
        
        start_time = time.time()
        with open(cache_file, 'wb') as f:
            pickle.dump(data, f, protocol=4)
        save_time = time.time() - start_time
        print(f"处理后的数据成功保存到缓存: {cache_file}，耗时: {save_time:.2f}秒")
    except Exception as e:
        print(f"保存缓存失败: {e}")
    
    # 创建数据加载器
    return create_dataloaders(x_train, y_train, x_val, y_val, x_test, y_test, batch_size)

def visualize_sample(x, y, title="样本可视化"):
    """可视化一个样本的时间序列数据"""
    sensor_names = ["Accelerometer X", "Accelerometer Y", "Accelerometer Z", "Gyroscope X", "Gyroscope Y", "Gyroscope Z"]
    class_names = ["BKG", "ALERT", "FALL"]
    
    # 获取类别
    if isinstance(y, np.ndarray) and y.ndim > 0 and y.shape[0] > 1:
        # 如果是独热编码
        class_idx = np.argmax(y)
    else:
        # 如果是单个类别索引
        class_idx = int(y)
    
    # 创建子图
    fig, axs = plt.subplots(6, 1, figsize=(10, 12))
    fig.suptitle(f"{title} - Class: {class_names[class_idx]}", fontsize=16)
    
    # 绘制每个传感器的数据
    for i in range(6):
        axs[i].plot(x[:, i])
        axs[i].set_title(sensor_names[i])
        axs[i].set_xlabel("Time Step")
        axs[i].set_ylabel("Value")
        axs[i].grid(True)
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.92)
    
    return fig

if __name__ == "__main__":
    data_dir = os.path.abspath("data")
    train_loader, val_loader, test_loader = prepare_dataloaders(
        data_dir, batch_size=64, apply_augmentation=True, apply_scaling=True
    )
    
    # 可视化一个样本
    x_batch, y_batch = next(iter(train_loader))
    print(f"批次形状: x={x_batch.shape}, y={y_batch.shape}")
    
    # 转换为numpy以便可视化
    x_sample = x_batch[0].numpy()
    y_sample = y_batch[0].item()
    
    # 可视化
    fig = visualize_sample(x_sample, y_sample, "训练集样本")
    os.makedirs("output", exist_ok=True)
    plt.savefig("output/sample_visualization.png")
    print("样本可视化已保存到 output/sample_visualization.png")
