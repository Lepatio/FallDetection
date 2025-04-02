import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.gridspec import GridSpec

# 设置中文字体支持
plt.rcParams['font.sans-serif'] = ['SimHei']  # 黑体
plt.rcParams['axes.unicode_minus'] = False  # 正确显示负号

# 设置输出目录
OUTPUT_DIR = os.path.abspath("output")
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

def load_x(file_name):
    """加载特征数据"""
    temp = np.memmap(file_name, dtype='float32', mode='r')
    X = np.reshape(temp, [-1, 256, 6])
    return X

def load_y(file_name):
    """加载标签数据"""
    n_classes = int(os.path.basename(file_name).split('_')[-1])
    temp = np.memmap(file_name, dtype='int8', mode='r')
    file_size = os.path.getsize(file_name)
    n_samples = file_size // n_classes
    Y = np.reshape(temp, [n_samples, n_classes])
    return Y

def find_samples_by_class(x_data, y_data):
    """根据标签找到每个类别的样本"""
    n_samples = y_data.shape[0]
    
    # 创建一个字典来存储每个类别的样本索引
    class_indices = {
        'BKG': [],   # 类别0
        'ALERT': [], # 类别1
        'FALL': []   # 类别2
    }
    
    # 遍历所有样本，根据标签将其分类
    for i in range(n_samples):
        label = np.argmax(y_data[i])
        if label == 0:
            class_indices['BKG'].append(i)
        elif label == 1:
            class_indices['ALERT'].append(i)
        elif label == 2:
            class_indices['FALL'].append(i)
    
    print(f"找到 BKG 样本: {len(class_indices['BKG'])} 个")
    print(f"找到 ALERT 样本: {len(class_indices['ALERT'])} 个")
    print(f"找到 FALL 样本: {len(class_indices['FALL'])} 个")
    
    # 为每个类别随机选择一些样本
    samples_per_class = {}
    for class_name, indices in class_indices.items():
        if indices:
            # 为每个类别随机选择5个样本（如果有足够的样本）
            n_samples_to_select = min(5, len(indices))
            selected_indices = np.random.choice(indices, size=n_samples_to_select, replace=False)
            samples_per_class[class_name] = [x_data[idx] for idx in selected_indices]
    
    return samples_per_class

def visualize_class_distribution(y_data, title="三分类数据 - 类别分布"):
    """绘制类别分布柱状图"""
    print(f"绘制{title}...")
    
    # 计算每个类别的数量
    class_counts = np.sum(y_data, axis=0)
    class_names = ['BKG', 'ALERT', 'FALL']
    
    # 创建柱状图
    plt.figure(figsize=(10, 6))
    
    # 使用不同颜色表示不同类别
    colors = ['#1f77b4', '#ff7f0e', '#d62728']
    
    # 绘制柱状图
    bars = plt.bar(range(len(class_counts)), class_counts, color=colors)
    
    # 添加数值标签
    for bar, count in zip(bars, class_counts):
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width()/2.,
            height + 0.1*max(class_counts),
            f'{count}\n({count/sum(class_counts)*100:.1f}%)',
            ha='center', 
            va='bottom'
        )
    
    # 标题和标签
    plt.title(title, fontsize=16)
    plt.xlabel('类别', fontsize=14)
    plt.ylabel('样本数量', fontsize=14)
    plt.xticks(range(len(class_counts)), class_names)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # 保存图像
    plt.tight_layout()
    output_path = os.path.join(OUTPUT_DIR, 'class_distribution.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"类别分布图保存为 {output_path}")
    
    return plt

def visualize_time_series_sample(data_dict, title="三分类数据 - 时间序列"):
    """绘制每个数据集的代表性样本时间序列"""
    print(f"绘制{title}...")
    
    # 传感器名称
    sensor_names = ["加速度计X", "加速度计Y", "加速度计Z", "陀螺仪X", "陀螺仪Y", "陀螺仪Z"]
    
    # 创建一个2x3网格图
    fig = plt.figure(figsize=(18, 10))
    gs = GridSpec(2, 3, figure=fig)
    
    # 为每个传感器创建子图
    for i, sensor_name in enumerate(sensor_names):
        ax = fig.add_subplot(gs[i//3, i%3])
        
        # 为每个数据集绘制时间序列
        for dataset_name, data in data_dict.items():
            # 使用第一个样本
            sample_idx = 0
            ax.plot(data[sample_idx, :, i], label=dataset_name, linewidth=1.5)
        
        ax.set_title(sensor_name, fontsize=14)
        ax.set_xlabel('时间步', fontsize=12)
        ax.set_ylabel('值', fontsize=12)
        ax.grid(True, linestyle='--', alpha=0.7)
        
        # 仅在第一个子图上显示图例
        if i == 0:
            ax.legend()
    
    # 添加总标题
    fig.suptitle(title, fontsize=18)
    plt.tight_layout()
    fig.subplots_adjust(top=0.92)
    
    # 保存图像
    output_path = os.path.join(OUTPUT_DIR, 'time_series_sample.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"时间序列样本图保存为 {output_path}")
    
    return fig

def visualize_dataset_comparison(data_dict, title="三分类数据 - 数据集对比"):
    """比较不同数据集的统计特性"""
    print(f"绘制{title}...")
    
    # 传感器名称
    sensor_names = ["加速度计X", "加速度计Y", "加速度计Z", "陀螺仪X", "陀螺仪Y", "陀螺仪Z"]
    
    # 收集每个数据集每个传感器的统计数据
    stats = {}
    for dataset_name, data in data_dict.items():
        stats[dataset_name] = {}
        for i, sensor_name in enumerate(sensor_names):
            # 随机抽取1000个样本进行分析，以加快处理速度
            sample_indices = np.random.choice(data.shape[0], min(1000, data.shape[0]), replace=False)
            sensor_data = data[sample_indices, :, i].flatten()
            
            stats[dataset_name][sensor_name] = {
                'mean': np.mean(sensor_data),
                'std': np.std(sensor_data),
                'min': np.min(sensor_data),
                'max': np.max(sensor_data),
                'data': sensor_data
            }
    
    # 创建一个2x3的网格
    fig = plt.figure(figsize=(18, 12))
    gs = GridSpec(3, 2, figure=fig)
    
    # 为每个传感器创建子图和箱线图
    for i, sensor_name in enumerate(sensor_names):
        ax = fig.add_subplot(gs[i//2, i%2])
        
        # 用于箱线图的数据
        box_data = []
        labels = []
        
        for dataset_name in data_dict.keys():
            box_data.append(stats[dataset_name][sensor_name]['data'])
            labels.append(dataset_name)
        
        # 绘制箱线图
        ax.boxplot(box_data, labels=labels, showfliers=False, patch_artist=True,
                   boxprops=dict(alpha=0.7),
                   medianprops=dict(color='black', linewidth=1.5))
        
        ax.set_title(sensor_name, fontsize=14)
        ax.set_ylabel('值', fontsize=12)
        ax.grid(True, linestyle='--', alpha=0.5)
        
        # 添加均值和标准差标签
        for j, dataset_name in enumerate(data_dict.keys()):
            mean = stats[dataset_name][sensor_name]['mean']
            std = stats[dataset_name][sensor_name]['std']
            ax.text(j+1, ax.get_ylim()[0] + 0.05*(ax.get_ylim()[1] - ax.get_ylim()[0]),
                   f'均值: {mean:.4f}\n标准差: {std:.4f}',
                   ha='center', va='bottom', fontsize=9)
    
    # 添加总标题
    fig.suptitle(title, fontsize=18)
    plt.tight_layout()
    fig.subplots_adjust(top=0.93, hspace=0.3)
    
    # 保存图像
    output_path = os.path.join(OUTPUT_DIR, 'dataset_comparison.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"数据集对比图保存为 {output_path}")
    
    return fig

def visualize_classes_comparison(samples_per_class, title="三分类数据 - 类别比较"):
    """将不同类别的样本绘制在同一视图中进行比较"""
    print(f"绘制{title}...")
    
    # 传感器名称
    sensor_names = ["加速度计X", "加速度计Y", "加速度计Z", "陀螺仪X", "陀螺仪Y", "陀螺仪Z"]
    
    # 类别颜色和样式
    class_colors = {
        'BKG': '#1f77b4',    # 蓝色
        'ALERT': '#ff7f0e',  # 橙色
        'FALL': '#d62728'    # 红色
    }
    
    # 创建一个2x3网格图
    fig = plt.figure(figsize=(20, 12))
    gs = GridSpec(2, 3, figure=fig)
    
    # 为每个传感器创建子图
    for i, sensor_name in enumerate(sensor_names):
        ax = fig.add_subplot(gs[i//3, i%3])
        
        # 为每个类别绘制样本
        for class_name, samples in samples_per_class.items():
            # 绘制第一个样本的完整线条
            if samples:
                ax.plot(samples[0][:, i], 
                       color=class_colors[class_name], 
                       label=f'{class_name}', 
                       linewidth=2)
                
                # 绘制其他样本的透明线条
                for j in range(1, len(samples)):
                    ax.plot(samples[j][:, i], 
                           color=class_colors[class_name], 
                           alpha=0.3,
                           linewidth=1)
        
        ax.set_title(sensor_name, fontsize=14)
        ax.set_xlabel('时间步', fontsize=12)
        ax.set_ylabel('值', fontsize=12)
        ax.grid(True, linestyle='--', alpha=0.7)
        
        # 仅在第一个子图上显示图例
        if i == 0:
            ax.legend(fontsize=12)
    
    # 添加解释文本
    fig.text(0.5, 0.01, 
             "注: 每种类别显示5个样本，其中1个为不透明主线，4个为半透明辅助线。\n"
             "BKG = 背景/正常活动(0), ALERT = 警告状态(1), FALL = 跌倒事件(2)", 
             ha='center', fontsize=12)
    
    # 添加总标题
    fig.suptitle(title, fontsize=18)
    plt.tight_layout()
    fig.subplots_adjust(top=0.92, bottom=0.08)
    
    # 保存图像
    output_path = os.path.join(OUTPUT_DIR, 'class_comparison.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"类别比较图保存为 {output_path}")
    
    return fig

def display_data_preview(data_dict):
    """显示各数据集的前50行数据预览"""
    for dataset_name, data in data_dict.items():
        # 选择第一个样本
        sample_idx = 0
        df = pd.DataFrame(
            data[sample_idx, :50, :],
            columns=["加速度计X", "加速度计Y", "加速度计Z", "陀螺仪X", "陀螺仪Y", "陀螺仪Z"]
        )
        
        print(f"\n{dataset_name} - 样本1的前50个时间步:")
        print(df)

def main():
    """主函数"""
    print("开始三分类数据可视化...")
    
    # 设置数据路径
    base_dir = os.path.abspath("data")
    three_classes_dir = os.path.join(base_dir, "Three Classes")
    
    # 加载数据
    print("\n加载数据...")
    x_train = load_x(os.path.join(three_classes_dir, "x_train_3"))
    y_train = load_y(os.path.join(three_classes_dir, "y_train_3"))
    x_val = load_x(os.path.join(three_classes_dir, "x_val_3"))
    x_test = load_x(os.path.join(three_classes_dir, "x_test_3"))
    
    print(f"x_train 形状: {x_train.shape}")
    print(f"y_train 形状: {y_train.shape}")
    print(f"x_val 形状: {x_val.shape}")
    print(f"x_test 形状: {x_test.shape}")
    
    # 数据集预览
    x_data = {
        "训练集": x_train,
        "验证集": x_val,
        "测试集": x_test
    }
    
    print("\n数据预览:")
    display_data_preview(x_data)
    
    # 1. 绘制类别分布柱状图
    plt.figure(1)
    visualize_class_distribution(y_train)
    
    # 2. 绘制时间序列图
    plt.figure(2)
    visualize_time_series_sample(x_data)
    
    # 3. 比较train/test/val数据
    plt.figure(3)
    visualize_dataset_comparison(x_data)
    
    # 4. 根据类别找到样本并绘制类别比较图
    samples_per_class = find_samples_by_class(x_train, y_train)
    plt.figure(4)
    visualize_classes_comparison(samples_per_class)
    
    print("\n可视化完成!")

if __name__ == "__main__":
    main() 