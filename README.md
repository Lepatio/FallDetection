# 跌倒检测系统

这是一个基于深度学习的跌倒检测系统，使用传感器数据进行三分类：正常活动(BKG)、警告状态(ALERT)和跌倒(FALL)。

## 项目结构

```
├── data/                  # 数据目录
│   └── Three Classes/     # 三分类数据集
│       ├── x_train        # 训练特征
│       ├── y_train        # 训练标签
│       ├── x_val          # 验证特征
│       ├── y_val          # 验证标签
│       ├── x_test         # 测试特征
│       └── y_test         # 测试标签
├── model/                 # 模型定义
│   └── TexFilter.py       # TexFilter模型
├── src/                   # 源代码
│   ├── data_preprocessing.py  # 数据预处理
│   └── trainer.py         # 训练器
├── checkpoints/           # 模型保存目录
├── output/                # 输出目录
├── processed_data/        # 处理后的数据缓存
├── main.py                # 主程序入口
└── requirements.txt       # 依赖项
```

## 安装依赖

```bash
pip install -r requirements.txt
```

## 使用方法

### 完整工作流

```bash
python main.py
```

这将执行数据可视化、模型训练和模型评估的完整流程。

### 单独模式

#### 数据可视化

```bash
python main.py --mode visualize
```

#### 模型训练

```bash
python main.py --mode train
```

#### 模型测试

```bash
python main.py --mode test
```

### 主要参数

- `--mode`: 运行模式 (visualize, train, test, all)
- `--batch_size`: 批处理大小
- `--epochs`: 训练轮数
- `--lr`: 学习率
- `--data_dir`: 数据目录
- `--output_dir`: 输出目录
- `--no_cache`: 禁用数据缓存
- `--no_augmentation`: 禁用数据增强
- `--no_scaling`: 禁用数据标准化

## 故障排除

如果遇到问题，请检查：

1. 数据文件格式是否正确
2. 路径是否正确
3. 必要的目录是否存在
4. 日志文件中的错误信息

## 数据格式

- 特征(x文件): 形状为 [n_samples, 256, 6] 的传感器数据
  - n_samples: 样本数量
  - 256: 序列长度(时间步)
  - 6: 特征通道数(3加速度计 + 3陀螺仪)

- 标签(y文件): 形状为 [n_samples, 3] 的独热编码标签
  - n_samples: 样本数量
  - 3: 类别数(BKG, ALERT, FALL) 