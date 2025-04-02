import sys
import os
import torch
import numpy as np

sys.path.append(os.path.abspath("."))
from model.TexFilter import Model

# 配置类，用于传递参数给模型
class Config:
    def __init__(self):
        self.seq_len = 256
        self.pred_len = 3  # 3分类预测
        self.enc_in = 6  # 6个特征
        self.embed_size = 128
        self.hidden_size = 256
        self.dropout = 0.1

def test_model():
    # 创建配置
    config = Config()
    
    # 创建模型
    model = Model(config)
    
    # 打印模型结构
    print(model)
    
    # 创建随机输入
    batch_size = 4
    seq_len = config.seq_len
    enc_in = config.enc_in
    
    x_enc = torch.randn(batch_size, seq_len, enc_in)  # 编码器输入
    x_mark_enc = torch.zeros((batch_size, seq_len, 1))  # 编码器标记
    x_dec = torch.zeros_like(x_enc)  # 解码器输入
    x_mark_dec = torch.zeros_like(x_mark_enc)  # 解码器标记
    
    # 前向传播
    try:
        output = model(x_enc, x_mark_enc, x_dec, x_mark_dec)
        print(f"模型输出形状: {output.shape}")
        print("模型测试成功！")
    except Exception as e:
        print(f"模型测试失败: {e}")

if __name__ == "__main__":
    test_model() 