import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np

# 暂时注释掉，如果后续需要再放开
# from layers.RevIN import RevIN

class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEmbedding, self).__init__()
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return self.pe[:, :x.size(1)]


class TokenEmbedding(nn.Module):
    def __init__(self, c_in, d_model):
        super(TokenEmbedding, self).__init__()
        padding = 1 if torch.__version__ >= '1.5.0' else 2
        self.tokenConv = nn.Conv1d(in_channels=c_in, out_channels=d_model,
                                    kernel_size=3, padding=padding, padding_mode='circular')
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='leaky_relu')

    def forward(self, x):
        x = self.tokenConv(x.permute(0, 2, 1)).transpose(1, 2)
        return x


class TemporalEmbedding(nn.Module):
    def __init__(self, d_model, embed_type='fixed'):
        super(TemporalEmbedding, self).__init__()

        self.embed_type = embed_type
        if self.embed_type == 'fixed':
            self.position_embedding = PositionalEmbedding(d_model=d_model)

    def forward(self, x, x_mark):
        if self.embed_type == 'fixed':
            return self.position_embedding(x)


class DataEmbedding(nn.Module):
    def __init__(self, c_in, d_model, embed_type='fixed', dropout=0.1):
        super(DataEmbedding, self).__init__()

        self.value_embedding = TokenEmbedding(c_in=c_in, d_model=d_model)
        self.position_embedding = TemporalEmbedding(d_model=d_model, embed_type=embed_type)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, x_mark):
        x = self.value_embedding(x) + self.position_embedding(x, x_mark)
        return self.dropout(x)


class ConvLayer(nn.Module):
    def __init__(self, c_in):
        super(ConvLayer, self).__init__()
        self.downConv = nn.Conv1d(in_channels=c_in,
                                  out_channels=c_in,
                                  kernel_size=3,
                                  padding=1,
                                  padding_mode='circular')
        self.norm = nn.BatchNorm1d(c_in)
        self.activation = nn.ELU()
        self.maxPool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        x = self.downConv(x.permute(0, 2, 1))
        x = self.norm(x)
        x = self.activation(x)
        x = self.maxPool(x)
        x = x.transpose(1, 2)
        return x


class TexFilter(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, dilation=1):
        super(TexFilter, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, dilation=dilation, padding=(kernel_size-1)//2 * dilation)
        self.conv2 = nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, dilation=dilation, padding=(kernel_size-1)//2 * dilation)
        self.bn = nn.BatchNorm1d(out_channels)

    def forward(self, input):
        a = self.conv1(input)
        b = self.conv2(input)
        out = a * torch.sigmoid(b)
        return self.bn(out)


class MultiScaleFeatureExtractor(nn.Module):
    """
    多尺度特征提取器
    结合多尺度卷积和注意力机制提取时间序列特征
    """
    def __init__(self, d_model, n_heads, d_ff, num_layers, dropout=0.1):
        super(MultiScaleFeatureExtractor, self).__init__()
        
        # 多层编码器
        self.layers = nn.ModuleList([
            EncoderLayer(d_model, d_ff, kernel_size=3, dropout=dropout)
            for _ in range(num_layers)
        ])
        
        # 自注意力机制
        self.attention = nn.MultiheadAttention(d_model, n_heads, dropout=dropout)
        
        # 层归一化
        self.norm = nn.LayerNorm(d_model)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        """
        Args:
            x: 输入特征 [batch_size, seq_len, d_model]
            
        Returns:
            输出特征 [batch_size, seq_len, d_model]
        """
        # 通过多层编码器
        for layer in self.layers:
            x = layer(x)
        
        # 自注意力机制
        x_t = x.transpose(0, 1)  # [seq_len, batch_size, d_model]
        attn_output, _ = self.attention(x_t, x_t, x_t)
        attn_output = attn_output.transpose(0, 1)  # [batch_size, seq_len, d_model]
        
        # 残差连接和层归一化
        x = x + self.dropout(attn_output)
        x = self.norm(x)
        
        return x


class EncoderLayer(nn.Module):
    def __init__(self, hidden_size, filter_size=256, kernel_size=3, dropout=0.1):
        super(EncoderLayer, self).__init__()
        d_ff = filter_size
        d_model = hidden_size

        # TexFilter层用于特征提取
        self.texfilter_layer = TexFilter(d_model, d_ff, kernel_size)
        
        # 投影层，用于将滤波器输出投影回原始维度
        self.proj = nn.Linear(d_ff, d_model)
        
        # 残差连接后的层归一化
        self.norm1 = nn.LayerNorm(d_model)
        
        # 前馈网络 (FFN)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model)
        )
        
        # FFN后的层归一化
        self.norm2 = nn.LayerNorm(d_model)
        
        # Dropout层
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        # 输入张量形状: [batch_size, seq_len, d_model]
        
        # 转换输入形状以适应conv1d
        x_t = x.transpose(1, 2)  # [batch_size, d_model, seq_len]
        
        # 通过TexFilter层
        filter_out = self.texfilter_layer(x_t)  # [batch_size, d_ff, seq_len]
        
        # 转换回原始形状
        filter_out = filter_out.transpose(1, 2)  # [batch_size, seq_len, d_ff]
        
        # 投影回原始维度 (如果d_ff != d_model)
        if filter_out.size(-1) != x.size(-1):
            filter_out = self.proj(filter_out)
        
        # 残差连接和层归一化
        x = self.norm1(x + self.dropout(filter_out))
        
        # 前馈网络
        ffn_out = self.ffn(x)
        
        # 残差连接和层归一化
        x = self.norm2(x + self.dropout(ffn_out))
        
        return x


class Encoder(nn.Module):
    def __init__(self, enc_in, d_model, num_layers=3, kernel_size=3, dropout=0.1):
        super(Encoder, self).__init__()
        self.d_model = d_model
        self.enc_in = enc_in

        # 数据嵌入
        self.embedding = DataEmbedding(enc_in, d_model, dropout=dropout)
        
        # 编码层
        self.encoder_layers = nn.ModuleList([
            EncoderLayer(d_model, filter_size=2*d_model, kernel_size=kernel_size, dropout=dropout)
            for _ in range(num_layers)
        ])
        
        # 下采样层
        self.downsample = nn.ModuleList([
            ConvLayer(d_model) for _ in range(2)
        ])

    def forward(self, x, x_mark):
        # 数据嵌入
        x = self.embedding(x, x_mark)  # [batch, seq_len, d_model]
        
        # 通过编码层
        for layer in self.encoder_layers:
            x = layer(x)
        
        # 下采样
        for layer in self.downsample:
            x = layer(x)
            
        return x


class DecoderLayer(nn.Module):
    def __init__(self, hidden_size, filter_size=256, kernel_size=3, dropout=0.1):
        super(DecoderLayer, self).__init__()
        d_ff = filter_size
        d_model = hidden_size

        self.texfilter_layer = TexFilter(d_model, d_ff, kernel_size)
        self.conv1 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.dropout = nn.Dropout(dropout)
        self.layernorm = nn.LayerNorm(d_model)

    def forward(self, x):
        x_texfilter = self.texfilter_layer(x.transpose(1, 2))
        x_texfilter = self.conv1(x_texfilter).transpose(1, 2)
        x = x + self.dropout(x_texfilter)
        x = self.layernorm(x)
        return x


class Decoder(nn.Module):
    def __init__(self, pred_len, d_model, num_layers=2, kernel_size=3, dropout=0.1):
        super(Decoder, self).__init__()
        self.d_model = d_model
        self.pred_len = pred_len

        # 解码层
        self.decoder_layers = nn.ModuleList([
            DecoderLayer(d_model, filter_size=2*d_model, kernel_size=kernel_size, dropout=dropout)
            for _ in range(num_layers)
        ])
        
        # 输出层
        self.projection = nn.Linear(d_model, pred_len)

    def forward(self, x):
        # 通过解码层
        for layer in self.decoder_layers:
            x = layer(x)
        
        # 全局平均池化
        x = torch.mean(x, dim=1)
        
        # 预测
        x = self.projection(x)
        
        return x


class TimeSeriesModel(nn.Module):
    """
    时间序列分类模型
    结合了Transformer和卷积的时间序列分类模型
    这是完整的模型类，不要与TexFilter混淆
    """
    def __init__(self, configs):
        super(TimeSeriesModel, self).__init__()
        self.configs = configs
        self.seq_len = configs.seq_len
        self.pred_len = 3  # 预测类别数量
        
        # 初始化数据标准化层
        # self.RevIN = RevIN(configs.enc_in)  # 导入问题，暂时注释掉
        
        # 1. 输入嵌入
        self.enc_embedding = DataEmbedding(
            configs.enc_in, 
            configs.d_model, 
            dropout=configs.dropout
        )
        
        # 2. 多尺度特征学习
        self.encoder = MultiScaleFeatureExtractor(
            configs.d_model, 
            configs.n_heads, 
            configs.d_ff,
            configs.e_layers,
            configs.dropout
        )
        
        # 3. 预测头
        self.projection = nn.Linear(configs.d_model, self.pred_len)
        
    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None):
        """
        前向传播
        
        Args:
            x_enc: 输入时间序列, [Batch, Length, Channel]
            x_mark_enc: 时间编码信息 (可选)
            x_dec: 解码器输入 (不使用)
            x_mark_dec: 解码器时间编码 (不使用)
            
        Returns:
            outputs: [Batch, num_classes] 类别预测
        """
        if x_mark_enc is None:
            # 创建空时间编码
            x_mark_enc = torch.zeros((x_enc.size(0), x_enc.size(1), 1), device=x_enc.device)
            
        # 提取输入特征形状
        B, L, C = x_enc.shape

        # # 数据正规化
        # x_enc = self.RevIN(x_enc, 'norm')
            
        # 1. 输入嵌入
        enc_out = self.enc_embedding(x_enc, x_mark_enc)  # [B, L, d_model]
        
        # 2. 多尺度特征提取
        enc_out = self.encoder(enc_out)  # [B, L, d_model]
        
        # 3. 全局池化和预测
        # 对序列维度进行平均池化，得到每个样本的特征表示
        enc_out = torch.mean(enc_out, dim=1)  # [B, d_model]
        
        # 预测类别
        outputs = self.projection(enc_out)  # [B, num_classes]
            
        # # 数据反标准化 (分类任务不需要)
        # # outputs = self.RevIN(outputs, 'denorm')
        
        return outputs


class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        # 使用getattr安全地获取配置，提供默认值
        self.enc_in = getattr(configs, 'enc_in', 6)
        self.pred_len = getattr(configs, 'pred_len', 3)  # 默认为3分类
        
        # 嵌入和模型维度
        self.d_model = getattr(configs, 'd_model', 
                               getattr(configs, 'embed_size', 128))
        
        # 隐藏层和dropout
        self.dropout = getattr(configs, 'dropout', 0.1)
        self.num_classes = 3  # 三分类问题
        
        # 打印配置信息以便调试
        print(f"Model configuration:")
        print(f"- enc_in: {self.enc_in}")
        print(f"- d_model: {self.d_model}")
        print(f"- dropout: {self.dropout}")
        print(f"- num_classes: {self.num_classes}")
        
        # 获取其他配置参数，提供默认值
        self.n_heads = getattr(configs, 'n_heads', 4)
        self.d_ff = getattr(configs, 'd_ff', 
                            getattr(configs, 'hidden_size', 256) * 2)
        self.e_layers = getattr(configs, 'e_layers', 3)
        
        print(f"- n_heads: {self.n_heads}")
        print(f"- d_ff: {self.d_ff}")
        print(f"- e_layers: {self.e_layers}")
        
        # 数据嵌入
        self.embedding = DataEmbedding(
            c_in=self.enc_in,
            d_model=self.d_model,
            dropout=self.dropout
        )
        
        # 多层编码器
        self.encoder_layers = nn.ModuleList([
            EncoderLayer(
                hidden_size=self.d_model,
                filter_size=self.d_ff,
                kernel_size=3,
                dropout=self.dropout
            ) for _ in range(self.e_layers)
        ])
        
        # 分类头
        self.classifier = nn.Sequential(
            nn.Linear(self.d_model, self.d_model // 2),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.d_model // 2, self.num_classes)
        )
        
    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None):
        # x_enc: [Batch, Length, Channel]
        
        # 如果没有提供标记，创建一个空的标记
        if x_mark_enc is None:
            x_mark_enc = torch.zeros((x_enc.size(0), x_enc.size(1), 1), device=x_enc.device)
        
        # 数据嵌入
        enc_out = self.embedding(x_enc, x_mark_enc)
        
        # 通过编码器层
        for layer in self.encoder_layers:
            enc_out = layer(enc_out)
        
        # 全局平均池化
        enc_out = torch.mean(enc_out, dim=1)
        
        # 分类
        output = self.classifier(enc_out)
        
        return output
