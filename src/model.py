
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class PositionalEncoder(nn.Module):
    def __init__(self, model_dim=300, max_seq_len=256):
        super().__init__()
        self.model_dim = model_dim
        pe = torch.zeros(max_seq_len, model_dim)

        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        pe = pe.to(device)

        for pos in range(max_seq_len):
            for i in range(0, model_dim, 2):
                pe[pos, i] = math.sin(pos / (10000 ** ((2 * i) / model_dim)))
                pe[pos, i + 1] = math.cos(pos /
                                          (10000 ** ((2 * (i + 1)) / model_dim)))

        self.pe = pe.unsqueeze(0)
        self.pe.requires_grad = False

    def forward(self, x):
        result = math.sqrt(self.model_dim) * x + self.pe
        return result


class Attention(nn.Module):
    def __init__(self, model_dim=300):
        super().__init__()

        self.q_linear = nn.Linear(model_dim, model_dim)
        self.v_linear = nn.Linear(model_dim, model_dim)
        self.k_linear = nn.Linear(model_dim, model_dim)

        self.out = nn.Linear(model_dim, model_dim)
        self.d_k = model_dim

    def forward(self, q, k, v):
        k = self.k_linear(k)
        q = self.q_linear(q)
        v = self.v_linear(v)

        weights = torch.matmul(q, k.transpose(1, 2)) / math.sqrt(self.d_k)

        normalized_weights = F.softmax(weights, dim=-1)
        output = torch.matmul(normalized_weights, v)
        output = self.out(output)

        return output, normalized_weights


class FeedForward(nn.Module):
    def __init__(self, input_dim, output_dim=512, ff_dim=1024, dropout=0):
        super().__init__()

        self.linear_1 = nn.Linear(input_dim, ff_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(ff_dim, output_dim)

    def forward(self, x):
        x = self.linear_1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.linear_2(x)

        return x


class TransformerBlock(nn.Module):
    def __init__(self, model_dim, ff_dim=1024, dropout=0):
        super().__init__()

        self.norm_1 = nn.LayerNorm(model_dim)
        self.norm_2 = nn.LayerNorm(model_dim)

        # self.attention = Attention(model_dim)
        self.attention = nn.MultiheadAttention(model_dim, 8, batch_first=True)

        self.feadforward = FeedForward(model_dim, model_dim, ff_dim)

        self.dropout_1 = nn.Dropout(dropout)
        self.dropout_2 = nn.Dropout(dropout)

    def forward(self, x):
        x_normalized = self.norm_1(x)
        attention_weights, normalized_weights = self.attention(
            x_normalized, x_normalized, x_normalized)

        attention_add_weights = x + self.dropout_1(attention_weights)

        x_normalized_2 = self.norm_2(attention_add_weights)
        x_normalized_2 = self.feadforward(x_normalized_2)
        output = attention_add_weights + self.dropout_2(x_normalized_2)

        return output, normalized_weights


class ClassificationHead(nn.Module):
    def __init__(self, model_dim=300, output_dim=2):
        super().__init__()

        self.linear = nn.Linear(model_dim, output_dim)

    def forward(self, x):
        x0 = x[:, 0, :]
        out = self.linear(x0)

        return out

class CNN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv1d(input_dim, 16, kernel_size=3, padding='same'),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.Conv1d(16, 32, kernel_size=3, padding='same'),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Conv1d(32, output_dim, kernel_size=3, padding='same'),
            nn.BatchNorm1d(output_dim),
        )

    def forward(self, x):
        x = x.transpose(2, 1)
        x = self.model(x)
        x = x.transpose(2, 1)
        return x

class TransformerClassification(nn.Module):
    def __init__(self, ch, length, model_dim, output_dim, depth, bin):
        super().__init__()

        self.ch, self.length = ch, length
        self.model_dim, self.output_dim = model_dim, output_dim
        self.depth, self.bin = depth, bin

        if (self.length % self.bin) == 0:
            self.tf_length = (self.length // self.bin)
            pad = 0
        else: 
            self.tf_length = (self.length // self.bin) + 1
            pad = self.tf_length*self.bin - self.length
        self.padding = nn.ZeroPad2d((0, 0, 0, pad))

        self.cls_token = nn.Parameter(torch.randn(1, 1, self.model_dim*self.bin))
        
        self.cnn = CNN(input_dim=self.ch, output_dim=self.model_dim)

        self.pe = PositionalEncoder(
            model_dim=self.model_dim*self.bin, max_seq_len=self.tf_length+1)

        if self.depth==1:
            self.tf = TransformerBlock(model_dim=self.model_dim*self.bin)
        elif self.depth==6:
            self.tf1 = TransformerBlock(model_dim=model_dim*self.bin)
            self.tf2 = TransformerBlock(model_dim=model_dim*self.bin)
            self.tf3 = TransformerBlock(model_dim=model_dim*self.bin)
            self.tf4 = TransformerBlock(model_dim=model_dim*self.bin)
            self.tf5 = TransformerBlock(model_dim=model_dim*self.bin)
            self.tf6 = TransformerBlock(model_dim=model_dim*self.bin)

        self.head = ClassificationHead(
            model_dim=self.model_dim*self.bin, output_dim=self.output_dim)

    def forward(self, x):
        x = self.padding(x)
        x = self.cnn(x)
        x = x.reshape(x.size(0), self.tf_length, -1)

        cls_tokens = self.cls_token.repeat(x.size(0), 1, 1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = self.pe(x)

        if self.depth==1:
            x, atten_w = self.tf(x)
        elif self.depth==6:
            x, normalized_weights = self.tf1(x)
            atten_w = normalized_weights
            x, normalized_weights = self.tf2(x)
            atten_w *= normalized_weights
            x, normalized_weights = self.tf3(x)
            atten_w *= normalized_weights
            x, normalized_weights = self.tf4(x)
            atten_w *= normalized_weights
            x, normalized_weights = self.tf5(x)
            atten_w *= normalized_weights
            x, normalized_weights = self.tf6(x)
            atten_w *= normalized_weights
        
        x = self.head(x)
        return x, atten_w
