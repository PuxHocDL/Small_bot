import torch
import torch.nn as nn
import math

class LayerNormalization(nn.Module):
    def __init__(self, features: int, eps: float = 10**-6) -> None:
        super().__init__()
        self.eps = eps
        # alpha là tham số có thể học được (learnable parameter)
        self.alpha = nn.Parameter(torch.ones(features))  
        # bias là tham số có thể học được (learnable parameter)
        self.bias = nn.Parameter(torch.zeros(features))  

    def forward(self, x):
        # Tính trung bình của x theo chiều cuối cùng (dim=-1)
        mean = x.mean(dim=-1, keepdim=True)  # Giữ kích thước để broadcast
        # Tính độ lệch chuẩn của x theo chiều cuối cùng (dim=-1)
        std = x.std(dim=-1, keepdim=True)  # Giữ kích thước để broadcast
        # Chuẩn hóa: (x - mean) / (std + eps) với eps để tránh chia cho 0
        return self.alpha * (x - mean) / (std + self.eps) + self.bias

class FeedForwardBlock(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float) -> None:
        super().__init__()
        # Lớp tuyến tính đầu tiên
        self.linear_1 = nn.Linear(d_model, d_ff)  
        # Dropout để tránh overfitting
        self.dropout = nn.Dropout(dropout)  
        # Lớp tuyến tính thứ hai
        self.linear_2 = nn.Linear(d_ff, d_model)  

    def forward(self, x):
        # Lần lượt áp dụng linear, ReLU, dropout và linear
        return self.linear_2(self.dropout(torch.relu(self.linear_1(x))))

class InputEmbeddings(nn.Module):
    def __init__(self, d_model: int, vocab_size: int) -> None:
        super().__init__()
        self.d_model = d_model
        # Lớp embedding ánh xạ từ vocab_size tới d_model
        self.embedding = nn.Embedding(vocab_size, d_model)  

    def forward(self, x):
        # Nhân với căn bậc hai của d_model để scale theo paper
        return self.embedding(x) * math.sqrt(self.d_model)
    
class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, seq_len: int, dropout: float) -> None:
        super().__init__()
        self.d_model = d_model
        self.seq_len = seq_len
        self.dropout = nn.Dropout(dropout)
        # Tạo ma trận cho positional encoding có kích thước (seq_len, d_model)
        pe = torch.zeros(seq_len, d_model)
        # Tạo vector vị trí (seq_len, 1)
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)
        # Vector div_term dùng để chuẩn hóa vị trí
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        # Áp dụng hàm sin cho các chỉ số chẵn
        pe[:, 0::2] = torch.sin(position * div_term)
        # Áp dụng hàm cos cho các chỉ số lẻ
        pe[:, 1::2] = torch.cos(position * div_term)
        # Thêm chiều batch vào positional encoding
        pe = pe.unsqueeze(0)  
        # Lưu buffer không yêu cầu gradient
        self.register_buffer('pe', pe)

    def forward(self, x):
        # Thêm positional encoding vào embedding input
        x = x + self.pe[:, :x.shape[1], :].requires_grad_(False)
        return self.dropout(x)

class ResidualConnection(nn.Module):
    def __init__(self, features: int, dropout: float) -> None:
        super().__init__()
        # Áp dụng dropout
        self.dropout = nn.Dropout(dropout)  
        # Áp dụng LayerNormalization
        self.norm = LayerNormalization(features)  

    def forward(self, x, sublayer):
        # Kết nối phần còn lại: input + sublayer(norm(input))
        return x + self.dropout(sublayer(self.norm(x)))

class MultiHeadAttentionBlock(nn.Module):
    def __init__(self, d_model: int, h: int, dropout: float) -> None:
        super().__init__()
        self.d_model = d_model  # Kích thước embedding
        self.h = h  # Số lượng đầu chú ý
        # Đảm bảo d_model chia hết cho h
        assert d_model % h == 0, "d_model không chia hết cho h"

        self.d_k = d_model // h  # Kích thước của mỗi vector trong đầu chú ý
        # Các lớp tuyến tính để tính Wq, Wk, Wv và Wo
        self.w_q = nn.Linear(d_model, d_model, bias=False)
        self.w_k = nn.Linear(d_model, d_model, bias=False)
        self.w_v = nn.Linear(d_model, d_model, bias=False)
        self.w_o = nn.Linear(d_model, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    @staticmethod
    def attention(query, key, value, mask, dropout: nn.Dropout):
        d_k = query.shape[-1]
        # Tính attention scores theo công thức (Q * K^T) / sqrt(d_k)
        attention_scores = (query @ key.transpose(-2, -1)) / math.sqrt(d_k)
        if mask is not None:
            # Gán giá trị -inf cho các vị trí được mask
            attention_scores.masked_fill_(mask == 0, -1e9)
        # Áp dụng softmax để chuẩn hóa attention scores
        attention_scores = attention_scores.softmax(dim=-1)
        if dropout is not None:
            attention_scores = dropout(attention_scores)
        # Tính đầu ra attention (attention_scores * V)
        return (attention_scores @ value), attention_scores

    def forward(self, q, k, v, mask):
        # Tính query, key và value từ các vector đầu vào
        query = self.w_q(q)
        key = self.w_k(k)
        value = self.w_v(v)

        # Chia nhỏ thành các đầu chú ý
        query = query.view(query.shape[0], query.shape[1], self.h, self.d_k).transpose(1, 2)
        key = key.view(key.shape[0], key.shape[1], self.h, self.d_k).transpose(1, 2)
        value = value.view(value.shape[0], value.shape[1], self.h, self.d_k).transpose(1, 2)

        # Tính toán attention
        x, self.attention_scores = MultiHeadAttentionBlock.attention(query, key, value, mask, self.dropout)
        
        # Gộp các đầu chú ý lại
        x = x.transpose(1, 2).contiguous().view(x.shape[0], -1, self.h * self.d_k)

        # Tính đầu ra từ lớp tuyến tính W_o
        return self.w_o(x)

class EncoderBlock(nn.Module):
    def __init__(self, features: int, self_attention_block: MultiHeadAttentionBlock, feed_forward_block: FeedForwardBlock, dropout: float) -> None:
        super().__init__()
        self.self_attention_block = self_attention_block
        self.feed_forward_block = feed_forward_block
        # Hai kết nối phần còn lại: attention và feed forward
        self.residual_connections = nn.ModuleList([ResidualConnection(features, dropout) for _ in range(2)])

    def forward(self, x, src_mask):
        # Áp dụng self-attention và feed forward block với residual connections
        x = self.residual_connections[0](x, lambda x: self.self_attention_block(x, x, x, src_mask))
        x = self.residual_connections[1](x, self.feed_forward_block)
        return x
    
class Encoder(nn.Module):
    def __init__(self, features: int, layers: nn.ModuleList) -> None:
        super().__init__()
        self.layers = layers
        # Chuẩn hóa output của encoder
        self.norm = LayerNormalization(features)

    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)

class DecoderBlock(nn.Module):
    def __init__(self, features: int, self_attention_block: MultiHeadAttentionBlock, cross_attention_block: MultiHeadAttentionBlock, feed_forward_block: FeedForwardBlock, dropout: float) -> None:
        super().__init__()
        self.self_attention_block = self_attention_block
        self.cross_attention_block = cross_attention_block
        self.feed_forward_block = feed_forward_block
        # Ba kết nối phần còn lại: self-attention, cross-attention, feed-forward
        self.residual_connections = nn.ModuleList([ResidualConnection(features, dropout) for _ in range(3)])

    def forward(self, x, encoder_output, src_mask, tgt_mask):
        # Áp dụng self-attention, cross-attention và feed-forward với residual connections
        x = self.residual_connections[0](x, lambda x: self.self_attention_block(x, x, x, tgt_mask))
        x = self.residual_connections[1](x, lambda x: self.cross_attention_block(x, encoder_output, encoder_output, src_mask))
        x = self.residual_connections[2](x, self.feed_forward_block)
        return x
    
class Decoder(nn.Module):
    def __init__(self, features: int, layers: nn.ModuleList) -> None:
        super().__init__()
        self.layers = layers
        # Chuẩn hóa output của decoder
        self.norm = LayerNormalization(features)

    def forward(self, x, encoder_output, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x, encoder_output, src_mask, tgt_mask)
        return self.norm(x)

class ProjectionLayer(nn.Module):
    def __init__(self, d_model, vocab_size) -> None:
        super().__init__()
        # Lớp tuyến tính chuyển đầu ra thành xác suất từ vựng
        self.proj = nn.Linear(d_model, vocab_size)

    def forward(self, x) -> None:
        return self.proj(x)
    
class Transformer(nn.Module):
    def __init__(self, encoder: Encoder, decoder: Decoder, src_embed: InputEmbeddings, tgt_embed: InputEmbeddings, src_pos: PositionalEncoding, tgt_pos: PositionalEncoding, projection_layer: ProjectionLayer) -> None:
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.src_pos = src_pos
        self.tgt_pos = tgt_pos
        self.projection_layer = projection_layer

    def encode(self, src, src_mask):
        # Mã hóa input nguồn
        src = self.src_embed(src)
        src = self.src_pos(src)
        return self.encoder(src, src_mask)
    
    def decode(self, encoder_output: torch.Tensor, src_mask: torch.Tensor, tgt: torch.Tensor, tgt_mask: torch.Tensor):
        # Giải mã input đích
        tgt = self.tgt_embed(tgt)
        tgt = self.tgt_pos(tgt)
        return self.decoder(tgt, encoder_output, src_mask, tgt_mask)
    
    def project(self, x):
        # Chiếu đầu ra sang không gian từ vựng
        return self.projection_layer(x)
    
def build_transformer(src_vocab_size: int, tgt_vocab_size: int, src_seq_len: int, tgt_seq_len: int, d_model: int=512, N: int=6, h: int=8, dropout: float=0.1, d_ff: int=2048) -> Transformer:
    # Tạo lớp embedding cho nguồn và đích
    src_embed = InputEmbeddings(d_model, src_vocab_size)
    tgt_embed = InputEmbeddings(d_model, tgt_vocab_size)

    # Tạo lớp positional encoding cho nguồn và đích
    src_pos = PositionalEncoding(d_model, src_seq_len, dropout)
    tgt_pos = PositionalEncoding(d_model, tgt_seq_len, dropout)
    
    # Tạo các khối encoder
    encoder_blocks = []
    for _ in range(N):
        encoder_self_attention_block = MultiHeadAttentionBlock(d_model, h, dropout)
        feed_forward_block = FeedForwardBlock(d_model, d_ff, dropout)
        encoder_block = EncoderBlock(d_model, encoder_self_attention_block, feed_forward_block, dropout)
        encoder_blocks.append(encoder_block)

    # Tạo các khối decoder
    decoder_blocks = []
    for _ in range(N):
        decoder_self_attention_block = MultiHeadAttentionBlock(d_model, h, dropout)
        decoder_cross_attention_block = MultiHeadAttentionBlock(d_model, h, dropout)
        feed_forward_block = FeedForwardBlock(d_model, d_ff, dropout)
        decoder_block = DecoderBlock(d_model, decoder_self_attention_block, decoder_cross_attention_block, feed_forward_block, dropout)
        decoder_blocks.append(decoder_block)
    
    # Tạo encoder và decoder
    encoder = Encoder(d_model, nn.ModuleList(encoder_blocks))
    decoder = Decoder(d_model, nn.ModuleList(decoder_blocks))
    
    # Tạo lớp chiếu
    projection_layer = ProjectionLayer(d_model, tgt_vocab_size)
    
    # Tạo mô hình transformer
    transformer = Transformer(encoder, decoder, src_embed, tgt_embed, src_pos, tgt_pos, projection_layer)
    
    # Khởi tạo tham số
    for p in transformer.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    
    return transformer
