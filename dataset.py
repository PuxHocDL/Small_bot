import torch
import torch.nn as nn
from torch.utils.data import Dataset

class BilingualDataset(Dataset):
    def __init__(self, ds, tokenizer_src, tokenizer_tgt, src_lang, tgt_lang, seq_len):
        super().__init__()
        self.seq_len = seq_len

        self.ds = ds  # Dữ liệu nguồn và đích (source-target)
        self.tokenizer_src = tokenizer_src  # Bộ mã hóa ngôn ngữ nguồn
        self.tokenizer_tgt = tokenizer_tgt  # Bộ mã hóa ngôn ngữ đích
        self.src_lang = src_lang  # Mã ngôn ngữ nguồn
        self.tgt_lang = tgt_lang  # Mã ngôn ngữ đích

        # Các token đặc biệt: bắt đầu chuỗi (SOS), kết thúc chuỗi (EOS), và token padding (PAD)
        self.sos_token = torch.tensor([tokenizer_tgt.token_to_id("[SOS]")], dtype=torch.int64)
        self.eos_token = torch.tensor([tokenizer_tgt.token_to_id("[EOS]")], dtype=torch.int64)
        self.pad_token = torch.tensor([tokenizer_tgt.token_to_id("[PAD]")], dtype=torch.int64)

    def __len__(self):
        return len(self.ds)  # Trả về độ dài của dataset

    def __getitem__(self, idx):
        src_target_pair = self.ds[idx]  # Lấy cặp ngôn ngữ nguồn-đích tại chỉ số idx
        src_text = src_target_pair['translation'][self.src_lang]  # Lấy văn bản ngôn ngữ nguồn
        tgt_text = src_target_pair['translation'][self.tgt_lang]  # Lấy văn bản ngôn ngữ đích

        # Mã hóa văn bản thành chuỗi token
        enc_input_tokens = self.tokenizer_src.encode(src_text).ids
        dec_input_tokens = self.tokenizer_tgt.encode(tgt_text).ids

        # Thêm các token SOS, EOS và padding cho từng câu
        enc_num_padding_tokens = self.seq_len - len(enc_input_tokens) - 2  # Trừ đi 2 để dành chỗ cho <s> và </s>
        dec_num_padding_tokens = self.seq_len - len(dec_input_tokens) - 1  # Trừ đi 1 để dành chỗ cho <s>

        # Kiểm tra xem số token padding có nhỏ hơn 0 không, nếu có thì câu quá dài
        if enc_num_padding_tokens < 0 or dec_num_padding_tokens < 0:
            raise ValueError("Câu quá dài")

        # Thêm token <s> và </s> vào đầu và cuối chuỗi
        encoder_input = torch.cat(
            [
                self.sos_token,  # Token bắt đầu chuỗi
                torch.tensor(enc_input_tokens, dtype=torch.int64),  # Token hóa văn bản
                self.eos_token,  # Token kết thúc chuỗi
                torch.tensor([self.pad_token] * enc_num_padding_tokens, dtype=torch.int64),  # Thêm padding
            ],
            dim=0,
        )

        # Thêm chỉ token <s> vào đầu chuỗi đầu vào của bộ giải mã
        decoder_input = torch.cat(
            [
                self.sos_token,  # Token bắt đầu chuỗi
                torch.tensor(dec_input_tokens, dtype=torch.int64),  # Token hóa văn bản
                torch.tensor([self.pad_token] * dec_num_padding_tokens, dtype=torch.int64),  # Thêm padding
            ],
            dim=0,
        )

        # Thêm token </s> vào cuối chuỗi đầu ra để làm nhãn
        label = torch.cat(
            [
                torch.tensor(dec_input_tokens, dtype=torch.int64),  # Token hóa văn bản
                self.eos_token,  # Token kết thúc chuỗi
                torch.tensor([self.pad_token] * dec_num_padding_tokens, dtype=torch.int64),  # Thêm padding
            ],
            dim=0,
        )

        # Đảm bảo tất cả các chuỗi đều có độ dài seq_len
        assert encoder_input.size(0) == self.seq_len
        assert decoder_input.size(0) == self.seq_len
        assert label.size(0) == self.seq_len

        return {
            "encoder_input": encoder_input,  # Chuỗi token đầu vào cho bộ mã hóa (seq_len)
            "decoder_input": decoder_input,  # Chuỗi token đầu vào cho bộ giải mã (seq_len)
            "encoder_mask": (encoder_input != self.pad_token).unsqueeze(0).unsqueeze(0).int(),  # Mặt nạ cho bộ mã hóa (1, 1, seq_len)
            "decoder_mask": (decoder_input != self.pad_token).unsqueeze(0).int() & causal_mask(decoder_input.size(0)),  # Mặt nạ cho bộ giải mã với causal_mask (1, seq_len)
            "label": label,  # Chuỗi nhãn đầu ra (seq_len)
            "src_text": src_text,  # Văn bản nguồn gốc
            "tgt_text": tgt_text,  # Văn bản đích
        }
    
def causal_mask(size):
    # Tạo một mặt nạ causal, che các token tương lai
    mask = torch.triu(torch.ones((1, size, size)), diagonal=1).type(torch.int)
    return mask == 0
