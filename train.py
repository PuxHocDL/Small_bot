# Import các thư viện cần thiết từ PyTorch và Hugging Face
from model import build_transformer
from dataset import BilingualDataset, causal_mask
from config import get_config, get_weights_file_path, latest_weights_file_path

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from torch.optim.lr_scheduler import LambdaLR

import warnings
from tqdm import tqdm
import os
from pathlib import Path

# Import datasets và tokenizers từ Huggingface
from datasets import load_dataset
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.trainers import WordLevelTrainer
from tokenizers.pre_tokenizers import Whitespace

import torchmetrics
from torch.utils.tensorboard import SummaryWriter

# Hàm greedy_decode thực hiện giải thuật "greedy decoding" 
# để dịch câu bằng cách chọn từ có xác suất cao nhất tại mỗi bước.
def greedy_decode(model, source, source_mask, tokenizer_src, tokenizer_tgt, max_len, device):
    # Lấy chỉ số token bắt đầu và kết thúc của ngôn ngữ đích
    sos_idx = tokenizer_tgt.token_to_id('[SOS]')
    eos_idx = tokenizer_tgt.token_to_id('[EOS]')

    # Tính toán trước output của bộ mã hóa (encoder)
    encoder_output = model.encode(source, source_mask)
    # Bắt đầu câu với token [SOS]
    decoder_input = torch.empty(1, 1).fill_(sos_idx).type_as(source).to(device)
    
    # Vòng lặp để dự đoán từng từ trong câu đích
    while True:
        if decoder_input.size(1) == max_len:  # Nếu đạt đến độ dài tối đa, dừng lại
            break

        # Tạo mask cho câu đích
        decoder_mask = causal_mask(decoder_input.size(1)).type_as(source_mask).to(device)

        # Dự đoán output tiếp theo của decoder
        out = model.decode(encoder_output, source_mask, decoder_input, decoder_mask)

        # Chọn từ có xác suất cao nhất ở bước hiện tại
        prob = model.project(out[:, -1])
        _, next_word = torch.max(prob, dim=1)
        decoder_input = torch.cat(
            [decoder_input, torch.empty(1, 1).type_as(source).fill_(next_word.item()).to(device)], dim=1
        )

        # Dừng lại nếu từ dự đoán là token [EOS]
        if next_word == eos_idx:
            break

    return decoder_input.squeeze(0)

# Hàm run_validation dùng để đánh giá mô hình trên tập dữ liệu kiểm tra.
def run_validation(model, validation_ds, tokenizer_src, tokenizer_tgt, max_len, device, print_msg, global_step, writer, num_examples=2):
    model.eval()
    count = 0

    # Danh sách để lưu các câu nguồn, câu đích và câu dự đoán
    source_texts = []
    expected = []
    predicted = []

    # Thử lấy độ rộng của console để in ra kết quả đẹp mắt
    try:
        with os.popen('stty size', 'r') as console:
            _, console_width = console.read().split()
            console_width = int(console_width)
    except:
        console_width = 80  # Nếu không lấy được, đặt mặc định là 80 ký tự

    with torch.no_grad():  # Tắt gradient để tăng tốc độ tính toán
        for batch in validation_ds:
            count += 1
            encoder_input = batch["encoder_input"].to(device)  # (b, seq_len)
            encoder_mask = batch["encoder_mask"].to(device)  # (b, 1, 1, seq_len)

            # Đảm bảo rằng kích thước batch là 1 (yêu cầu khi kiểm tra)
            assert encoder_input.size(0) == 1, "Kích thước batch phải là 1 khi kiểm tra"

            # Dự đoán đầu ra của mô hình bằng greedy decoding
            model_out = greedy_decode(model, encoder_input, encoder_mask, tokenizer_src, tokenizer_tgt, max_len, device)

            # Lấy văn bản nguồn và đích từ batch
            source_text = batch["src_text"][0]
            target_text = batch["tgt_text"][0]
            model_out_text = tokenizer_tgt.decode(model_out.detach().cpu().numpy())

            # Thêm kết quả vào danh sách để tính toán sau này
            source_texts.append(source_text)
            expected.append(target_text)
            predicted.append(model_out_text)
            
            # In kết quả ra màn hình
            print_msg('-' * console_width)
            print_msg(f"{f'SOURCE: ':>12}{source_text}")
            print_msg(f"{f'TARGET: ':>12}{target_text}")
            print_msg(f"{f'PREDICTED: ':>12}{model_out_text}")

            if count == num_examples:  # Kiểm tra xong số lượng ví dụ yêu cầu thì dừng lại
                print_msg('-' * console_width)
                break
    
    # Ghi nhận các chỉ số đánh giá (CER, WER, BLEU) nếu có TensorBoard
    if writer:
        metric = torchmetrics.CharErrorRate()
        cer = metric(predicted, expected)
        writer.add_scalar('validation cer', cer, global_step)
        writer.flush()

        metric = torchmetrics.WordErrorRate()
        wer = metric(predicted, expected)
        writer.add_scalar('validation wer', wer, global_step)
        writer.flush()

        metric = torchmetrics.BLEUScore()
        bleu = metric(predicted, expected)
        writer.add_scalar('validation BLEU', bleu, global_step)
        writer.flush()

# Hàm get_all_sentences lấy toàn bộ câu từ dataset theo ngôn ngữ yêu cầu
def get_all_sentences(ds, lang):
    for item in ds:
        yield item['translation'][lang]

# Hàm get_or_build_tokenizer tạo hoặc nạp tokenizer nếu đã tồn tại
def get_or_build_tokenizer(config, ds, lang):
    tokenizer_path = Path(config['tokenizer_file'].format(lang))
    if not Path.exists(tokenizer_path):
        # Tạo tokenizer mới nếu chưa có
        tokenizer = Tokenizer(WordLevel(unk_token="[UNK]"))
        tokenizer.pre_tokenizer = Whitespace()
        trainer = WordLevelTrainer(special_tokens=["[UNK]", "[PAD]", "[SOS]", "[EOS]"], min_frequency=2)
        tokenizer.train_from_iterator(get_all_sentences(ds, lang), trainer=trainer)
        tokenizer.save(str(tokenizer_path))
    else:
        # Nạp tokenizer từ file đã lưu
        tokenizer = Tokenizer.from_file(str(tokenizer_path))
    return tokenizer

# Hàm get_ds tạo dataset cho huấn luyện và kiểm tra
def get_ds(config):
    # Tải dataset từ Hugging Face
    ds_big = load_dataset('opus100', 'en-vi')
    ds_raw = ds_big.select(range(len(ds_big) // 100))

    # Tạo tokenizer cho ngôn ngữ nguồn và đích
    tokenizer_src = get_or_build_tokenizer(config, ds_raw['train'], config['lang_src'])
    tokenizer_tgt = get_or_build_tokenizer(config, ds_raw['train'], config['lang_tgt'])

    # Sử dụng tập train và validation có sẵn
    train_ds = BilingualDataset(ds_raw['train'], tokenizer_src, tokenizer_tgt, config['lang_src'], config['lang_tgt'], config['seq_len'])
    val_ds = BilingualDataset(ds_raw['validation'], tokenizer_src, tokenizer_tgt, config['lang_src'], config['lang_tgt'], config['seq_len'])

    # Tạo DataLoader
    train_dataloader = DataLoader(train_ds, batch_size=config['batch_size'], shuffle=True)
    val_dataloader = DataLoader(val_ds, batch_size=1, shuffle=True)

    return train_dataloader, val_dataloader, tokenizer_src, tokenizer_tgt



# Hàm get_model khởi tạo mô hình Transformer dựa trên cấu hình
def get_model(config, vocab_src_len, vocab_tgt_len):
    model = build_transformer(vocab_src_len, vocab_tgt_len, config["seq_len"], config['seq_len'], d_model=config['d_model'])
    return model

# Hàm train_model thực hiện quá trình huấn luyện mô hình
def train_model(config):
    # Xác định thiết bị sử dụng: GPU nếu có, ngược lại sẽ dùng CPU
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.has_mps or torch.backends.mps.is_available() else "cpu"
    print("Thiết bị đang sử dụng:", device)
    if device == 'cuda':
        print(f"Tên thiết bị: {torch.cuda.get_device_name(device.index)}")
        print(f"Dung lượng bộ nhớ: {torch.cuda.get_device_properties(device.index).total_memory / 1024 ** 3} GB")
    elif device == 'mps':
        print(f"Tên thiết bị: <mps>")
    else:
        print("Lưu ý: Nếu bạn có GPU, hãy cân nhắc sử dụng nó để huấn luyện.")
    
    device = torch.device(device)

    # Đảm bảo rằng thư mục chứa trọng số của mô hình tồn tại
    Path(f"{config['datasource']}_{config['model_folder']}").mkdir(parents=True, exist_ok=True)

    # Nạp dữ liệu huấn luyện và kiểm tra
    train_dataloader, val_dataloader, tokenizer_src, tokenizer_tgt = get_ds(config)
    # Khởi tạo mô hình Transformer
    model = get_model(config, tokenizer_src.get_vocab_size(), tokenizer_tgt.get_vocab_size()).to(device)

    # Tensorboard để theo dõi quá trình huấn luyện
    writer = SummaryWriter(config['experiment_name'])

    # Tạo optimizer Adam để tối ưu hóa mô hình
    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'], eps=1e-9)

    # Tải mô hình đã huấn luyện trước (nếu có)
    initial_epoch = 0
    global_step = 0
    preload = config['preload']
    model_filename = latest_weights_file_path(config) if preload == 'latest' else get_weights_file_path(config, preload) if preload else None
    if model_filename:
        print(f'Nạp mô hình từ {model_filename}')
        state = torch.load(model_filename)
        model.load_state_dict(state['model_state_dict'])
        initial_epoch = state['epoch'] + 1
        optimizer.load_state_dict(state['optimizer_state_dict'])
        global_step = state['global_step']
    else:
        print('Bắt đầu huấn luyện từ đầu')

    # Định nghĩa hàm loss với CrossEntropyLoss
    loss_fn = nn.CrossEntropyLoss(ignore_index=tokenizer_src.token_to_id('[PAD]'), label_smoothing=0.1).to(device)

    # Bắt đầu huấn luyện theo từng epoch
    for epoch in range(initial_epoch, config['num_epochs']):
        torch.cuda.empty_cache()  # Giải phóng bộ nhớ GPU
        model.train()  # Đặt mô hình ở chế độ huấn luyện
        batch_iterator = tqdm(train_dataloader, desc=f"Processing Epoch {epoch:02d}")
        for batch in batch_iterator:
            # Lấy dữ liệu từ batch
            encoder_input = batch['encoder_input'].to(device)  # (b, seq_len)
            decoder_input = batch['decoder_input'].to(device)  # (b, seq_len)
            encoder_mask = batch['encoder_mask'].to(device)  # (b, 1, 1, seq_len)
            decoder_mask = batch['decoder_mask'].to(device)  # (b, 1, seq_len, seq_len)

            # Chạy qua encoder và decoder
            encoder_output = model.encode(encoder_input, encoder_mask)  # (b, seq_len, d_model)
            decoder_output = model.decode(encoder_output, encoder_mask, decoder_input, decoder_mask)  # (b, seq_len, d_model)
            proj_output = model.project(decoder_output)  # (b, seq_len, vocab_size)

            # Lấy nhãn để so sánh
            label = batch['label'].to(device)  # (b, seq_len)

            # Tính loss sử dụng CrossEntropy
            loss = loss_fn(proj_output.view(-1, tokenizer_tgt.get_vocab_size()), label.view(-1))
            batch_iterator.set_postfix({"loss": f"{loss.item():6.3f}"})

            # Ghi lại loss vào Tensorboard
            writer.add_scalar('train loss', loss.item(), global_step)
            writer.flush()

            # Tính gradient và cập nhật trọng số mô hình
            loss.backward()
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

            global_step += 1

        # Chạy kiểm tra mô hình sau mỗi epoch
        run_validation(model, val_dataloader, tokenizer_src, tokenizer_tgt, config['seq_len'], device, lambda msg: batch_iterator.write(msg), global_step, writer)

        # Lưu trạng thái mô hình sau mỗi epoch
        model_filename = get_weights_file_path(config, f"{epoch:02d}")
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'global_step': global_step
        }, model_filename)

# Khởi động quá trình huấn luyện khi chạy file
if __name__ == '__main__':
    warnings.filterwarnings("ignore")
    config = get_config()
    train_model(config)
