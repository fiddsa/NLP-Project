import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from datasets import load_dataset, Dataset

from src.dataset import Vocabulary, BilingualDataset, Collate, SubwordVocabulary, SpmBilingualDataset
from src.model.transformer import Transformer
from src.train import Trainer, create_optimizer, create_scheduler, WarmupScheduler
from src.inference import GreedySearchDecoder, BeamSearchDecoder
from src.evaluate import Evaluator
from src.visualization import generate_all_plots
from src.data_processor import preprocess_dataset
import os
import sentencepiece as spm

def load_text_to_dataset(en_path, vi_path):
    if en_path is None or vi_path is None:
        return Dataset.from_dict({"en": [], "vi": []})

    with open(en_path, 'r', encoding='utf-8') as f:
        en_lines = [line.strip() for line in f]
    with open(vi_path, 'r', encoding='utf-8') as f:
        vi_lines = [line.strip() for line in f]

    min_len = min(len(en_lines), len(vi_lines))
    en_lines = en_lines[:min_len]
    vi_lines = vi_lines[:min_len]

    return Dataset.from_dict({"en": en_lines, "vi": vi_lines})

def main(config={}):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Device:", device)

    CONFIG = {
        'src': 'en',
        'trg': 'vi',
        'use_rope': True,
        'use_swig': False,
        'vocab_size_en': 12000,
        'vocab_size_vi': 7000,
        'vocab_model_type': 'unigram',
        'num_workers': 4,

        'model_dim': 384,
        'num_heads': 6,
        'num_enc_layers': 4,
        'num_dec_layers': 4,
        'ff_hidden_dim': 1536,
        'dropout': 0.1,
        'max_len_en': 800,
        'max_len_vi': 180,
        
        'batch_size': 6,
        'num_epochs': 20,
        'learning_rate': 5e-5,
        'weight_decay': 1e-5,
        'warmup_steps': 4000,
        'patience': 5,  # Early stopping

        'freq_threshold': 2,
        
        'beam_size': 5,
        'length_penalty': 0.6,

        # 'model_path': None,
        # 'spm_en_path': None,
        # 'spm_vi_path': None,
        # 'history_path': None,
        'train_en_path': None,
        'train_vi_path': None,
        'valid_en_path': None,
        'valid_vi_path': None,
        'test_en_path': None,
        'test_vi_path': None
    }

    for key, value in CONFIG.items():
        if key in config:
            CONFIG[key] = config[key]
    
    print("CONFIGURATION")
    for key, value in CONFIG.items():
        print(key, ":", value)
    
    train_dataset = load_text_to_dataset(CONFIG['train_en_path'], CONFIG['train_vi_path'])
    val_dataset = load_text_to_dataset(CONFIG['valid_en_path'], CONFIG['valid_vi_path'])
    test_dataset = load_text_to_dataset(CONFIG['test_en_path'], CONFIG['test_vi_path'])
    
    train_dataset = preprocess_dataset(train_dataset)
    val_dataset = preprocess_dataset(val_dataset)
    test_dataset = preprocess_dataset(test_dataset)
    
    print(f"  Train samples: {len(train_dataset):,}")
    print(f"  Val samples:   {len(val_dataset):,}")
    print(f"  Test samples:  {len(test_dataset):,}")
    print("="*60 + "\n")
    
    print("Building vocabulary...")

    with open("temp_train.en", "w", encoding="utf-8") as f_en, \
     open("temp_train.vi", "w", encoding="utf-8") as f_vi:
        for x in train_dataset:
            f_en.write(x["en"].strip() + "\n")
            f_vi.write(x["vi"].strip() + "\n")

    spm.SentencePieceTrainer.train(
        input="temp_train.en",
        model_prefix="spm_en",
        vocab_size=CONFIG['vocab_size_en'],
        model_type=CONFIG['vocab_model_type'],
        character_coverage=1.0,
        pad_id=0, bos_id=1, eos_id=2, unk_id=3
    )

    spm.SentencePieceTrainer.train(
        input="temp_train.vi",
        model_prefix="spm_vi",
        vocab_size=CONFIG['vocab_size_vi'],
        model_type=CONFIG['vocab_model_type'],
        character_coverage=0.9995,
        pad_id=0, bos_id=1, eos_id=2, unk_id=3
    )
    os.remove("temp_train.en")
    os.remove("temp_train.vi")
    
    src_model_path = CONFIG.get(f"spm_{CONFIG['src']}_path") or f"spm_{CONFIG['src']}.model"
    trg_model_path = CONFIG.get(f"spm_{CONFIG['trg']}_path") or f"spm_{CONFIG['trg']}.model"

    src_vocab = SubwordVocabulary(src_model_path)
    trg_vocab = SubwordVocabulary(trg_model_path)

    print("Creating dataloaders...")

    train_data = SpmBilingualDataset(
        train_dataset, 
        src_vocab=src_vocab, 
        trg_vocab=trg_vocab, 
        max_src_len=CONFIG[f"max_len_{CONFIG['src']}"],
        max_trg_len=CONFIG[f"max_len_{CONFIG['trg']}"],
        src_lang=CONFIG['src'], 
        trg_lang=CONFIG['trg']
    )
    val_data = SpmBilingualDataset(
        val_dataset, 
        src_vocab=src_vocab, 
        trg_vocab=trg_vocab, 
        max_src_len=CONFIG[f"max_len_{CONFIG['src']}"],
        max_trg_len=CONFIG[f"max_len_{CONFIG['trg']}"],
        src_lang=CONFIG['src'], 
        trg_lang=CONFIG['trg']
    )
    test_data = SpmBilingualDataset(
        test_dataset, 
        src_vocab=src_vocab, 
        trg_vocab=trg_vocab, 
        max_src_len=CONFIG[f"max_len_{CONFIG['src']}"],
        max_trg_len=CONFIG[f"max_len_{CONFIG['trg']}"],
        src_lang=CONFIG['src'], 
        trg_lang=CONFIG['trg']
    )
    
    src_pad_idx = src_vocab.pad_idx
    trg_pad_idx = trg_vocab.pad_idx
    train_loader = DataLoader(
        train_data,
        batch_size=CONFIG['batch_size'],
        shuffle=True,
        collate_fn=Collate(
            src_pad_idx=src_pad_idx,
            trg_pad_idx=trg_pad_idx, 
            max_src_len=CONFIG[f"max_len_{CONFIG['src']}"],
            max_trg_len=CONFIG[f"max_len_{CONFIG['trg']}"]
        ),
        num_workers=CONFIG['num_workers']
    )
    
    val_loader = DataLoader(
        val_data,
        batch_size=CONFIG['batch_size'],
        shuffle=False,
        collate_fn=Collate(
            src_pad_idx=src_pad_idx,
            trg_pad_idx=trg_pad_idx, 
            max_src_len=CONFIG[f"max_len_{CONFIG['src']}"],
            max_trg_len=CONFIG[f"max_len_{CONFIG['trg']}"]
        ),
        num_workers=CONFIG['num_workers']
    )
    
    test_loader = DataLoader(
        test_data,
        batch_size=1,
        shuffle=False,
        collate_fn=Collate(
            src_pad_idx=src_pad_idx,
            trg_pad_idx=trg_pad_idx, 
            max_src_len=CONFIG[f"max_len_{CONFIG['src']}"],
            max_trg_len=CONFIG[f"max_len_{CONFIG['trg']}"]
        ),
        num_workers=CONFIG['num_workers']
    )
    
    print("Creating Transformer model...")
    
    model = Transformer(
        src_vocab_size=len(src_vocab),
        tgt_vocab_size=len(trg_vocab),
        model_dim=CONFIG['model_dim'],
        num_heads=CONFIG['num_heads'],
        num_enc_layers=CONFIG['num_enc_layers'],
        num_dec_layers=CONFIG['num_dec_layers'],
        ff_hidden_dim=CONFIG['ff_hidden_dim'],
        max_len_src=CONFIG[f"max_len_{CONFIG['src']}"],
        max_len_trg=CONFIG[f"max_len_{CONFIG['trg']}"],
        dropout=CONFIG['dropout'],
        pos_type='rope' if CONFIG['use_rope'] else 'pos',
        use_swig=CONFIG['use_swig']
    ).to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print("Total parameters:", total_params)
    print("Trainable parameters:", trainable_params)

    print("Setting up training...")
    
    criterion = nn.CrossEntropyLoss(ignore_index=trg_pad_idx, label_smoothing=0.05)
    optimizer = create_optimizer(
        model,
        learning_rate=CONFIG['learning_rate'],
        weight_decay=CONFIG['weight_decay']
    )
    warmup_scheduler = WarmupScheduler(
        optimizer,
        d_model=CONFIG['model_dim'],
        warmup_steps=CONFIG['warmup_steps']
    )
    plateau_scheduler = create_scheduler(
        optimizer,
        mode='plateau',
        factor=0.5,
        patience=3
    )
    
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        criterion=criterion,
        device=device,
        src_vocab=src_vocab,
        trg_vocab=trg_vocab,
        max_tgt_len=CONFIG[f"max_len_{CONFIG['trg']}"]
    )
    trainer.train(
        num_epochs=CONFIG['num_epochs'],
        warmup_scheduler=warmup_scheduler,
        plateau_scheduler=plateau_scheduler,
        patience=CONFIG['patience']
    )
    checkpoint = torch.load('best_model.pt', map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
 
    greedy_decoder = GreedySearchDecoder(model, max_len=CONFIG[f"max_len_{CONFIG['trg']}"])
    beam_decoder = BeamSearchDecoder(
        model,
        beam_size=CONFIG['beam_size'],
        max_len=CONFIG[f"max_len_{CONFIG['trg']}"],
        length_penalty=CONFIG['length_penalty']
    )

    evaluator = Evaluator(model, test_loader, src_vocab, trg_vocab, device)
    comparison_results = evaluator.compare_decoders(greedy_decoder, beam_decoder)

    generate_all_plots(
        history_path='history.json',
        comparison_results=comparison_results,
        save_dir='figures'
    )
    
if __name__ == "__main__":
    main()
