from .tokenizers import create_tokenizers, create_loaders, VOCAB_SIZE
from .train import evaluate, train
from .models import Encoder, Decoder, Seq2SeqModel
from .utils import (
    epoch_time,
    initialize_weights,
    init_decoder_cross_attention_weights,
    init_decoder_self_attention_weights,
    init_encoder_weights,
)
import time
import torch
import torch.nn as nn

device = "cuda"

mr_tokenizer, or_tokenizer, hi_tokenizer = create_tokenizers()
src_piv_loader, piv_trg_loader, src_trg_loader = create_loaders(
    mr_tokenizer, or_tokenizer, hi_tokenizer
)

INPUT_DIM = OUTPUT_DIM = VOCAB_SIZE
SRC_PAD_IDX = mr_tokenizer.pad_token_id
TRG_PAD_IDX = hi_tokenizer.pad_token_id
HID_DIM = 160
N_HEADS = 8
PF_DIM = 128
N_LAYERS = 8


def train_src_piv():
    src_piv_enc = Encoder(INPUT_DIM, HID_DIM, N_HEADS, PF_DIM, N_LAYERS, device)
    src_piv_dec = Decoder(OUTPUT_DIM, HID_DIM, N_HEADS, PF_DIM, N_LAYERS, device)
    src_piv_model = Seq2SeqModel(
        src_piv_enc, src_piv_dec, SRC_PAD_IDX, TRG_PAD_IDX, device
    ).to(device)

    src_piv_model.apply(initialize_weights)

    LR = 0.0005
    optimizer = torch.optim.AdamW(src_piv_model.parameters(), lr=LR, betas=(0.9, 0.98))
    criterion = nn.CrossEntropyLoss(ignore_index=TRG_PAD_IDX, label_smoothing=0.1)

    N_EPOCHS = 15
    CLIP = 1
    train_losses = []
    valid_losses = []

    best_valid_loss = float("inf")

    for epoch in range(N_EPOCHS):
        start_time = time.time()

        train_loss = train(
            src_piv_model, src_piv_loader.train_loader, optimizer, criterion, CLIP
        )
        valid_loss = evaluate(src_piv_model, src_piv_loader.dev_loader, criterion)

        end_time = time.time()

        train_losses.append(train_loss)
        valid_losses.append(valid_loss)

        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            torch.save(
                src_piv_model.state_dict(),
                f"mr-hi-model-epoch-{epoch+1}-val-{best_valid_loss:.3f}.pt",
            )

        epoch_mins, epoch_secs = epoch_time(start_time, end_time)

        print(f"Epoch: {epoch+1:02} | Time: {epoch_mins}m {epoch_secs}s")
        print(f"\tTrain Loss: {train_loss:.3f} | Valid Loss : {valid_loss:.3f}")

    torch.save(
        src_piv_model.state_dict(),
        f"mr-hi-model-epoch-{epoch+1}-trn-{train_loss:.3f}.pt",
    )


def train_piv_trg():
    piv_trg_enc = Encoder(INPUT_DIM, HID_DIM, N_HEADS, PF_DIM, N_LAYERS, device)
    piv_trg_dec = Decoder(OUTPUT_DIM, HID_DIM, N_HEADS, PF_DIM, N_LAYERS, device)
    piv_trg_model = Seq2SeqModel(
        piv_trg_enc, piv_trg_dec, SRC_PAD_IDX, TRG_PAD_IDX, device
    ).to(device)

    piv_trg_model.apply(initialize_weights)

    LR = 0.0005
    optimizer = torch.optim.AdamW(piv_trg_model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss(ignore_index=TRG_PAD_IDX)

    N_EPOCHS = 5
    CLIP = 1
    train_losses = []
    valid_losses = []

    best_valid_loss = float("inf")

    for epoch in range(N_EPOCHS):
        start_time = time.time()

        train_loss = train(
            piv_trg_model, piv_trg_loader.train_loader, optimizer, criterion, CLIP
        )
        valid_loss = evaluate(piv_trg_model, piv_trg_loader.dev_loader, criterion)

        end_time = time.time()

        train_losses.append(train_loss)
        valid_losses.append(valid_loss)

        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            torch.save(
                piv_trg_model.state_dict(),
                f"piv-trg-model-epoch-{epoch}-val-{best_valid_loss:.3f}.pt",
            )

        epoch_mins, epoch_secs = epoch_time(start_time, end_time)

        print(f"Epoch: {epoch+1:02} | Time: {epoch_mins}m {epoch_secs}s")
        print(f"\tTrain Loss: {train_loss:.3f} | Valid Loss : {valid_loss:.3f}")

    torch.save(
        piv_trg_model.state_dict(), f"piv-trg-model-trn-{train_loss:.3f}-last.pt"
    )


def train_src_trg(src_piv_model, piv_trg_model):
    src_trg_enc = Encoder(INPUT_DIM, HID_DIM, N_HEADS, PF_DIM, N_LAYERS, device)
    src_trg_dec = Decoder(OUTPUT_DIM, HID_DIM, N_HEADS, PF_DIM, N_LAYERS, device)
    src_trg_model = Seq2SeqModel(
        src_trg_enc, src_trg_dec, SRC_PAD_IDX, TRG_PAD_IDX, device
    ).to(device)

    init_decoder_cross_attention_weights(src_piv_model, src_trg_model)

    init_decoder_self_attention_weights(piv_trg_model, src_trg_model)

    init_encoder_weights(src_piv_model, src_trg_model)

    LR = 0.0005
    optimizer = torch.optim.AdamW(piv_trg_model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss(ignore_index=TRG_PAD_IDX)

    N_EPOCHS = 5
    CLIP = 1
    train_losses = []
    valid_losses = []

    best_valid_loss = float("inf")

    for epoch in range(N_EPOCHS):
        start_time = time.time()

        train_loss = train(
            src_trg_model, src_trg_loader.train_loader, optimizer, criterion, CLIP
        )
        valid_loss = evaluate(src_trg_model, src_trg_loader.dev_loader, criterion)

        end_time = time.time()

        train_losses.append(train_loss)
        valid_losses.append(valid_loss)

        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            torch.save(
                src_trg_model.state_dict(),
                f"src-trg-model-epoch-{epoch}-val-{best_valid_loss:.3f}.pt",
            )

        epoch_mins, epoch_secs = epoch_time(start_time, end_time)

        print(f"Epoch: {epoch+1:02} | Time: {epoch_mins}m {epoch_secs}s")
        print(f"\tTrain Loss: {train_loss:.3f} | Valid Loss : {valid_loss:.3f}")

    torch.save(
        src_trg_model.state_dict(), f"src-trg-model-trn-{train_loss:.3f}-last.pt"
    )
