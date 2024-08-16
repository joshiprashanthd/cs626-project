import torch
from indicnlp.transliterate.unicode_transliterate import UnicodeIndicTransliterator


def translate_sentence(
    sentence,
    tokenizer,
    src_vocab,
    trg_vocab,
    model,
    device,
    max_len=50,
    transform=None,
    lang=None,
):
    model.eval()

    if transform:
        sentence = transform(sentence)

    tokens = tokenizer(sentence)

    tokens = ["<bos>"] + tokens + ["<eos>"]
    print("tokens = ", tokens)

    src_indexes = [src_vocab[token] for token in tokens]

    src_tensor = torch.LongTensor(src_indexes).unsqueeze(0).to(device)
    src_mask = model.make_src_mask(src_tensor)

    with torch.no_grad():
        enc_src, _ = model.encoder(src_tensor, src_mask)

    trg_indexes = [trg_vocab["<bos>"]]

    for i in range(max_len):

        trg_tensor = torch.LongTensor(trg_indexes).unsqueeze(0).to(device)

        trg_mask = model.make_trg_mask(trg_tensor)

        with torch.no_grad():
            output, attention = model.decoder(trg_tensor, enc_src, trg_mask, src_mask)

        pred_token = output.argmax(2)[:, -1].item()
        trg_indexes.append(pred_token)

        if pred_token == trg_vocab["<eos>"]:
            break

    trg_tokens = trg_vocab.lookup_tokens(trg_indexes)

    return (
        list(
            trg_tokens[1:].map(
                lambda x: UnicodeIndicTransliterator.transliterate(x, "mr", "or")
            )
        ),
        attention,
    )
    return trg_tokens[1:], attention


translation, attn = translate_sentence(
    "मी माझ्या डोळ्यांनी एक मुलगा पाहिला",
    indic_tokenizer,
    src_piv_loader.src_vocab,
    piv_trg_loader.trg_vocab,
    src_trg_model,
    device,
    transform=mr_normalizer.normalize,
    lang="mar",
)
