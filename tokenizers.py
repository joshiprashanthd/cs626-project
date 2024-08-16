from itertools import chain
from transformers import AutoTokenizer
from .dataset import create, Loader

mr_hi_dataset, hi_or_dataset, mr_or_dataset = create()

VOCAB_SIZE = 50000


def yield_tokens(dataset, idx):
    for e in dataset:
        yield e[idx]


def create_tokenizers():
    tokenizer = AutoTokenizer.from_pretrained(
        "ai4bharat/indic-bert", do_lower_case=False, keep_accents=True
    )
    mr_tokenizer = tokenizer.train_new_from_iterator(
        chain(yield_tokens(mr_hi_dataset, 0), yield_tokens(mr_or_dataset, 0)),
        VOCAB_SIZE,
    )
    special_tokens_dict = {"eos_token": "<eos>", "bos_token": "<bos>"}
    mr_tokenizer.add_special_tokens(special_tokens_dict)
    mr_tokenizer.save_pretrained("mr_tokenizer_indicbert_50000")

    hi_tokenizer = tokenizer.train_new_from_iterator(
        chain(yield_tokens(mr_hi_dataset, 1), yield_tokens(hi_or_dataset, 0)),
        VOCAB_SIZE,
    )
    special_tokens_dict = {"eos_token": "<eos>", "bos_token": "<bos>"}
    hi_tokenizer.add_special_tokens(special_tokens_dict)
    hi_tokenizer.save_pretrained("hi_tokenizer_indicbert_50000")

    or_tokenizer = tokenizer.train_new_from_iterator(
        chain(yield_tokens(hi_or_dataset, 1), yield_tokens(mr_or_dataset, 1)),
        VOCAB_SIZE,
    )
    special_tokens_dict = {"eos_token": "<eos>", "bos_token": "<bos>"}
    or_tokenizer.add_special_tokens(special_tokens_dict)
    or_tokenizer.save_pretrained("or_tokenizer_indicbert_50000")

    return mr_tokenizer, or_tokenizer, hi_tokenizer


def create_loaders(mr_tokenizer, or_tokenizer, hi_tokenizer):
    src_piv_loader = Loader(mr_hi_dataset, mr_tokenizer, hi_tokenizer)
    src_piv_loader.build_dataloaders(batch_size=32)

    piv_trg_loader = Loader(hi_or_dataset, hi_tokenizer, or_tokenizer)
    piv_trg_loader.build_dataloaders(batch_size=32)

    src_trg_loader = Loader(mr_or_dataset, mr_tokenizer, or_tokenizer)
    src_trg_loader.build_dataloaders(batch_size=32)

    return src_piv_loader, piv_trg_loader, src_trg_loader
