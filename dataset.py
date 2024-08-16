import torch
from torch.utils.data import Dataset, DataLoader, random_split
from indicnlp.normalize.indic_normalize import DevanagariNormalizer, OriyaNormalizer
import io, os, re
from utils import generate_batch

DRIVE_PATH = "/kaggle/input/indic-hindi-marathi-oriya-parallel-corpus"
HI_MR_PATH = "/hi-mr-small"
HI_OR_PATH = "/hi-or-small"
MR_OR_PATH = "/mr-or-small"


class LangDataset(Dataset):
    def __init__(
        self,
        source_path,
        target_path,
        source_transform=None,
        target_transform=None,
        num_samples=None,
    ):
        self.source_transform = source_transform
        self.target_transform = target_transform

        self.data = []

        source_file = io.open(DRIVE_PATH + source_path, "r")
        target_file = io.open(DRIVE_PATH + target_path, "r")

        for i, e in enumerate(zip(source_file, target_file)):
            if num_samples != None and i == num_samples:
                break
            src = e[0].strip()
            trg = e[1].strip()

            # remove special characters
            src = re.sub(r"[\*\,\(\)\{\}\[\]\:\;\&\%\'\"\#\@\!]*", "", src)
            trg = re.sub(r"[\*\,\(\)\{\}\[\]\:\;\&\%\'\"\#\@\!]*", "", trg)

            self.data.append((src, trg))

        source_file.close()
        target_file.close()

    def __getitem__(self, idx):
        s, t = self.data[idx]

        if self.source_transform:
            s = self.source_transform(s)

        if self.target_transform:
            t = self.target_transform(t)

        return (s, t)

    def __len__(self):
        return len(self.data)


class Loader:
    def __init__(self, dataset, source_tokenizer, target_tokenizer):
        self.dataset = dataset

        self.train_dataset = None
        self.dev_dataset = None
        self.test_dataset = None

        self.train_loader = None
        self.dev_loader = None
        self.test_loader = None
        self.source_tokenizer = source_tokenizer
        self.target_tokenizer = target_tokenizer

    def build_dataloaders(self, batch_size=32):
        gen = torch.Generator().manual_seed(42)
        self.train_dataset, self.dev_dataset, self.test_dataset = random_split(
            self.dataset, [0.8, 0.1, 0.1], generator=gen
        )

        self.train_loader = DataLoader(
            dataset=self.train_dataset,
            batch_size=batch_size,
            shuffle=True,
            drop_last=True,
            collate_fn=generate_batch(self.source_tokenizer, self.target_tokenizer),
        )

        self.dev_loader = DataLoader(
            dataset=self.dev_dataset,
            batch_size=batch_size,
            shuffle=True,
            drop_last=True,
            collate_fn=generate_batch(self.source_tokenizer, self.target_tokenizer),
        )

        self.test_loader = DataLoader(
            dataset=self.test_dataset,
            batch_size=batch_size,
            shuffle=True,
            drop_last=True,
            collate_fn=generate_batch(self.source_tokenizer, self.target_tokenizer),
        )


def create():
    hi_normalizer = DevanagariNormalizer(lang="hi")
    mr_normalizer = DevanagariNormalizer(lang="mr")
    or_normalizer = OriyaNormalizer()

    SOURCE_PATH = os.path.join(HI_MR_PATH, "train.mr")
    TARGET_PATH = os.path.join(HI_MR_PATH, "train.hi")

    mr_hi_dataset = LangDataset(
        SOURCE_PATH,
        TARGET_PATH,
        source_transform=mr_normalizer.normalize,
        target_transform=hi_normalizer.normalize,
    )

    SOURCE_PATH = os.path.join(HI_OR_PATH, "train.hi")
    TARGET_PATH = os.path.join(HI_OR_PATH, "train.or")

    hi_or_dataset = LangDataset(
        SOURCE_PATH,
        TARGET_PATH,
        source_transform=hi_normalizer.normalize,
        target_transform=or_normalizer.normalize,
    )

    SOURCE_PATH = os.path.join(MR_OR_PATH, "train.mr")
    TARGET_PATH = os.path.join(MR_OR_PATH, "train.or")

    mr_or_dataset = LangDataset(
        SOURCE_PATH,
        TARGET_PATH,
        source_transform=mr_normalizer.normalize,
        target_transform=or_normalizer.normalize,
        num_samples=150000,
    )

    return mr_hi_dataset, hi_or_dataset, mr_or_dataset
