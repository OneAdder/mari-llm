from pathlib import Path

from datasets import load_from_disk, Dataset
from tqdm import tqdm
from transformers import BertTokenizerFast


def train_tokenizer(dataset: Dataset) -> BertTokenizerFast:
    def batch_iterator(batch_size=10000):
        for i in tqdm(range(0, len(dataset), batch_size)):
            yield dataset[i: i + batch_size]["text"]

    tokenizer = BertTokenizerFast.from_pretrained("bert-base-cased")
    return tokenizer.train_new_from_iterator(text_iterator=batch_iterator(), vocab_size=64_000)


if __name__ == '__main__':
    dataset = load_from_disk(Path(__file__).parent / 'pretrain_mari_llm.dataset')
    tokenizer = train_tokenizer(dataset)
    tokenizer.save_pretrained("mari-bert-tokenizer")
