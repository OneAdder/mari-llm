from datasets import load_dataset, Dataset
from tqdm import tqdm
from transformers import BertTokenizerFast


def train_tokenizer(dataset: Dataset, vocab_size: int) -> BertTokenizerFast:
    def batch_iterator(batch_size=10000):
        for i in tqdm(range(0, len(dataset), batch_size)):
            yield dataset[i: i + batch_size]["text"]

    tokenizer = BertTokenizerFast.from_pretrained("bert-base-cased")
    return tokenizer.train_new_from_iterator(text_iterator=batch_iterator(), vocab_size=vocab_size)


if __name__ == '__main__':
    data = [
        ('hill', 'OneAdder/mari-bert-pretrain-hill-mari', 30_000),
        ('meadow', 'OneAdder/mari-bert-pretrain-meadow-mari', 30_000),
        ('large', 'OneAdder/mari-bert-pretrain-large', 50_000)
    ]

    for name, data, vocab_size in data:
        dataset = load_dataset(data, split='train')
        tokenizer = train_tokenizer(dataset, vocab_size)
        id_ = f"{name}-mari-bert-tokenizer"
        tokenizer.save_pretrained(id_)
        # tokenizer.push_to_hub(f'OneAdder/{id_}')
