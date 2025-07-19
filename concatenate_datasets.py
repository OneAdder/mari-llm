from typing import Iterable, Mapping

from datasets import concatenate_datasets, load_dataset, Dataset
from nltk.tokenize import casual_tokenize
from tqdm import tqdm


def _add_title_to_text(entry: Mapping) -> dict:
    return {"text": f"{entry['title']}\n{entry['text']}"}


def _leave_only_text(dataset: Dataset) -> Dataset:
    return dataset.remove_columns([col for col in dataset.column_names if col != "text"])


def _calculate_num_tokens(texts: Iterable[str], total: int) -> int:
    num_tokens = 0
    for text in tqdm(texts, total=total, desc='Counting number of tokens...'):
        num_tokens += len(casual_tokenize(text))
    return num_tokens


def load_large(calculate_num_tokens: bool = False) -> Dataset:
    wiki_hill = load_dataset("wikimedia/wikipedia", "20231101.mrj", split="train")
    wiki_hill = wiki_hill.map(_add_title_to_text)
    wiki_meadow = load_dataset("wikimedia/wikipedia", "20231101.mhr", split="train")
    wiki_meadow = wiki_meadow.map(_add_title_to_text)

    meadow_marnii = load_dataset('OneAdder/meadow-mari-marnii-corpus', split="train")
    meadow_hse_main = load_dataset('OneAdder/meadow-mari-hse-main-corpus', split="train")
    meadow_hse_social = load_dataset('OneAdder/meadow-mari-hse-social-corpus', split="train")

    hill_mari_spoken_msu = load_dataset('OneAdder/hill-mari-msu-spoken-corpus', split="train")
    hill_mari_news = load_dataset('OneAdder/hill-mari-news-corpus', split="train")
    hill_mari_books = load_dataset('OneAdder/hill-mari-book-corpus', split="train")

    mari_pdfs = load_dataset('OneAdder/mari-pdf-corpus', split="train")

    dataset = concatenate_datasets(list(map(_leave_only_text, [
        wiki_hill,
        wiki_meadow,
        meadow_marnii,
        meadow_hse_main,
        meadow_hse_social,
        hill_mari_spoken_msu,
        hill_mari_news,
        hill_mari_books,
        mari_pdfs,
    ])))

    if calculate_num_tokens:
        print('Num tokens (Large):', _calculate_num_tokens(dataset['text'], total=len(dataset['text'])))

    return dataset


def load_small_meadow(calculate_num_tokens: bool = False) -> Dataset:
    meadow_marnii = load_dataset('OneAdder/meadow-mari-marnii-corpus', split="train")
    meadow_hse_main = load_dataset('OneAdder/meadow-mari-hse-main-corpus', split="train")

    mari_pdfs = load_dataset('OneAdder/mari-pdf-corpus', split="train")
    mari_pdfs_filtered = mari_pdfs.filter(
        lambda entry: entry['year'] > 1938 and entry['lang'] == 'Meadow Mari' and entry['category'] != 'ocr'
    )

    dataset = concatenate_datasets(list(map(_leave_only_text, [
        meadow_marnii,
        meadow_hse_main,
        mari_pdfs_filtered,
    ])))

    if calculate_num_tokens:
        print('Num tokens (Meadow Mari):', _calculate_num_tokens(dataset['text'], total=len(dataset['text'])))

    return dataset


def load_small_hill(calculate_num_tokens: bool = False) -> Dataset:
    mari_pdfs = load_dataset('OneAdder/mari-pdf-corpus', split="train")
    mari_pdfs_filtered = mari_pdfs.filter(
        lambda entry: entry['year'] > 1938 and entry['lang'] == 'Hill Mari' and entry['category'] != 'ocr'
    )

    hill_mari_spoken_msu = load_dataset('OneAdder/hill-mari-msu-spoken-corpus', split="train")
    hill_mari_news = load_dataset('OneAdder/hill-mari-news-corpus', split="train")
    hill_mari_books = load_dataset('OneAdder/hill-mari-book-corpus', split="train")

    dataset = concatenate_datasets(list(map(_leave_only_text, [
        hill_mari_spoken_msu,
        hill_mari_news,
        hill_mari_books,
        mari_pdfs_filtered,
    ])))

    if calculate_num_tokens:
        print('Num tokens (Hill Mari):', _calculate_num_tokens(dataset['text'], total=len(dataset['text'])))

    return dataset


if __name__ == '__main__':
    print('Clean Hill Mari')
    load_small_hill(calculate_num_tokens=True)
    print('-----------------')
    print('Clean Meadow Mari')
    load_small_meadow(calculate_num_tokens=True)
    print('-----------------')
    print('Large both Mari')
    load_large(calculate_num_tokens=True)
    print('-----------------')
