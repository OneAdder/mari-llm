from pathlib import Path
from typing import Mapping

from datasets import concatenate_datasets, load_dataset, load_from_disk


def _add_title_to_text(entry: Mapping) -> dict:
    return {"text": f"{entry['title']}\n{entry['text']}"}


wiki_hill = load_dataset("wikimedia/wikipedia", "20231101.mrj", split="train")
wiki_hill = wiki_hill.map(_add_title_to_text)
wiki_hill = wiki_hill.remove_columns([col for col in wiki_hill.column_names if col != "text"])

wiki_meadow = load_dataset("wikimedia/wikipedia", "20231101.mhr", split="train")
wiki_meadow = wiki_meadow.map(_add_title_to_text)
wiki_meadow = wiki_meadow.remove_columns([col for col in wiki_meadow.column_names if col != "text"])

corpora = Path(__file__).parent / 'corpora'
press_meadow = load_from_disk(corpora / 'meadow-mari-hse-main-corpus')
social_meadow = load_from_disk(corpora / 'meadow-mari-hse-social-corpus')
marnii_corpus = load_from_disk(corpora / 'meadow-mari-marnii-corpus')
mari_press_and_students_books = load_from_disk(corpora / 'mari-press-students-books-corpus')

raw_datasets = concatenate_datasets([wiki_meadow, wiki_hill, press_meadow, social_meadow])

raw_datasets.save_to_disk('pretrain_mari_llm.dataset')
