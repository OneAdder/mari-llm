"""Convert https://meadow-mari.web-corpora.net/ to HF dataset and save"""

import json
from itertools import chain
from pathlib import Path
from typing import Iterable, Optional, Sequence, TypedDict

from datasets import Dataset


def load_plain_text_corpus(path: Path) -> Iterable[str]:
    num_entries = 0
    for text_file in path.glob('**/*.txt'):
        yield text_file.read_text()
        num_entries += 1
    print(f'{path.name}: {num_entries}')


_MEADOW_MARI = 0


class _Sent(TypedDict):
    lang: int
    text: str


def _process_sentences(sentences: Sequence[_Sent]) -> Optional[str]:
    text = ' '.join(sentence['text'] for sentence in sentences if sentence['lang'] == _MEADOW_MARI)
    if text:
        return text
    return None


def load_json_corpus(path: Path) -> Iterable[str]:
    num_entries = 0
    for json_file in path.iterdir():
        if json_file.suffix != '.json':
            continue
        datka = json.loads(json_file.read_text())
        for post in datka['posts'].values():
            post_text = _process_sentences(post['sentences'])
            if post_text:
                yield post_text
                num_entries += 1
            if 'repost_sentences' in post:
                repost_text = _process_sentences(post['repost_sentences'])
                if repost_text:
                    yield repost_text
                    num_entries += 1
            for comment in post['comments'].values():
                comment_text = _process_sentences(comment['sentences'])
                if comment_text:
                    yield comment_text
                    num_entries += 1
    print(f'{path.name}: {num_entries}')


if __name__ == '__main__':
    main_root = Path(__file__).parent / 'tsakorpus_meadow-mari_main' / 'src_convertors' / 'corpus' / 'txt'
    nonfiction = load_plain_text_corpus(main_root / 'nonfiction')
    press = load_plain_text_corpus(main_root / 'press')
    main_corpus = Dataset.from_list([{'text': entry} for entry in chain(nonfiction, press)])

    social_root = Path(__file__).parent / 'tsakorpus_meadow-mari_social' / 'src_convertors' / 'corpus' / 'json_input'
    users = load_json_corpus(social_root / 'users')
    posts = load_json_corpus(social_root)
    social_corpus = Dataset.from_list([{'text': entry} for entry in chain(users, posts)])

    main_corpus.save_to_disk(Path(__file__).parent / 'corpora' / 'meadow-mari-hse-main-corpus')
    social_corpus.save_to_disk(Path(__file__).parent / 'corpora' / 'meadow-mari-hse-social-corpus')
