"""Convert МарНИИЯЛИ https://cloud.mail.ru/public/PB1W/EXXvvCmJv to HF dataset"""

import re
from pathlib import Path
from typing import Iterable, NamedTuple, Optional

from bs4 import BeautifulSoup
from datasets import Dataset


def remove_tags(xml: str) -> str:
    parser = BeautifulSoup(xml, features='xml')
    return parser.get_text()


_META = re.compile(r'\{.*?@.*?@.*?@.*?@.*?\}\n')
_PAD_META = '...'


class CorpusEntry(NamedTuple):
    text: str
    author: str = _PAD_META
    title: str = _PAD_META
    genre: str = _PAD_META
    publisher: str = _PAD_META
    year: str = _PAD_META

    def dict(self) -> dict[str, str]:
        return self._asdict()


def _process_meta(meta: str) -> list[str]:
    return [el.strip() for el in meta.strip('{}\n.').split('@') if len(el) > 4]


def _split(text: str) -> list[CorpusEntry]:
    split = _META.split(text)
    meta = _META.findall(text)
    texts = split[1:]
    if len(texts) != len(meta):
        entries = list(map(CorpusEntry, texts))
    else:
        entries = [CorpusEntry(t, *_process_meta(m)) for t, m in zip(texts, meta)]
    return entries


def parse_marnii_xml(xml: str) -> Iterable[CorpusEntry]:
    text = remove_tags(xml)
    if text:
        yield from _split(text)


def parse_corpus(root_path: Path) -> Iterable[CorpusEntry]:
    for path in root_path.iterdir():
        if path.name == '10_toman_muter_il_pr.xml':
            xml = path.read_text()
            text = remove_tags(xml)
            meta = _process_meta(_META.findall(text)[0])
            text = _META.split(text)[-1]
            for line in text.split('\n'):
                yield CorpusEntry(line, *meta)
            continue
        yield from parse_marnii_xml(path.read_text())


if __name__ == '__main__':
    root_path = Path(__file__).parent / 'mari-corpus'
    dataset = Dataset.from_list([entry.dict() for entry in parse_corpus(root_path)])
    dataset.save_to_disk(Path(__file__).parent / 'corpora' / 'meadow-mari-marnii-corpus')
    # dataset.push_to_hub('OneAdder/meadow-mari-marnii-corpus')
