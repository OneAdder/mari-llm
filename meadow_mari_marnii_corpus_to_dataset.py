"""Convert МарНИИЯЛИ https://cloud.mail.ru/public/PB1W/EXXvvCmJv to HF dataset"""

import re
from pathlib import Path
from typing import Iterable

from bs4 import BeautifulSoup
from datasets import Dataset


def _remove_tags(xml: str) -> str:
    parser = BeautifulSoup(xml, features='xml')
    return parser.get_text()


def _split(text: str) -> list[str]:
    split = re.split(r'\{.*?@.*?@.*?@.*?@.*?\}', text, flags=re.DOTALL)
    return split[1:]


def parse_corpus(root_path: Path) -> Iterable[str]:
    for path in root_path.iterdir():
        if path.name == '10_toman_muter_il_pr.xml':
            continue
        xml = path.read_text()
        text = _remove_tags(xml)
        yield from _split(text)


if __name__ == '__main__':
    root_path = Path(__file__).parent / 'mari-corpus'
    dataset = Dataset.from_list([{'text': entry} for entry in parse_corpus(root_path)])
    dataset.save_to_disk(Path(__file__).parent / 'corpora' / 'meadow-mari-marnii-corpus')
