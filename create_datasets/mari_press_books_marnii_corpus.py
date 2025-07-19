import os
import re
from collections import Counter
from itertools import chain
from pathlib import Path
from typing import Iterable, Literal

import fitz
from nltk.tokenize import casual_tokenize
import numpy as np
from datasets import Dataset
from tqdm import tqdm

from utils.mari_vs_russian_classifier import MariVsRussianClassifier


class PDFCorpus:
    NAME: str

    TXT_OUTPUT = Path(__file__).parent.parent / 'corpora' / 'magazines_and_student_books_txt'
    TXT_OUTPUT.mkdir(exist_ok=True)

    _DASH_BREAK_REGEXP = re.compile(r'[-‑‒–—―﹘﹣－]\n?\r?', flags=re.DOTALL)

    def __init__(self, path: Path, desc: str):
        self._path = path
        self._desc = desc
        self._classifier = MariVsRussianClassifier()

    def _postprocess(self, text: str) -> str:
        text = self._filter_russian_out(text)
        return text

    @staticmethod
    def _read_pdf(path: Path) -> str:
        shards = []
        with fitz.open(path) as doc:
            for page in doc.pages():
                shards.append(page.get_text())
        return '\n\n'.join(shards)

    def _filter_russian_out(self, text: str):
        shards = text.split('\n\n')
        res = []
        for shard in shards:
            # the less the score the Marier
            predictions = self._classifier.predict(shard.split('\n'))
            average_score = np.mean(predictions)
            if average_score <= 0.8:
                res.append(shard)
        return '\n\n'.join(res)

    def _read_pdfs(self) -> Iterable[tuple[Path, str]]:
        for filename in tqdm(self._path.iterdir(), desc=self._desc, total=len(os.listdir(self._path))):
            txt_path = (self.TXT_OUTPUT / filename.with_suffix('.txt').name)
            if txt_path.exists():
                yield txt_path, txt_path.read_text()
                continue
            yield txt_path, self._read_pdf(filename)

    def __iter__(self) -> Iterable[tuple[str, str, str]]:
        for txt_path, text in self._read_pdfs():
            if txt_path.exists():
                yield txt_path.name, text, self.NAME
                continue
            text = self._postprocess(text)
            text = self._filter_russian_out(text)
            text = self._DASH_BREAK_REGEXP.sub('', text)
            txt_path.write_text(text)
            yield txt_path.name, text, self.NAME


class PDFCorpusClean(PDFCorpus):
    NAME = 'clean'


class PDFCorpusOCR(PDFCorpus):
    NAME = 'ocr'


class PDFCorpusRemap(PDFCorpus):
    NAME = 'remap'

    _SYM_MAP = {
        'Є': 'Ӧ',  # both
        'є': 'ӧ',
        '™': 'Ӱ',
        '¢': 'ӱ',
        'Ў': 'Ӱ',
        'ў': 'ӱ',
        '‰': 'ҥ',  # meadow
        'І': 'Ӓ',  # hill
        'і': 'ӓ',
        'Ї': 'Ӹ',
        'ї': 'ӹ',
    }

    def _remap(self, text: str) -> str:
        return ''.join(self._SYM_MAP.get(sym, sym) for sym in text)

    def _postprocess(self, text: str) -> str:
        text = self._remap(text)
        return text


class PDFCorpusFixEncoding(PDFCorpusRemap):
    NAME = 'fix'

    def _postprocess(self, text: str) -> str:
        encoded = text.encode('cp1252', errors='replace')
        decoded = encoded.decode('cp1251', errors='replace')
        text = re.sub(r'\?\?+?', '', decoded)

        text = self._remap(text)
        return text


def _is_hill_mari(text: str) -> bool:
    lowered = text.lower()
    hill_mari_symbols = {'ӓ', 'ӹ'}
    counter = Counter(lowered)
    symbols = all(counter[sym] >= 5 for sym in hill_mari_symbols)
    vla = 'влӓ ' in text
    return symbols and vla


def _is_meadow_mari(text: str) -> bool:
    lowered = text.lower()
    symbols = Counter(lowered)['ҥ'] >= 2
    vlaks = ['влак ', 'влакын ']
    return symbols and any(vlak in lowered for vlak in vlaks)


def get_lang(text: str) -> Literal['Hill Mari', 'Meadow Mari', 'Hill and Meadow Mari', 'unknown']:
    hill = _is_hill_mari(text)
    meadow = _is_meadow_mari(text)
    if hill and meadow:
        return 'Hill and Meadow Mari'
    if hill:
        return 'Hill Mari'
    if meadow:
        return 'Meadow Mari'
    return 'unknown'


def get_year(name: str) -> int:
    year = re.findall(r'\d\d\d\d', name)
    if not year:
        return 0
    return int(year[0])


def get_num_tokens(subset: Iterable[str]) -> int:
    num_tokens = 0
    for text in tqdm(subset, desc='Estimating amount of tokens'):
        num_tokens += len(casual_tokenize(text))
    return num_tokens


if __name__ == '__main__':
    main_root = Path(__file__).parent.parent / 'magazines_and_students_books'

    classifier = MariVsRussianClassifier()

    fixed_very_broken = PDFCorpusFixEncoding(main_root / 'requires_encoding_fix', 'Loading broken PDFs')
    fixed_partly_broken = PDFCorpusRemap(main_root / 'requires_remap', 'Loading partially broken PDFs')
    fixed_super_clean = PDFCorpusClean(main_root / 'utf8_already', 'Loading clean PDFS')
    fixed_after_ocr = PDFCorpusOCR(main_root / 'after_ocr', 'Loading OCR PDFs')

    dataset = Dataset.from_list([
        {
            'text': text,
            'lang': get_lang(text),
            'name': name,
            'category': category,
            'year': get_year(name),
        }
        for name, text, category in chain(
            fixed_very_broken, fixed_partly_broken, fixed_super_clean, fixed_after_ocr
        )
        if text and not text.isspace()
    ])

    # print('Total tokens:', get_num_tokens(dataset['text']))
    # print('Hill tokens:', get_num_tokens(entry['text'] for entry in dataset if entry['lang'] == 'Hill Mari'))
    # print('Meadow tokens:', get_num_tokens(entry['text'] for entry in dataset if entry['lang'] == 'Meadow Mari'))
    # print('No OCR tokens:', get_num_tokens(entry['text'] for entry in dataset if entry['category'] != 'ocr'))
    # print('Modern tokens:', get_num_tokens(entry['text'] for entry in dataset if entry['year'] > 1938))

    dataset.save_to_disk(Path(__file__).parent.parent / 'mari_datasets' / 'mari-press-students-books-corpus')
    # dataset.push_to_hub('OneAdder/mari-pdf-corpus')
