import codecs
import os
import re
from itertools import chain
from pathlib import Path
from typing import Iterable

import fitz
import numpy as np
from datasets import Dataset
from tqdm import tqdm

from mari_vs_russian_classifier import MariVsRussianClassifier


class PDFCorpus:
    TXT_OUTPUT = Path(__file__).parent / 'magazines_and_student_books_txt'
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
                continue
            yield txt_path, self._read_pdf(filename)

    def __iter__(self) -> Iterable[str]:
        for txt_path, text in self._read_pdfs():
            text = self._postprocess(text)
            text = self._filter_russian_out(text)
            text = self._DASH_BREAK_REGEXP.sub('', text)
            txt_path.write_text(text)
            yield text


class PDFCorpusRemap(PDFCorpus):
    _SYM_MAP = {
        'Є': 'Ӧ',  # both
        'є': 'ӧ',
        '™': 'Ӱ',  # meadow
        '¢': 'ӱ',
        'Ў': 'Ӱ',
        'ў': 'ӱ',
        '‰': 'ҥ',
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


_PAD = '\x7F'
codecs.register_error('pad_with_pad_encode', lambda exception: (_PAD.encode(exception.encoding), exception.end))
codecs.register_error('pad_with_pad_decode', lambda exception: (_PAD, exception.end))


class PDFCorpusFixEncoding(PDFCorpusRemap):
    def _postprocess(self, text: str) -> str:
        encoded = text.encode('cp1252', errors='pad_with_pad_encode')
        decoded = encoded.decode('cp1251', errors='pad_with_pad_decode')
        text = decoded.replace(_PAD, '')

        text = self._remap(text)
        return text


if __name__ == '__main__':
    main_root = Path(__file__).parent / 'magazines_and_students_books'

    classifier = MariVsRussianClassifier()

    fixed_very_broken = PDFCorpusFixEncoding(main_root / 'requires_encoding_fix', 'Loading broken PDFs')
    fixed_partly_broken = PDFCorpusRemap(main_root / 'requires_remap', 'Loading partially broken PDFs')
    fixed_super_clean = PDFCorpus(main_root / 'utf8_already', 'Loading clean PDFS')
    fixed_after_ocr = PDFCorpus(main_root / 'after_ocr', 'Loading OCR PDFs')

    dataset = Dataset.from_list([
        {'text': text}
        for text in chain(fixed_very_broken, fixed_partly_broken, fixed_super_clean, fixed_after_ocr)
    ])
    dataset.save_to_disk(Path(__file__).parent / 'corpora' / 'mari-press-students-books-corpus')
