import re
import warnings
from itertools import chain
from pathlib import Path
from string import punctuation
from typing import Iterable, NamedTuple, Optional

from datasets import Dataset
from nltk.tokenize import casual_tokenize, sent_tokenize

from fieldworks_parser import FieldWorksProcessor


mapping = {
    'e': 'е',
    'r': 'р',
    't': 'т',
    'u': 'у',
    'i': 'и',
    'o': 'о',
    'p': 'п',
    'a': 'а',
    's': 'с',
    'd': 'д',
    'f': 'ф',
    'g': 'г',
    'j': 'й',
    'k': 'к',
    'l': 'л',
    'z': 'з',
    'c': 'ц',
    'v': 'в',
    'b': 'б',
    'n': 'н',
    'm': 'м',
    'ə': 'ы',
    'š': 'ш',
    'č': 'ч',
    'x': 'х',
    'ö': 'ӧ',
    'ü': 'ӱ',
    'ž': 'ж',
    'y': 'ы',
    'ӛ': 'ӹ',
    'ä': 'ӓ',
}
escape = ''' \n\r-'",..;:()[]{}*?\\/!@#№$%~`ыӹщйуаь'''
ignore = "'"
spaces = re.compile(r'\s\s+', flags=re.DOTALL)

def remove_j(cyr):
    cyr = re.sub(r'й[у]', 'ю', cyr)
    cyr = re.sub(r'й[а]', 'я', cyr)
    return cyr

def remove_pairs(text: str) -> str:
    text = re.sub(r'ə̑', 'ы', text)
    text = re.sub(r'ə̈', 'ӹ', text)
    text = re.sub(r"š'", 'щ', text)
    text = re.sub(r"š'", 'щ', text)
    text = re.sub(r"(.)' ", '\g<1>ь ', text)
    return text

def capitalize(cyr):
    sentences = sent_tokenize(cyr, language='russian')
    return ' '.join(sent.capitalize() for sent in sentences)

def latin_to_cyrillic(lat):
    cyr = []
    for sym in remove_pairs(lat):
        translit = mapping.get(sym)
        if translit:
            cyr.append(mapping[sym])
            continue
        if not sym in ignore:
            cyr.append(sym)
            if not sym in escape:
                warnings.warn('The symbol {} is not recognised, skipping.'.format(sym))
    return spaces.sub('', capitalize(remove_j(''.join(cyr))))


class CorpusEntryMeta(NamedTuple):
    narrated_by: str
    recorded_by: Optional[str] = None
    year: Optional[str] = None
    transcribed_by: Optional[str] = None
    proofread_by: Optional[str] = None
    glossed_by: Optional[str] = None

    def to_dict(self) -> dict[str, str]:
        return self._asdict()


METADATA_PATTERNS = [
    re.compile(
        (  # default case
            r'(?:.*?)Рассказчики?\: (?P<narrated_by>.*?)'
            r'Запись\: (?P<recorded_by>.*?)'
            r'Год записи\: (?P<year>\d\d\d\d)(?:.*)'
            r'Расшифровка\: (?P<transcribed_by>.*?)'
            r'Вычитка: (?P<proofread_by>.*?)'
            r'Глоссирование: (?P<glossed_by>.*)'
        ),
        flags=re.DOTALL,
    ),
    re.compile(
        (  # case for non-expedition text
            r'(?P<narrated_by>.*?)'
            r'Год записи: (?P<year>\d\d\d\d)(?:.*)'
            r'Опубликовано: (?P<transcribed_by>.*?)'
            r'Техническая подготовка: (?P<proofread_by>.*?)'
            r'Глоссирование: (?P<glossed_by>.*)'),
        flags=re.DOTALL,
    ),
    re.compile(
        (  # fallback for cases when meta is broken
            r'(?:.*?)Рассказчики?\: (?P<narrated_by>.*?)'
            r'Запись\: (?P<recorded_by>.*?)'
            r'Год записи\: (?P<year>\d\d\d\d)(?:.*)'
            r'Расшифровка\: (?P<transcribed_by>.*?)'
        ),
        flags=re.DOTALL,
    ),
    re.compile(  # fallback for cases when meta is completely broken
        r'(?:.*?)Рассказчики?\: (?P<narrated_by>.*)',
        flags=re.DOTALL,
    ),
]


def format_metadata(meta: str) -> CorpusEntryMeta:
    if not meta:
        return CorpusEntryMeta(narrated_by='Unknown')
    match = None
    for pattern in METADATA_PATTERNS:
        match = pattern.match(meta)
        if match:
            break
    if not match:
        raise ValueError('Cannot decipher metadata')
    kwargs = {key: value.strip() for key, value in match.groupdict().items()}
    return CorpusEntryMeta(**kwargs)


if __name__ == '__main__':
    fw_processor = FieldWorksProcessor(Path(__file__).parent / 'Gornomari.fwdata')
    fw_processor.run()
    res = fw_processor()

    dataset = Dataset.from_list([
        {
            "name": entry.name,
            "text": latin_to_cyrillic(entry.text),
            "transcription": entry.text,
            "translation": entry.translation,
            **format_metadata(entry.meta).to_dict()
        } for entry in res
    ])
    num_tokens = 0
    for text in dataset['text']:
        num_tokens += len(casual_tokenize(text))
    # dataset.push_to_hub('OneAdder/hill-mari-msu-spoken-corpus')
