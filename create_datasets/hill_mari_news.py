import re
from pathlib import Path

from bs4 import BeautifulSoup
from datasets import Dataset


def remove_tags_and_header(xml: str) -> str:
    parser = BeautifulSoup(xml, features='xml')
    for data in parser(['header']):
        data.decompose()
    return parser.get_text()


if __name__ == '__main__':
    texts = []
    for filename in (Path(__file__).parent.parent / 'hill-news-corpus').glob('**/*.xml'):
        texts.append((filename.name, remove_tags_and_header(filename.read_text())))
    dataset = Dataset.from_list([
        {'text': text, 'name': name, 'year': re.findall(r'\d\d\d\d', name)[0]}
        for name, text in texts
    ])
    # dataset.push_to_hub('OneAdder/hill-mari-news-corpus', private=True)
