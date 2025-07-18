from pathlib import Path
from datasets import Dataset

from hill_mari_news_to_dataset import remove_tags_and_header
from meadow_mari_marnii_corpus_to_dataset import parse_marnii_xml


if __name__ == '__main__':
    corpus_root = Path(__file__).parent / 'misc-hill-corpus'
    constitution = remove_tags_and_header((corpus_root / 'konstitucijagornaja2015bezvydelenija.doc.xml').read_text())

    xmls = (corpus_root / 'ervi_2010.xml').read_text().split('<?xml version="1.0" encoding="UTF-8"?>')
    datka = [corpus_entry.dict() for xml in xmls for corpus_entry in parse_marnii_xml(xml)]
    datka.append({
        'text': constitution,
        'author': 'Мары Эл Республикын Кугижаншы Погынымаш',
        'title': 'МАРЫ ЭЛ РЕСПУБЛИКӸН КОНСТИТУЦИЖӸ',
        'genre': 'конституци',
        'publisher': 'Мары Эл Республикын Кугижаншы Погынымаш',
        'year': '2015',
    })
    dataset = Dataset.from_list(datka)
    # dataset.push_to_hub('OneAdder/hill-mari-book-corpus')
