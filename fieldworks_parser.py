from pathlib import Path
from typing import Iterable, NamedTuple, Optional

from lxml.etree import iterparse, Element

_TextID = str
_SegmentID = str
_AnalysisID = str


class FieldWorksMetaDataProcessor:
    def __init__(self, path: Path):
        self.path = path

    @staticmethod
    def _get_meta_id(elem: Element) -> str:
        return elem.find('.//objsur').get('guid')

    @staticmethod
    def _get_prev_elements(elem: Element, n: int) -> Optional[Element]:
        i = 0
        while elem is not None and i <= n:
            elem = elem.getprevious()
            i += 1
        return elem

    def __call__(self) -> tuple[dict[_TextID, str], dict[_TextID, str]]:
        names = {}
        meta = {}
        for _, elem in iterparse(self.path, events=('end',), tag=('Name', 'Source')):
            if elem.tag == 'Name':
                next_ = elem.getnext()
                if next_ is None or next_.tag != 'Source':
                    continue
                auni = elem.find('.//AUni')
                if auni is not None:
                    names.update({self._get_meta_id(self._get_prev_elements(elem, 3)): auni.text})
            if elem.tag == 'Source':
                run = elem.find('.//Run')
                if run is not None:
                    meta.update({self._get_meta_id(self._get_prev_elements(elem, 4)): run.text})
        return names, meta


class _Text(NamedTuple):
    text_segments: list[str]
    segment_ids: list[_SegmentID]


class _Segment(NamedTuple):
    translations: list[str]
    analysis_ids: list[_AnalysisID]


class FlexEntry(NamedTuple):
    name: str
    meta: str
    text_segments: list[str]
    translations: list[list[str]]

    @property
    def text(self) -> str:
        return ''.join(self.text_segments).strip()

    @property
    def translation(self) -> str:
        return ' '.join(''.join(translation).strip() for translation in self.translations).strip()


class FieldWorksProcessor:
    """FieldWorks `.fwdata` file parser

    Extracts name, metadata, text and translation (no glosses so far).

    Example:
    ```python
    processor = FieldWorksProcessor(Path(__file__).parent / 'Corpus.fwdata')
    processor.run()
    texts = list(processor())
    ```
    """

    def __init__(self, path: Path):
        self.path = path
        meta_extractor = FieldWorksMetaDataProcessor(path)
        self._names, self._meta = meta_extractor()
        self._texts: dict[_TextID, _Text] = {}
        self._segments: dict[_TextID, _Segment] = {}

    @staticmethod
    def _get_segment_ids(elem: Element) -> Iterable[_SegmentID]:
        segments = elem.find('Segments')
        for segment in segments.findall('.//objsur'):
            yield segment.get('guid')

    def _get_text(self, elem: Element) -> Optional[_Text]:
        contents_elem = elem.find('.//Contents')
        if contents_elem is None:
            return None
        text = []
        str_elem = contents_elem.find('.//Str')
        if str_elem is None:
            return None
        for run_elem in str_elem.findall('./Run'):
            if run_elem.get('ws') != 'mrj':
                continue
            text.append(run_elem.text)
        if not text:
            return None
        return _Text(
            text_segments=text,
            segment_ids=list(self._get_segment_ids(elem)),
        )

    @staticmethod
    def _get_segment(elem: Element) -> Optional[_Segment]:
        segment = _Segment(analysis_ids=[], translations=[])
        analyses = elem.find('.//Analyses')
        if analyses is None:
            return None
        for analysis in elem.find('.//Analyses').findall('.//objsur'):
            segment.analysis_ids.append(analysis.get('guid'))
        translations = elem.find('.//FreeTranslation')
        if translations is not None:
            for translation in translations.findall('.//Run'):
                segment.translations.append(translation.text)
        return segment

    def run(self) -> None:
        for _, elem in iterparse(self.path, events=('end',), tag='rt'):
            if elem.tag == 'rt':
                if elem.get('class') == 'StTxtPara':
                    text = self._get_text(elem)
                    if text:
                        self._texts.update({elem.get('ownerguid'): self._get_text(elem)})
                if elem.get('class') == 'Segment':
                    segment_id = elem.get('guid')
                    if segment_id is None:
                        continue
                    segment = self._get_segment(elem)
                    if not segment:
                        continue
                    self._segments.update({segment_id: segment})
            elem.clear()

    def __call__(self) -> Iterable[FlexEntry]:
        for id_, text in self._texts.items():
            yield FlexEntry(
                text_segments=text.text_segments,
                name=self._names.get(id_, 'Name Unknown'),
                meta=self._meta.get(id_, ''),
                translations=[self._segments[segment_id].translations for segment_id in text.segment_ids],
            )
