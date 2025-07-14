from typing import Iterable

from bs4 import BeautifulSoup
from gdown import download, exceptions
from requests import get


def get_gdrive_ids(page: str) -> Iterable[str]:
    parsed = BeautifulSoup(page, features='lxml')
    for link in parsed.find_all('a', href=True):
        href = link['href']
        if 'drive.google.com' not in href:
            continue
        yield href.split('/')[-2]


if __name__ == '__main__':
    pages = [
        'https://mari-lab.ru/index.php/%D0%A3%D1%87%D0%B5%D0%B1%D0%BD%D0%B8%D0%BA%D0%B8_%D0%BD%D0%B0_%D0%BC%D0%B0%D1%80%D0%B8%D0%B9%D1%81%D0%BA%D0%BE%D0%BC_%D1%8F%D0%B7%D1%8B%D0%BA%D0%B5',
        'https://mari-lab.ru/index.php/%D0%9C%D0%B0%D1%80%D0%B8%D0%B9_%D1%81%D0%B0%D0%BD%D0%B4%D0%B0%D0%BB%D1%8B%D0%BA_(PDF)',
        'https://mari-lab.ru/index.php/%D0%A3_%D1%81%D0%B5%D0%BC_(PDF)',
        'https://mari-lab.ru/index.php/%D0%9E%D0%BD%D1%87%D1%8B%D0%BA%D0%BE_(PDF)',
    ]
    total = 0
    for page in pages:
        resp = get(pages[0], verify=False)
        gdrive_ids = get_gdrive_ids(resp.text)
        for gdrive_id in gdrive_ids:
            total += 1
            try:
                download(
                    id=gdrive_id, output='magazines_and_students_books/',
                    user_agent=(
                        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                        "AppleWebKit/537.36 (KHTML, like Gecko) "
                        "Chrome/125.0.0.0 Safari/537.36"
                    ),
                    resume=True,
                )
            except exceptions.FileURLRetrievalError:
                print(f'Failed to download {gdrive_id} from {page}')
    print(total)
