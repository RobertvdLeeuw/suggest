from requests import get
from time import sleep

import xmltodict


def get_xml_resp(url: str):
    response = get(url, headers={'User-Agent': 'Data_retriever/1.0 ( robert.van.der.leeuw@gmail.com )'})

    sleep(1.5)  # To prevent rate-limiting by MusicBrainz.

    if response.status_code != 200:
        return

    return xmltodict.parse(response.text)


def _normalize(isrc: str) -> str:
    return ''.join([char for char in isrc if char.isalnum()])


def _GetDBEntry(isrc: str) -> dict | None:
    for i in range(5):
        try:
            data = get_xml_resp(f"https://musicbrainz.org/ws/2/isrc/{_normalize(isrc)}")
            break
        except:
            print(f"Failed attempt {i} at getting MB data.")
            sleep(1.5)
    else:
        print("It ain't working!")
        return {}

    if not data:
        return {}

    recordings = data['metadata']['isrc']['recording-list']['recording']
    mbID = recordings['@id'] if isinstance(recordings, dict) else recordings[0]['@id']

    for i in range(5):
        try:
            return get_xml_resp(f"https://musicbrainz.org/ws/2/recording/{mbID}?inc=genres+tags")
        except:
            print(f"Failed attempt {i} at getting MB data.")
            sleep(1.5)
    else:
        print("It ain't working!")
        return {}

def get_genres_tags(i, total, isrc: str) -> dict:
    print(f"{i}/{total}", end="\r")

    if not isrc:
        return {"genres": [], "tags": []}

    genresAndTags = {}
    genres, tags = [], []

    if data := _GetDBEntry(isrc):
        if genreList := data['metadata']['recording'].get('genre-list'):
            genres = [g['name'] for g in genreList['genre']] if \
                isinstance(genreList['genre'], list) else [genreList['genre']['name']]

        if tagList := data['metadata']['recording'].get('tag-list'):
            tags = [tag['name'] for tag in tagList['tag']] if \
                isinstance(tagList['tag'], list) else [tagList['tag']['name']]

            tags = list(filter(lambda t: any([_normalize(t) == _normalize(g) for g in genres]), tags))

        genresAndTags['genres'] = genres
        genresAndTags['tags'] = tags

    # print(f"Checking {isrc}. Found so far: {tracksChecked - missingDataTracker}/{tracksChecked} "
          # f"({1 - (missingDataTracker / tracksChecked):.2%}).")

    return genresAndTags

