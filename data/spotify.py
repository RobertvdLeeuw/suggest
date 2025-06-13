import json
from pprint import pprint
from musicbrainz import get_genres_tags

import spotipy
from spotipy.oauth2 import SpotifyOAuth


SCOPES = ['user-library-read',
          'user-library-modify',
          'playlist-read-private',
          'playlist-read-collaborative',
          'playlist-modify-private',
          'playlist-modify-public']

sp = spotipy.Spotify(auth_manager=SpotifyOAuth(scope=SCOPES,
                                               client_id='a766651ba4b744ed82f1e520a75b2455',
                                               client_secret='767732da0b064b838ebe5d0e3f6ce4eb',
                                               redirect_uri='http://localhost:8888/callback'))
BIG_BIN = "6C4djqIQwoGNnpTjtYBqXq"
ROCK_PLAYLISTS = [
    "1Z7PMYqjWyFlBZaVdKUeWi",
    "1VWaIm5M8X6D5aevLHYsBA",
    "3rrTt52J9C6ey2qS2OkscV",
    "6t0Rp6Xl7VXLDOQOXnLGsv",
]
ICONIC = "6AmmzGkRimiqVqOzvLe2XV"

def _get_playlist_tracks(playlist_id: str) -> list[dict]:
    chunk = sp.playlist(playlist_id)["tracks"]
    tracks = chunk["items"]

    i = 0
    while chunk.get("next"):
        i += 1
        chunk = sp.next(chunk)

        tracks.extend(chunk["items"])
        print(f"Loading chunk {i}", end="\r")
    
    return tracks


def _get_name(track: dict) -> str:
    artists = [artist["name"] for artist in track["track"]["album"]["artists"]]

    return f"{', '.join(artists)} - {track["track"]["name"]}"  # TODO: Re.sub for filename match
    

def get_data():
    items = {_get_name(track): {"score": "near", 
                                "orphan": False, 
                                "isrc": track["track"]["external_ids"].get("isrc")}
             for track in _get_playlist_tracks(BIG_BIN)}

    chunk = sp.current_user_saved_tracks(limit=50)
    liked = chunk["items"]

    while chunk.get("next"):
        chunk = sp.next(chunk)
        liked.extend(chunk["items"])

    items.update({_get_name(track): {"score": "not", 
                                     "orphan": True, 
                                     "isrc": track["track"]["external_ids"].get("isrc")}   # BIG BIN
                  for track in liked})

    chunk = sp.user_playlists(sp.current_user()['id'])
    playlists = chunk["items"]

    while chunk.get("next"):
        chunk = sp.next(chunk)
        playlists.extend(chunk["items"])

    for p in playlists:
        if p["id"] in ROCK_PLAYLISTS + [ICONIC, BIG_BIN] or p["owner"]["id"] != "1192119558":
            print("Skipping", p["name"])
            continue
        print(p["name"])

        items.update({_get_name(track): {"score": "not", 
                                         "orphan": False,
                                         "isrc": track["track"]["external_ids"].get("isrc")}
                      for track in _get_playlist_tracks(p["id"]) if track["track"] is not None})  # OTHER PLAYLISTS
    
    for playlist in ROCK_PLAYLISTS:
        items.update({_get_name(track): {"score": "liked", 
                                         "orphan": False, 
                                         "isrc": track["track"]["external_ids"].get("isrc")}  # LIKED
                      for track in _get_playlist_tracks(playlist)})

    items.update({_get_name(track): {"score": "loved", 
                                     "orphan": False, 
                                     "isrc": track["track"]["external_ids"].get("isrc")}  # LOVED
                  for track in _get_playlist_tracks(ICONIC)})



    length = len(items)
    items = [{"name": key, 
              **val,
              **get_genres_tags(i, length, val["isrc"])
              }
             for i, (key, val) in enumerate(items.items())]
    
    return items


def update():
    with open("data/spotify.json", "r") as f:
        data = json.load(f)

    from copy import deepcopy
    for i, entry in enumerate(deepcopy(data)):
        # print(entry)
        if "tags" in entry or "genres" in entry:
            continue
        
        data[i].update(get_genres_tags(i, len(data), entry["isrc"]))

    with open("data/spotify_new.json", "w") as f:
        json.dump(data, f, indent=4)

update()
        
# data = get_data()
# with open("spotify_data_isrc.json", "w") as f:
#     json.dump(data, f, indent=4)
