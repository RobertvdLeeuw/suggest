import numpy as np
import random
import os

class jukemirlib_fake:
    def extract(self, audio: np.array, layers: list[int], 
                meanpool: bool, offset=0.0) -> dict[int, np.array]:
        return {l: np.random.rand(4800) for l in layers}
    
    def load_audio(self, file_path: str, offset=0.0, duration=None) -> np.array:
        return np.array([])

SAMPLE_RATE = 32_000
from auditus.transform import AudioArray
class auditus_fake:
    def AudioEmbedding(self, return_tensors="pt") -> callable:
        def inner(audio: AudioArray) -> np.array:
            return np.random.rand(3, 768)

        return inner

    class AudioLoader:
        def load_audio(self, file_path: str) -> AudioArray:
            return AudioArray(a=np.random.rand(2, 100_000), sr=SAMPLE_RATE)


from pylast import TopItem
class NamedItem:
    def __init__(self, name: str):
        self.name = name

    def get_name(self) -> str:
        return self.name

class Artist_fake(NamedItem):
    def get_top_tags(self) -> list[TopItem]:
        return [TopItem(item=NamedItem("Large"), weight=random.random()),
                TopItem(item=NamedItem("Non-fiction"), weight=random.random()),
                TopItem(item=NamedItem("Mystery"), weight=random.random()),
                TopItem(item=NamedItem("Thriller"), weight=random.random())]

class Track_fake:
    def __init__(self, artist: str, title: str):
        self.artist = artist
        self.title = title

    def get_artist(self) -> Artist_fake: return Artist_fake(self.artist)

    def get_top_tags(self) -> list[TopItem]:
        return [TopItem(item=NamedItem("Funky"), weight=random.random()),
                TopItem(item=NamedItem("Groovy"), weight=random.random()),
                TopItem(item=NamedItem("Abhorrent"), weight=random.random()),
                TopItem(item=NamedItem("Smelly"), weight=random.random())]

class LastFM_fake:
    def enable_rate_limit(self): pass 

    def get_track(self, artist: str, title: str) -> Track_fake:
        return Track_fake(artist=artist, title=title)


class pylast_fake:
    def LastFMNetwork(self, api_key: str, api_secret: str, 
                      username: str, password_hash: str) -> LastFM_fake:
        return LastFM_fake()
    
    def md5(self, password: str) -> str:
        return password

Json = dict | list
from musicbrainzngs.musicbrainz import ResponseError
class musicbrainz_fake:
    def set_useragent(self, name: str, version: str, contact: str): pass

    def set_rate_limit(self): pass

    def auth(self, user_name: str, password_hash: str): pass

    def search_recordings(self, query: str) -> Json:
        if random.random() < 0.1: return {'recording-list': [], 'recording-count': 0}
        # TODO: urllib.error.HTTPError: HTTP Error 400: Bad Request

        return {
            "recording-list": [
                {
                    "id": "9d8d88c1-ebed-48ad-8b1a-52e5e9a49ecc",
                    "ext:score": "100",
                    "title": "Vitamin C",
                    "length": "212000",
                    "artist-credit": [
                        {
                            "name": "CAN",
                            "artist": {
                                "id": "13501c7d-d181-45ba-af52-5f101d8516a0",
                                "name": "Can",
                                "sort-name": "Can",
                                "disambiguation": "German rock band",
                                "alias-list": [
                                    {
                                        "sort-name": "\u30ab\u30f3",
                                        "alias": "\u30ab\u30f3"
                                    }
                                ]
                            }
                        }
                    ]
                }
            ]
        }
    
    def get_artist_by_id(self, artist_id: str, includes=["tags", "user-tags"]) -> Json:
        if random.random() < 0.01: raise ResponseError("Not found")
        # TODO: urllib.error.HTTPError: HTTP Error 400: Bad Request

        return {
            "artist": {
                "id": "13501c7d-d181-45ba-af52-5f101d8516a0",
                "type": "Group",
                "name": "Can",
                "sort-name": "Can",
                "country": "DE",
                "area": {
                    "id": "85752fda-13c4-31a3-bee5-0e5cb1f51dad",
                    "name": "Germany",
                    "sort-name": "Germany",
                    "iso-3166-1-code-list": [
                        "DE"
                    ]
                },
                "begin-area": {
                    "id": "b8a2776a-eedf-48ea-a6f3-1a9070f0b823",
                    "name": "K\u00f6ln",
                    "sort-name": "K\u00f6ln"
                },
                "disambiguation": "German rock band",
                "isni-list": [
                    "000000046027920X"
                ],
                "life-span": {
                    "begin": "1968",
                    "end": "1991",
                    "ended": "true"
                }
            }
        }

        
    def get_recording_by_id(self, recording_id: str, includes=["tags", "user-tags"]) -> Json:
        if random.random() < 0.01: raise ResponseError("Not found")
        # TODO: urllib.error.HTTPError: HTTP Error 400: Bad Request

        return {
            'recording': 
            {
                'id': '9d8d88c1-ebed-48ad-8b1a-52e5e9a49ecc', 
                'title': 'Vitamin C', 
                'length': '212000'
            }
        }

class Spotifake:
    def artist_top_tracks(self, artist_id: str) -> Json: 
        return {
            "tracks": [
                {
                    "album": {
                        "album_type": "album",
                        "artists": [
                            {
                                "external_urls": {
                                    "spotify": "https://open.spotify.com/artist/4l8xPGtl6DHR2uvunqrl8r"
                                },
                                "href": "https://api.spotify.com/v1/artists/4l8xPGtl6DHR2uvunqrl8r",
                                "id": "4l8xPGtl6DHR2uvunqrl8r",
                                "name": "CAN",
                                "type": "artist",
                                "uri": "spotify:artist:4l8xPGtl6DHR2uvunqrl8r"
                            }
                        ],
                        "external_urls": {
                            "spotify": "https://open.spotify.com/album/1MLxE2czxo5A9OVZ2m8FV3"
                        },
                        "href": "https://api.spotify.com/v1/albums/1MLxE2czxo5A9OVZ2m8FV3",
                        "id": "1MLxE2czxo5A9OVZ2m8FV3",
                        "is_playable": True,
                        "name": "Ege Bamyasi (Remastered)",
                        "release_date": "1972-01-01",
                        "release_date_precision": "day",
                        "total_tracks": 7,
                        "type": "album",
                        "uri": "spotify:album:1MLxE2czxo5A9OVZ2m8FV3"
                    },
                    "artists": [
                        {
                            "external_urls": {
                                "spotify": "https://open.spotify.com/artist/4l8xPGtl6DHR2uvunqrl8r"
                            },
                            "href": "https://api.spotify.com/v1/artists/4l8xPGtl6DHR2uvunqrl8r",
                            "id": "4l8xPGtl6DHR2uvunqrl8r",
                            "name": "CAN",
                            "type": "artist",
                            "uri": "spotify:artist:4l8xPGtl6DHR2uvunqrl8r"
                        }
                    ],
                    "disc_number": 1,
                    "duration_ms": 212000,
                    "explicit": False,
                    "external_ids": {
                        "isrc": "DEX170420804"
                    },
                    "external_urls": {
                        "spotify": "https://open.spotify.com/track/4zdsBics0asw0gj4L5wu5v"
                    },
                    "href": "https://api.spotify.com/v1/tracks/4zdsBics0asw0gj4L5wu5v",
                    "id": "4zdsBics0asw0gj4L5wu5v",
                    "is_local": False,
                    "is_playable": True,
                    "name": "Vitamin C",
                    "popularity": 53,
                    "preview_url": None,
                    "track_number": 4,
                    "type": "track",
                    "uri": "spotify:track:4zdsBics0asw0gj4L5wu5v"
                },
                {
                    "album": {
                        "album_type": "album",
                        "artists": [
                            {
                                "external_urls": {
                                    "spotify": "https://open.spotify.com/artist/4l8xPGtl6DHR2uvunqrl8r"
                                },
                                "href": "https://api.spotify.com/v1/artists/4l8xPGtl6DHR2uvunqrl8r",
                                "id": "4l8xPGtl6DHR2uvunqrl8r",
                                "name": "CAN",
                                "type": "artist",
                                "uri": "spotify:artist:4l8xPGtl6DHR2uvunqrl8r"
                            }
                        ],
                        "external_urls": {
                            "spotify": "https://open.spotify.com/album/2B9ioizVW7n33BQYaHJd6A"
                        },
                        "href": "https://api.spotify.com/v1/albums/2B9ioizVW7n33BQYaHJd6A",
                        "id": "2B9ioizVW7n33BQYaHJd6A",
                        "is_playable": True,
                        "name": "Soundtracks (Remastered)",
                        "release_date": "1970-01-01",
                        "release_date_precision": "day",
                        "total_tracks": 7,
                        "type": "album",
                        "uri": "spotify:album:2B9ioizVW7n33BQYaHJd6A"
                    },
                    "artists": [
                        {
                            "external_urls": {
                                "spotify": "https://open.spotify.com/artist/4l8xPGtl6DHR2uvunqrl8r"
                            },
                            "href": "https://api.spotify.com/v1/artists/4l8xPGtl6DHR2uvunqrl8r",
                            "id": "4l8xPGtl6DHR2uvunqrl8r",
                            "name": "CAN",
                            "type": "artist",
                            "uri": "spotify:artist:4l8xPGtl6DHR2uvunqrl8r"
                        }
                    ],
                    "disc_number": 1,
                    "duration_ms": 244533,
                    "explicit": False,
                    "external_ids": {
                        "isrc": "DEX170420507"
                    },
                    "external_urls": {
                        "spotify": "https://open.spotify.com/track/4bYqOnDzT5OpmfA9apUxyj"
                    },
                    "href": "https://api.spotify.com/v1/tracks/4bYqOnDzT5OpmfA9apUxyj",
                    "id": "4bYqOnDzT5OpmfA9apUxyj",
                    "is_local": False,
                    "is_playable": True,
                    "name": "She Brings the Rain",
                    "popularity": 43,
                    "preview_url": None,
                    "track_number": 7,
                    "type": "track",
                    "uri": "spotify:track:4bYqOnDzT5OpmfA9apUxyj"
                }
            ]
        }

    def artist(self, artist_id: str) -> Json:
        return {
            "external_urls": {
                "spotify": "https://open.spotify.com/artist/4l8xPGtl6DHR2uvunqrl8r"
            },
            "followers": {
                "href": None,
                "total": 345636
            },
            "genres": [
                "krautrock",
                "space rock",
                "progressive rock",
                "experimental",
                "psychedelic rock",
                "art rock",
                "proto-punk"
            ],
            "href": "https://api.spotify.com/v1/artists/4l8xPGtl6DHR2uvunqrl8r",
            "id": "4l8xPGtl6DHR2uvunqrl8r",
            "name": "CAN",
            "popularity": 48,
            "type": "artist",
            "uri": "spotify:artist:4l8xPGtl6DHR2uvunqrl8r"
        }

    def track(self, track_id: str) -> Json:
        return {
            "album": {
                "album_type": "album",
                "artists": [
                    {
                        "external_urls": {
                            "spotify": "https://open.spotify.com/artist/4l8xPGtl6DHR2uvunqrl8r"
                        },
                        "href": "https://api.spotify.com/v1/artists/4l8xPGtl6DHR2uvunqrl8r",
                        "id": "4l8xPGtl6DHR2uvunqrl8r",
                        "name": "CAN",
                        "type": "artist",
                        "uri": "spotify:artist:4l8xPGtl6DHR2uvunqrl8r"
                    }
                ],
                "external_urls": {
                    "spotify": "https://open.spotify.com/album/1MLxE2czxo5A9OVZ2m8FV3"
                },
                "href": "https://api.spotify.com/v1/albums/1MLxE2czxo5A9OVZ2m8FV3",
                "id": "1MLxE2czxo5A9OVZ2m8FV3",
                "name": "Ege Bamyasi (Remastered)",
                "release_date": "1972-01-01",
                "release_date_precision": "day",
                "total_tracks": 7,
                "type": "album",
                "uri": "spotify:album:1MLxE2czxo5A9OVZ2m8FV3"
            },
            "artists": [
                {
                    "external_urls": {
                        "spotify": "https://open.spotify.com/artist/4l8xPGtl6DHR2uvunqrl8r"
                    },
                    "href": "https://api.spotify.com/v1/artists/4l8xPGtl6DHR2uvunqrl8r",
                    "id": "4l8xPGtl6DHR2uvunqrl8r",
                    "name": "CAN",
                    "type": "artist",
                    "uri": "spotify:artist:4l8xPGtl6DHR2uvunqrl8r"
                }
            ],
            "disc_number": 1,
            "duration_ms": 184493,
            "explicit": False,
            "external_ids": {
                "isrc": "DEX170420807"
            },
            "external_urls": {
                "spotify": "https://open.spotify.com/track/0tno4OXs8e3rquqKMkF9XM"
            },
            "href": "https://api.spotify.com/v1/tracks/0tno4OXs8e3rquqKMkF9XM",
            "id": "0tno4OXs8e3rquqKMkF9XM",
            "is_local": False,
            "name": "Spoon",
            "popularity": 32,
            "preview_url": None,
            "track_number": 7,
            "type": "track",
            "uri": "spotify:track:0tno4OXs8e3rquqKMkF9XM"
        }

    def current_user(self) -> Json:
        return {"display_name": "BigLittle", "id": "randomstring"}

    def playlist(self, playlist_id: str) -> Json: 
        return {
            "href": "https://api.spotify.com/v1/playlists/6AmmzGkRimiqVqOzvLe2XV/tracks?offset=0&limit=2&additional_types=track",
            "items": [
                {
                    "added_at": "2025-02-21T00:12:40Z",
                    "added_by": {
                        "external_urls": {
                            "spotify": "https://open.spotify.com/user/1192119558"
                        },
                        "href": "https://api.spotify.com/v1/users/1192119558",
                        "id": "1192119558",
                        "type": "user",
                        "uri": "spotify:user:1192119558"
                    },
                    "is_local": False,
                    "primary_color": None,
                    "track": {
                        "preview_url": None,
                        "explicit": False,
                        "type": "track",
                        "episode": False,
                        "track": True,
                        "album": {
                            "type": "album",
                            "album_type": "album",
                            "href": "https://api.spotify.com/v1/albums/2URYTUSzS5fghtxNS01mtR",
                            "id": "2URYTUSzS5fghtxNS01mtR",
                            "name": "The Occultation of Light",
                            "release_date": "2016-02-26",
                            "release_date_precision": "day",
                            "uri": "spotify:album:2URYTUSzS5fghtxNS01mtR",
                            "artists": [
                                {
                                    "external_urls": {
                                        "spotify": "https://open.spotify.com/artist/3r1b9pqXbTCfPejZtghkKV"
                                    },
                                    "href": "https://api.spotify.com/v1/artists/3r1b9pqXbTCfPejZtghkKV",
                                    "id": "3r1b9pqXbTCfPejZtghkKV",
                                    "name": "Mondo Drag",
                                    "type": "artist",
                                    "uri": "spotify:artist:3r1b9pqXbTCfPejZtghkKV"
                                }
                            ],
                            "external_urls": {
                                "spotify": "https://open.spotify.com/album/2URYTUSzS5fghtxNS01mtR"
                            },
                            "total_tracks": 8
                        },
                        "artists": [
                            {
                                "external_urls": {
                                    "spotify": "https://open.spotify.com/artist/3r1b9pqXbTCfPejZtghkKV"
                                },
                                "href": "https://api.spotify.com/v1/artists/3r1b9pqXbTCfPejZtghkKV",
                                "id": "3r1b9pqXbTCfPejZtghkKV",
                                "name": "Mondo Drag",
                                "type": "artist",
                                "uri": "spotify:artist:3r1b9pqXbTCfPejZtghkKV"
                            }
                        ],
                        "disc_number": 1,
                        "track_number": 1,
                        "duration_ms": 339354,
                        "external_ids": {
                            "isrc": "USYBL1501010"
                        },
                        "external_urls": {
                            "spotify": "https://open.spotify.com/track/32nkHeXgzQg11oh680pzsT"
                        },
                        "href": "https://api.spotify.com/v1/tracks/32nkHeXgzQg11oh680pzsT",
                        "id": "32nkHeXgzQg11oh680pzsT",
                        "name": "Initiation",
                        "popularity": 27,
                        "uri": "spotify:track:32nkHeXgzQg11oh680pzsT",
                        "is_local": False
                    },
                    "video_thumbnail": {
                        "url": None
                    }
                },
                {
                    "added_at": "2025-03-09T13:42:49Z",
                    "added_by": {
                        "external_urls": {
                            "spotify": "https://open.spotify.com/user/1192119558"
                        },
                        "href": "https://api.spotify.com/v1/users/1192119558",
                        "id": "1192119558",
                        "type": "user",
                        "uri": "spotify:user:1192119558"
                    },
                    "is_local": False,
                    "primary_color": None,
                    "track": {
                        "preview_url": None,
                        "available_markets": [],
                        "explicit": False,
                        "type": "track",
                        "episode": False,
                        "track": True,
                        "album": {
                            "available_markets": [],
                            "type": "album",
                            "album_type": "album",
                            "href": "https://api.spotify.com/v1/albums/5s0ZG942WFkixSlqn4hDY8",
                            "id": "5s0ZG942WFkixSlqn4hDY8",
                            "name": "Forest of Lost Children",
                            "release_date": "2014-05-20",
                            "release_date_precision": "day",
                            "uri": "spotify:album:5s0ZG942WFkixSlqn4hDY8",
                            "artists": [
                                {
                                    "external_urls": {
                                        "spotify": "https://open.spotify.com/artist/0hrb5WRiNlj8vh3WnCgXFq"
                                    },
                                    "href": "https://api.spotify.com/v1/artists/0hrb5WRiNlj8vh3WnCgXFq",
                                    "id": "0hrb5WRiNlj8vh3WnCgXFq",
                                    "name": "Kikagaku Moyo",
                                    "type": "artist",
                                    "uri": "spotify:artist:0hrb5WRiNlj8vh3WnCgXFq"
                                }
                            ],
                            "external_urls": {
                                "spotify": "https://open.spotify.com/album/5s0ZG942WFkixSlqn4hDY8"
                            },
                            "total_tracks": 6
                        },
                        "artists": [
                            {
                                "external_urls": {
                                    "spotify": "https://open.spotify.com/artist/0hrb5WRiNlj8vh3WnCgXFq"
                                },
                                "href": "https://api.spotify.com/v1/artists/0hrb5WRiNlj8vh3WnCgXFq",
                                "id": "0hrb5WRiNlj8vh3WnCgXFq",
                                "name": "Kikagaku Moyo",
                                "type": "artist",
                                "uri": "spotify:artist:0hrb5WRiNlj8vh3WnCgXFq"
                            }
                        ],
                        "disc_number": 1,
                        "track_number": 3,
                        "duration_ms": 436171,
                        "external_ids": {
                            "isrc": "QMMS41401303"
                        },
                        "external_urls": {
                            "spotify": "https://open.spotify.com/track/3J5JpH6aNqarzzC56svx3V"
                        },
                        "href": "https://api.spotify.com/v1/tracks/3J5JpH6aNqarzzC56svx3V",
                        "id": "3J5JpH6aNqarzzC56svx3V",
                        "name": "Smoke and Mirrors",
                        "popularity": 0,
                        "uri": "spotify:track:3J5JpH6aNqarzzC56svx3V",
                        "is_local": False
                    },
                    "video_thumbnail": {
                        "url": None
                    }
                }
            ],
            "limit": 2,
            "next": "https://api.spotify.com/v1/playlists/6AmmzGkRimiqVqOzvLe2XV/tracks?offset=2&limit=2&additional_types=track",
            "offset": 0,
            "previous": None,
            "total": 84
        }

    def current_user_saved_tracks(self, limit: int) -> Json: 
        return {
            "href": "https://api.spotify.com/v1/me/tracks?offset=0&limit=2",
            "items": [
                {
                    "added_at": "2025-07-30T16:38:14Z",
                    "track": {
                        "album": {
                            "album_type": "album",
                            "artists": [
                                {
                                    "external_urls": {
                                        "spotify": "https://open.spotify.com/artist/0Iv00ucAIqr5KVS7bXGFa9"
                                    },
                                    "href": "https://api.spotify.com/v1/artists/0Iv00ucAIqr5KVS7bXGFa9",
                                    "id": "0Iv00ucAIqr5KVS7bXGFa9",
                                    "name": "Ozric Tentacles",
                                    "type": "artist",
                                    "uri": "spotify:artist:0Iv00ucAIqr5KVS7bXGFa9"
                                }
                            ],
                            "external_urls": {
                                "spotify": "https://open.spotify.com/album/1O8LiHN6IizBkjNv1kqnKg"
                            },
                            "href": "https://api.spotify.com/v1/albums/1O8LiHN6IizBkjNv1kqnKg",
                            "id": "1O8LiHN6IizBkjNv1kqnKg",
                            "is_playable": True,
                            "name": "Strangeitude",
                            "release_date": "1991",
                            "release_date_precision": "year",
                            "total_tracks": 8,
                            "type": "album",
                            "uri": "spotify:album:1O8LiHN6IizBkjNv1kqnKg"
                        },
                        "artists": [
                            {
                                "external_urls": {
                                    "spotify": "https://open.spotify.com/artist/0Iv00ucAIqr5KVS7bXGFa9"
                                },
                                "href": "https://api.spotify.com/v1/artists/0Iv00ucAIqr5KVS7bXGFa9",
                                "id": "0Iv00ucAIqr5KVS7bXGFa9",
                                "name": "Ozric Tentacles",
                                "type": "artist",
                                "uri": "spotify:artist:0Iv00ucAIqr5KVS7bXGFa9"
                            }
                        ],
                        "disc_number": 1,
                        "duration_ms": 313121,
                        "explicit": False,
                        "external_ids": {
                            "isrc": "GBCQV9900054"
                        },
                        "external_urls": {
                            "spotify": "https://open.spotify.com/track/3BjrqMXidJU1mKNM1nU92k"
                        },
                        "href": "https://api.spotify.com/v1/tracks/3BjrqMXidJU1mKNM1nU92k",
                        "id": "3BjrqMXidJU1mKNM1nU92k",
                        "is_local": False,
                        "is_playable": True,
                        "name": "Weirditude",
                        "popularity": 14,
                        "preview_url": None,
                        "track_number": 8,
                        "type": "track",
                        "uri": "spotify:track:3BjrqMXidJU1mKNM1nU92k"
                    }
                },
                {
                    "added_at": "2025-07-30T16:32:22Z",
                    "track": {
                        "album": {
                            "album_type": "album",
                            "artists": [
                                {
                                    "external_urls": {
                                        "spotify": "https://open.spotify.com/artist/0Iv00ucAIqr5KVS7bXGFa9"
                                    },
                                    "href": "https://api.spotify.com/v1/artists/0Iv00ucAIqr5KVS7bXGFa9",
                                    "id": "0Iv00ucAIqr5KVS7bXGFa9",
                                    "name": "Ozric Tentacles",
                                    "type": "artist",
                                    "uri": "spotify:artist:0Iv00ucAIqr5KVS7bXGFa9"
                                }
                            ],
                            "external_urls": {
                                "spotify": "https://open.spotify.com/album/1O8LiHN6IizBkjNv1kqnKg"
                            },
                            "href": "https://api.spotify.com/v1/albums/1O8LiHN6IizBkjNv1kqnKg",
                            "id": "1O8LiHN6IizBkjNv1kqnKg",
                            "is_playable": True,
                            "name": "Strangeitude",
                            "release_date": "1991",
                            "release_date_precision": "year",
                            "total_tracks": 8,
                            "type": "album",
                            "uri": "spotify:album:1O8LiHN6IizBkjNv1kqnKg"
                        },
                        "artists": [
                            {
                                "external_urls": {
                                    "spotify": "https://open.spotify.com/artist/0Iv00ucAIqr5KVS7bXGFa9"
                                },
                                "href": "https://api.spotify.com/v1/artists/0Iv00ucAIqr5KVS7bXGFa9",
                                "id": "0Iv00ucAIqr5KVS7bXGFa9",
                                "name": "Ozric Tentacles",
                                "type": "artist",
                                "uri": "spotify:artist:0Iv00ucAIqr5KVS7bXGFa9"
                            }
                        ],
                        "disc_number": 1,
                        "duration_ms": 436302,
                        "explicit": False,
                        "external_ids": {
                            "isrc": "GBCQV9300117"
                        },
                        "external_urls": {
                            "spotify": "https://open.spotify.com/track/4rw3F1Vl8elsV9x05j1pSP"
                        },
                        "href": "https://api.spotify.com/v1/tracks/4rw3F1Vl8elsV9x05j1pSP",
                        "id": "4rw3F1Vl8elsV9x05j1pSP",
                        "is_local": False,
                        "is_playable": True,
                        "name": "Live Throbbe",
                        "popularity": 16,
                        "preview_url": None,
                        "track_number": 7,
                        "type": "track",
                        "uri": "spotify:track:4rw3F1Vl8elsV9x05j1pSP"
                    }
                }
            ],
            "limit": 2,
            "next": "https://api.spotify.com/v1/me/tracks?offset=2&limit=2",
            "offset": 0,
            "previous": None,
            "total": 2711
        }

    def album_tracks(self, album_id: str, limit: int) -> Json: 
        return {
            "href": "https://api.spotify.com/v1/albums/6u9vLxqiEzXAd9VE5zi2EN/tracks?offset=0&limit=2",
            "items": [
                {
                    "artists": [
                        {
                            "external_urls": {
                                "spotify": "https://open.spotify.com/artist/4l8xPGtl6DHR2uvunqrl8r"
                            },
                            "href": "https://api.spotify.com/v1/artists/4l8xPGtl6DHR2uvunqrl8r",
                            "id": "4l8xPGtl6DHR2uvunqrl8r",
                            "name": "CAN",
                            "type": "artist",
                            "uri": "spotify:artist:4l8xPGtl6DHR2uvunqrl8r"
                        }
                    ],
                    "disc_number": 1,
                    "duration_ms": 605000,
                    "explicit": False,
                    "external_urls": {
                        "spotify": "https://open.spotify.com/track/59aSAf42HataSl7leXNtAa"
                    },
                    "href": "https://api.spotify.com/v1/tracks/59aSAf42HataSl7leXNtAa",
                    "id": "59aSAf42HataSl7leXNtAa",
                    "name": "Keele 77 Eins",
                    "preview_url": None,
                    "track_number": 1,
                    "type": "track",
                    "uri": "spotify:track:59aSAf42HataSl7leXNtAa",
                    "is_local": False
                },
                {
                    "artists": [
                        {
                            "external_urls": {
                                "spotify": "https://open.spotify.com/artist/4l8xPGtl6DHR2uvunqrl8r"
                            },
                            "href": "https://api.spotify.com/v1/artists/4l8xPGtl6DHR2uvunqrl8r",
                            "id": "4l8xPGtl6DHR2uvunqrl8r",
                            "name": "CAN",
                            "type": "artist",
                            "uri": "spotify:artist:4l8xPGtl6DHR2uvunqrl8r"
                        }
                    ],
                    "disc_number": 1,
                    "duration_ms": 907437,
                    "explicit": False,
                    "external_urls": {
                        "spotify": "https://open.spotify.com/track/0yG6nGRX6feqo8QTOVWKnZ"
                    },
                    "href": "https://api.spotify.com/v1/tracks/0yG6nGRX6feqo8QTOVWKnZ",
                    "id": "0yG6nGRX6feqo8QTOVWKnZ",
                    "name": "Keele 77 Zwei",
                    "preview_url": None,
                    "track_number": 2,
                    "type": "track",
                    "uri": "spotify:track:0yG6nGRX6feqo8QTOVWKnZ",
                    "is_local": False
                }
            ],
            "limit": 2,
            "next": "https://api.spotify.com/v1/albums/6u9vLxqiEzXAd9VE5zi2EN/tracks?offset=2&limit=2",
            "offset": 0,
            "previous": None,
            "total": 5
        }

    def artist_albums(self, artist_id: str, limit: int) -> Json:
        return {
            "href": "https://api.spotify.com/v1/artists/4l8xPGtl6DHR2uvunqrl8r/albums?offset=0&limit=2&include_groups=album,single,compilation,appears_on",
            "limit": 2,
            "next": "https://api.spotify.com/v1/artists/4l8xPGtl6DHR2uvunqrl8r/albums?offset=2&limit=2&include_groups=album,single,compilation,appears_on",
            "offset": 0,
            "previous": None,
            "total": 38,
            "items": [
                {
                    "album_type": "album",
                    "total_tracks": 5,
                    "external_urls": {
                        "spotify": "https://open.spotify.com/album/6u9vLxqiEzXAd9VE5zi2EN"
                    },
                    "href": "https://api.spotify.com/v1/albums/6u9vLxqiEzXAd9VE5zi2EN",
                    "id": "6u9vLxqiEzXAd9VE5zi2EN",
                    "name": "LIVE IN KEELE 1977",
                    "release_date": "2024-11-22",
                    "release_date_precision": "day",
                    "type": "album",
                    "uri": "spotify:album:6u9vLxqiEzXAd9VE5zi2EN",
                    "artists": [
                        {
                            "external_urls": {
                                "spotify": "https://open.spotify.com/artist/4l8xPGtl6DHR2uvunqrl8r"
                            },
                            "href": "https://api.spotify.com/v1/artists/4l8xPGtl6DHR2uvunqrl8r",
                            "id": "4l8xPGtl6DHR2uvunqrl8r",
                            "name": "CAN",
                            "type": "artist",
                            "uri": "spotify:artist:4l8xPGtl6DHR2uvunqrl8r"
                        }
                    ],
                    "album_group": "album"
                },
                {
                    "album_type": "album",
                    "total_tracks": 4,
                    "external_urls": {
                        "spotify": "https://open.spotify.com/album/5Ju5X4yL4nwWpflI0xlTgB"
                    },
                    "href": "https://api.spotify.com/v1/albums/5Ju5X4yL4nwWpflI0xlTgB",
                    "id": "5Ju5X4yL4nwWpflI0xlTgB",
                    "name": "LIVE IN ASTON 1977",
                    "release_date": "2024-05-31",
                    "release_date_precision": "day",
                    "type": "album",
                    "uri": "spotify:album:5Ju5X4yL4nwWpflI0xlTgB",
                    "artists": [
                        {
                            "external_urls": {
                                "spotify": "https://open.spotify.com/artist/4l8xPGtl6DHR2uvunqrl8r"
                            },
                            "href": "https://api.spotify.com/v1/artists/4l8xPGtl6DHR2uvunqrl8r",
                            "id": "4l8xPGtl6DHR2uvunqrl8r",
                            "name": "CAN",
                            "type": "artist",
                            "uri": "spotify:artist:4l8xPGtl6DHR2uvunqrl8r"
                        }
                    ],
                    "album_group": "album"
                }
            ]
        }

    def current_user_playlists(self, limit: int) -> Json: 
        return {
            "href": "https://api.spotify.com/v1/users/1192119558/playlists?offset=0&limit=2",
            "limit": 2,
            "next": "https://api.spotify.com/v1/users/1192119558/playlists?offset=2&limit=2",
            "offset": 0,
            "previous": None,
            "total": 52,
            "items": [
                {
                    "collaborative": False,
                    "description": "",
                    "external_urls": {
                        "spotify": "https://open.spotify.com/playlist/3ZqbMEnImUVSfX1llarkrT"
                    },
                    "href": "https://api.spotify.com/v1/playlists/3ZqbMEnImUVSfX1llarkrT",
                    "id": "3ZqbMEnImUVSfX1llarkrT",
                    "name": "D3S3RT DR1V3",
                    "owner": {
                        "display_name": "Robert van der Leeuw",
                        "external_urls": {
                            "spotify": "https://open.spotify.com/user/1192119558"
                        },
                        "href": "https://api.spotify.com/v1/users/1192119558",
                        "id": "1192119558",
                        "type": "user",
                        "uri": "spotify:user:1192119558"
                    },
                    "primary_color": None,
                    "public": True,
                    "snapshot_id": "AAAAGh4wh9lSmQwdoYrUCuk60HMF0PrR",
                    "tracks": {
                        "href": "https://api.spotify.com/v1/playlists/3ZqbMEnImUVSfX1llarkrT/tracks",
                        "total": 14
                    },
                    "type": "playlist",
                    "uri": "spotify:playlist:3ZqbMEnImUVSfX1llarkrT"
                },
                {
                    "collaborative": False,
                    "description": "",
                    "external_urls": {
                        "spotify": "https://open.spotify.com/playlist/2tYFkNLp2jFrsg7ReyiEOE"
                    },
                    "href": "https://api.spotify.com/v1/playlists/2tYFkNLp2jFrsg7ReyiEOE",
                    "id": "2tYFkNLp2jFrsg7ReyiEOE",
                    "images": [
                        {
                            "height": None,
                            "url": "https://image-cdn-ak.spotifycdn.com/image/ab67706c0000da8422da05884f196f6d956320d9",
                            "width": None
                        }
                    ],
                    "name": "N1GHT DR1V3",
                    "owner": {
                        "display_name": "Robert van der Leeuw",
                        "external_urls": {
                            "spotify": "https://open.spotify.com/user/1192119558"
                        },
                        "href": "https://api.spotify.com/v1/users/1192119558",
                        "id": "1192119558",
                        "type": "user",
                        "uri": "spotify:user:1192119558"
                    },
                    "primary_color": None,
                    "public": True,
                    "snapshot_id": "AAAAIIvk5a8V1ZQHNgLgve8WjgM0xf3J",
                    "tracks": {
                        "href": "https://api.spotify.com/v1/playlists/2tYFkNLp2jFrsg7ReyiEOE/tracks",
                        "total": 19
                    },
                    "type": "playlist",
                    "uri": "spotify:playlist:2tYFkNLp2jFrsg7ReyiEOE"
                }
            ]
        }

    def search(self, query: str, type: str) -> Json:
        return {
        "tracks": {
            "href": "https://api.spotify.com/v1/search?offset=0&limit=2&query=CAN%20-%20Spoon&type=track",
            "limit": 2,
            "next": "https://api.spotify.com/v1/search?offset=2&limit=2&query=CAN%20-%20Spoon&type=track",
            "offset": 0,
            "previous": None,
            "total": 899,
            "items": [
                {
                    "album": {
                        "album_type": "album",
                        "artists": [
                            {
                                "external_urls": {
                                    "spotify": "https://open.spotify.com/artist/4l8xPGtl6DHR2uvunqrl8r"
                                },
                                "href": "https://api.spotify.com/v1/artists/4l8xPGtl6DHR2uvunqrl8r",
                                "id": "4l8xPGtl6DHR2uvunqrl8r",
                                "name": "CAN",
                                "type": "artist",
                                "uri": "spotify:artist:4l8xPGtl6DHR2uvunqrl8r"
                            }
                        ],
                        "external_urls": {
                            "spotify": "https://open.spotify.com/album/1MLxE2czxo5A9OVZ2m8FV3"
                        },
                        "href": "https://api.spotify.com/v1/albums/1MLxE2czxo5A9OVZ2m8FV3",
                        "id": "1MLxE2czxo5A9OVZ2m8FV3",
                        "is_playable": True,
                        "name": "Ege Bamyasi (Remastered)",
                        "release_date": "1972-01-01",
                        "release_date_precision": "day",
                        "total_tracks": 7,
                        "type": "album",
                        "uri": "spotify:album:1MLxE2czxo5A9OVZ2m8FV3"
                    },
                    "artists": [
                        {
                            "external_urls": {
                                "spotify": "https://open.spotify.com/artist/4l8xPGtl6DHR2uvunqrl8r"
                            },
                            "href": "https://api.spotify.com/v1/artists/4l8xPGtl6DHR2uvunqrl8r",
                            "id": "4l8xPGtl6DHR2uvunqrl8r",
                            "name": "CAN",
                            "type": "artist",
                            "uri": "spotify:artist:4l8xPGtl6DHR2uvunqrl8r"
                        }
                    ],
                    "disc_number": 1,
                    "duration_ms": 184493,
                    "explicit": False,
                    "external_ids": {
                        "isrc": "DEX170420807"
                    },
                    "external_urls": {
                        "spotify": "https://open.spotify.com/track/0tno4OXs8e3rquqKMkF9XM"
                    },
                    "href": "https://api.spotify.com/v1/tracks/0tno4OXs8e3rquqKMkF9XM",
                    "id": "0tno4OXs8e3rquqKMkF9XM",
                    "is_local": False,
                    "is_playable": True,
                    "name": "Spoon",
                    "popularity": 32,
                    "preview_url": None,
                    "track_number": 7,
                    "type": "track",
                    "uri": "spotify:track:0tno4OXs8e3rquqKMkF9XM"
                },
                {
                    "album": {
                        "album_type": "album",
                        "artists": [
                            {
                                "external_urls": {
                                    "spotify": "https://open.spotify.com/artist/0K1q0nXQ8is36PzOKAMbNe"
                                },
                                "href": "https://api.spotify.com/v1/artists/0K1q0nXQ8is36PzOKAMbNe",
                                "id": "0K1q0nXQ8is36PzOKAMbNe",
                                "name": "Spoon",
                                "type": "artist",
                                "uri": "spotify:artist:0K1q0nXQ8is36PzOKAMbNe"
                            }
                        ],
                        "external_urls": {
                            "spotify": "https://open.spotify.com/album/1pitNtT99leODbWecrt7XJ"
                        },
                        "href": "https://api.spotify.com/v1/albums/1pitNtT99leODbWecrt7XJ",
                        "id": "1pitNtT99leODbWecrt7XJ",
                        "is_playable": True,
                        "name": "Girls Can Tell",
                        "release_date": "2001-02-20",
                        "release_date_precision": "day",
                        "total_tracks": 11,
                        "type": "album",
                        "uri": "spotify:album:1pitNtT99leODbWecrt7XJ"
                    },
                    "artists": [
                        {
                            "external_urls": {
                                "spotify": "https://open.spotify.com/artist/0K1q0nXQ8is36PzOKAMbNe"
                            },
                            "href": "https://api.spotify.com/v1/artists/0K1q0nXQ8is36PzOKAMbNe",
                            "id": "0K1q0nXQ8is36PzOKAMbNe",
                            "name": "Spoon",
                            "type": "artist",
                            "uri": "spotify:artist:0K1q0nXQ8is36PzOKAMbNe"
                        }
                    ],
                    "disc_number": 1,
                    "duration_ms": 244240,
                    "explicit": False,
                    "external_ids": {
                        "isrc": "USMRG0340006"
                    },
                    "external_urls": {
                        "spotify": "https://open.spotify.com/track/7iUlymgfGurgnsFlek7Djo"
                    },
                    "href": "https://api.spotify.com/v1/tracks/7iUlymgfGurgnsFlek7Djo",
                    "id": "7iUlymgfGurgnsFlek7Djo",
                    "is_local": False,
                    "is_playable": True,
                    "name": "Everything Hits At Once",
                    "popularity": 47,
                    "preview_url": None,
                    "track_number": 1,
                    "type": "track",
                    "uri": "spotify:track:7iUlymgfGurgnsFlek7Djo"
                }
            ]
        }
    }

    # Putting in an intricate mock chunker for all types is above my pay grade (open source).
    def next(self, chunk: Json) -> Json: return {"items": []}

    def current_playback(self) -> Json: 
        return {
            "device": {
                "id": "783a78c05302cd4c0e8561291d909728a8f58c39",
                "is_active": True,
                "is_private_session": False,
                "is_restricted": False,
                "name": "nixos",
                "supports_volume": True,
                "type": "Computer",
                "volume_percent": 100
            },
            "shuffle_state": False,
            "smart_shuffle": False,
            "repeat_state": "context",
            "is_playing": True,
            "timestamp": 1754003701073,
            "context": None,
            "progress_ms": 233933,
            "item": {
                "album": {
                    "album_type": "album",
                    "artists": [
                        {
                            "external_urls": {
                                "spotify": "https://open.spotify.com/artist/0yNLKJebCb8Aueb54LYya3"
                            },
                            "href": "https://api.spotify.com/v1/artists/0yNLKJebCb8Aueb54LYya3",
                            "id": "0yNLKJebCb8Aueb54LYya3",
                            "name": "New Order",
                            "type": "artist",
                            "uri": "spotify:artist:0yNLKJebCb8Aueb54LYya3"
                        }
                    ],
                    "external_urls": {
                        "spotify": "https://open.spotify.com/album/6iHuSGy6pq4tNGFV3ZVPtl"
                    },
                    "href": "https://api.spotify.com/v1/albums/6iHuSGy6pq4tNGFV3ZVPtl",
                    "id": "6iHuSGy6pq4tNGFV3ZVPtl",
                    "name": "Substance",
                    "release_date": "1987-08-17",
                    "release_date_precision": "day",
                    "total_tracks": 24,
                    "type": "album",
                    "uri": "spotify:album:6iHuSGy6pq4tNGFV3ZVPtl"
                },
                "artists": [
                    {
                        "external_urls": {
                            "spotify": "https://open.spotify.com/artist/0yNLKJebCb8Aueb54LYya3"
                        },
                        "href": "https://api.spotify.com/v1/artists/0yNLKJebCb8Aueb54LYya3",
                        "id": "0yNLKJebCb8Aueb54LYya3",
                        "name": "New Order",
                        "type": "artist",
                        "uri": "spotify:artist:0yNLKJebCb8Aueb54LYya3"
                    }
                ],
                "disc_number": 1,
                "duration_ms": 449160,
                "explicit": False,
                "external_ids": {
                    "isrc": "GBAAP0001115"
                },
                "external_urls": {
                    "spotify": "https://open.spotify.com/track/6hHc7Pks7wtBIW8Z6A0iFq"
                },
                "href": "https://api.spotify.com/v1/tracks/6hHc7Pks7wtBIW8Z6A0iFq",
                "id": "6hHc7Pks7wtBIW8Z6A0iFq",
                "is_local": False,
                "name": "Blue Monday",
                "popularity": 73,
                "preview_url": None,
                "track_number": 4,
                "type": "track",
                "uri": "spotify:track:6hHc7Pks7wtBIW8Z6A0iFq"
            },
            "currently_playing_type": "track",
            "actions": {
                "disallows": {
                    "resuming": True
                }
            }
        }

    def queue(self) -> Json: 
        return {
            "queue": [
                {
                    "album": {
                        "album_type": "album",
                        "artists": [
                            {
                                "external_urls": {
                                    "spotify": "https://open.spotify.com/artist/0k17h0D3J5VfsdmQ1iZtE9"
                                },
                                "href": "https://api.spotify.com/v1/artists/0k17h0D3J5VfsdmQ1iZtE9",
                                "id": "0k17h0D3J5VfsdmQ1iZtE9",
                                "name": "Pink Floyd",
                                "type": "artist",
                                "uri": "spotify:artist:0k17h0D3J5VfsdmQ1iZtE9"
                            }
                        ],
                        "external_urls": {
                            "spotify": "https://open.spotify.com/album/5Dbax7G8SWrP9xyzkOvy2F"
                        },
                        "href": "https://api.spotify.com/v1/albums/5Dbax7G8SWrP9xyzkOvy2F",
                        "id": "5Dbax7G8SWrP9xyzkOvy2F",
                        "name": "The Wall",
                        "release_date": "1979-11-30",
                        "release_date_precision": "day",
                        "total_tracks": 26,
                        "type": "album",
                        "uri": "spotify:album:5Dbax7G8SWrP9xyzkOvy2F"
                    },
                    "artists": [
                        {
                            "external_urls": {
                                "spotify": "https://open.spotify.com/artist/0k17h0D3J5VfsdmQ1iZtE9"
                            },
                            "href": "https://api.spotify.com/v1/artists/0k17h0D3J5VfsdmQ1iZtE9",
                            "id": "0k17h0D3J5VfsdmQ1iZtE9",
                            "name": "Pink Floyd",
                            "type": "artist",
                            "uri": "spotify:artist:0k17h0D3J5VfsdmQ1iZtE9"
                        }
                    ],
                    "disc_number": 1,
                    "duration_ms": 238746,
                    "explicit": False,
                    "external_ids": {
                        "isrc": "GBN9Y1100099"
                    },
                    "external_urls": {
                        "spotify": "https://open.spotify.com/track/4gMgiXfqyzZLMhsksGmbQV"
                    },
                    "href": "https://api.spotify.com/v1/tracks/4gMgiXfqyzZLMhsksGmbQV",
                    "id": "4gMgiXfqyzZLMhsksGmbQV",
                    "is_local": False,
                    "name": "Another Brick in the Wall, Pt. 2",
                    "popularity": 84,
                    "preview_url": None,
                    "track_number": 5,
                    "type": "track",
                    "uri": "spotify:track:4gMgiXfqyzZLMhsksGmbQV"
                },
                {
                    "album": {
                        "album_type": "album",
                        "artists": [
                            {
                                "external_urls": {
                                    "spotify": "https://open.spotify.com/artist/1KHtjJV5UNijjzuAvyv5Wm"
                                },
                                "href": "https://api.spotify.com/v1/artists/1KHtjJV5UNijjzuAvyv5Wm",
                                "id": "1KHtjJV5UNijjzuAvyv5Wm",
                                "name": "Mote",
                                "type": "artist",
                                "uri": "spotify:artist:1KHtjJV5UNijjzuAvyv5Wm"
                            }
                        ],
                        "external_urls": {
                            "spotify": "https://open.spotify.com/album/5p4X11tphJoaVPyEQn9mgC"
                        },
                        "href": "https://api.spotify.com/v1/albums/5p4X11tphJoaVPyEQn9mgC",
                        "id": "5p4X11tphJoaVPyEQn9mgC",
                        "name": "Samalas",
                        "release_date": "2019-03-14",
                        "release_date_precision": "day",
                        "total_tracks": 7,
                        "type": "album",
                        "uri": "spotify:album:5p4X11tphJoaVPyEQn9mgC"
                    },
                    "artists": [
                        {
                            "external_urls": {
                                "spotify": "https://open.spotify.com/artist/1KHtjJV5UNijjzuAvyv5Wm"
                            },
                            "href": "https://api.spotify.com/v1/artists/1KHtjJV5UNijjzuAvyv5Wm",
                            "id": "1KHtjJV5UNijjzuAvyv5Wm",
                            "name": "Mote",
                            "type": "artist",
                            "uri": "spotify:artist:1KHtjJV5UNijjzuAvyv5Wm"
                        }
                    ],
                    "disc_number": 1,
                    "duration_ms": 448613,
                    "explicit": False,
                    "external_ids": {
                        "isrc": "uscgj1964553"
                    },
                    "external_urls": {
                        "spotify": "https://open.spotify.com/track/040bc9LaTiSb0EwAJ01brc"
                    },
                    "href": "https://api.spotify.com/v1/tracks/040bc9LaTiSb0EwAJ01brc",
                    "id": "040bc9LaTiSb0EwAJ01brc",
                    "is_local": False,
                    "name": "Hollow (Bonus Track)",
                    "popularity": 3,
                    "preview_url": None,
                    "track_number": 7,
                    "type": "track",
                    "uri": "spotify:track:040bc9LaTiSb0EwAJ01brc"
                }
            ]
        }


from spotipy.oauth2 import SpotifyOAuth
class spotipy_fake:
    def Spotify(self, auth: SpotifyOAuth):
        return Spotifake()

from spotdl.types.song import Song
import soundfile as sf
class Spotdl:
    def __init__(self, no_cache: bool, spotify_client=None, downloader_settings=None, loop=None):
        pass

    def search(self, spotify_urls: list[str]): 
        return [
            Song(name='Halleluhwah',
                 artists=['CAN'],
                 artist='CAN',
                 genres=['krautrock',
                         'space rock',
                         'progressive rock',
                         'experimental',
                         'psychedelic rock',
                         'art rock',
                         'proto-punk'],
                 disc_number=1,
                 disc_count=1,
                 album_name='Tago Mago (Remastered)',
                 album_artist='CAN',
                 duration=1112,
                 year=1971,
                 date='1971-01-01',
                 track_number=4,
                 tracks_count=7,
                 song_id='0E5fRGNiCT8C0C6BNPb3qh',
                 explicit=False,
                 publisher='Mute',
                 url='https://open.spotify.com/track/0E5fRGNiCT8C0C6BNPb3qh',
                 isrc='DEX170420604',
                 cover_url='https://i.scdn.co/image/ab67616d0000b2734c8df6e776670be529fef096',
                 copyright_text='2004 2011 Spoon Records',
                 download_url=None,
                 lyrics=None,
                 popularity=28,
                 album_id='5iWtrhvwOCWnHN14rqub04',
                 list_name=None,
                 list_url=None,
                 list_position=None,
                 list_length=None,
                 artist_id='4l8xPGtl6DHR2uvunqrl8r',
                 album_type='album',
                 spotify_id=spotify_url.replace("https://open.spotify.com/track/", ""))  # For unique filenames later.
            for spotify_url in spotify_urls
        ]

    def download(self, song: Song) -> tuple[Song, str]:
        mock_path = "./mock_downloads"

        if not os.path.isdir(mock_path):
            os.mkdir(mock_path)

        file_path = f"{mock_path}/{song.spotify_id}.wav"

        duration_s = 30.0 
        num_samples = int(duration_s * SAMPLE_RATE)
        
        # Random audio (stereo)
        audio_data = np.random.uniform(-0.1, 0.1, (num_samples, 2)).astype(np.float32)
        
        sf.write(file_path, audio_data, SAMPLE_RATE)

        return song, file_path

