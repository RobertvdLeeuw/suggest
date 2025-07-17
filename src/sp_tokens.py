import spotipy
from spotipy.oauth2 import SpotifyOAuth
import os

from dotenv import load_dotenv
load_dotenv()

# Your Spotify app credentials
CLIENT_ID = os.environ["SPOTIFY_CLIENT_ID"]
CLIENT_SECRET = os.environ["SPOTIFY_CLIENT_SECRET"]
REDIRECT_URI = "http://localhost:8080/callback"
SCOPES = ['user-library-read',
          'playlist-read-private',
          'playlist-read-collaborative',
          'user-read-currently-playing',
          'user-read-playback-state']

sp_oauth = SpotifyOAuth(
    client_id=CLIENT_ID,
    client_secret=CLIENT_SECRET,
    redirect_uri=REDIRECT_URI,
    scope=SCOPES
)

# This will open your browser for authentication
token_info = sp_oauth.get_access_token()
print(f"Refresh Token: {token_info['refresh_token']}")
print(f"Access Token: {token_info['access_token']}")
