from spotdl import Spotdl
from spotdl.types.options import DownloaderOptions

DOWNLOAD_LOC = "./downloads"
spotdl = Spotdl(
    no_cache=True,
    client_id="a766651ba4b744ed82f1e520a75b2455",
    client_secret="767732da0b064b838ebe5d0e3f6ce4eb",
    downloader_settings=DownloaderOptions(format="wav", output=DOWNLOAD_LOC)
)

print(spotdl.search(["https://open.spotify.com/track/021su98MW3QWMbO0RRQXIR"]))
