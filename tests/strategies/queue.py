from hypothesis import strategies as st
from collecter.embedders import SongQueue, QueueObject
from collecter.downloader import QUEUE_MAX_LEN

@st.composite
def queue(draw, state: str, *, q_type: QueueObject = None, refills=False):
    q = SongQueue()  # Name, q_type
    match state:
        case "empty": 

        case "partial":

        case "full":

