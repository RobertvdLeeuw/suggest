from suggester.objects import Metric, TrainEvent


class SongsPicked(Metric):
    name = "Songs picked during trajectory."
    id = "songs_picked"
    on_event = TrainEvent.STEP_END
    
    metric_schema = {
        "type": "object",
        "properties": {
            "picked_songs": {
                "type": "array",
                "items": {"type": "integer"}
            },
            "step": {"type": "integer"}
        },
        "required": ["picked_songs", "step"]
    }

    arguments = ["picked"]
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.step = 0

    def calc_inner(self, picked: list[str]):
        self.values.append({
            "picked_songs": list(picked),
            "step": self.step
        })
        self.step += 1


class Reward(Metric):
    name = "Rewards observed during trajectory"
    id = "rewards"
    on_event = TrainEvent.REWARD_OBSERVED
    
    metric_schema = {
        "type": "object", 
        "properties": {
            "reward": {"type": "number"},
            "song_id": {"type": "integer"},
            "step": {"type": "integer"}
        },
        "required": ["reward", "song_id", "step"]
    }

    arguments = ["reward", "song_id"]
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.step = 0

    def calc_inner(self, reward: float, song_id: int):
        self.values.append({
            "reward": float(reward),
            "song_id": int(song_id), 
            "step": self.step
        })
        self.step += 1
