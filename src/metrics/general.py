from objects import Metric, TrainEvent


class Reward(Metric):
    name = "Rewards"
    on_event = TrainEvent.REWARD_OBSERVED

    def calc(self, reward: float):
        self.values.append(reward)

class GoodSongsLeft(Metric):
    name = "Good songs left in dataset"
    on_event = TrainEvent.REWARD_OBSERVED

    def __init__(self, good_songs: int=0, **kwargs):
        super().__init__()
        self.values.append(good_songs)

    def calc_inner(self, reward: int):
        self.values.append(self.values[-1] - reward)

class SongsPicked(Metric):
    name = "Songs picked during trajectory."
    on_event = TrainEvent.STEP_END

    def calc_inner(self, picked: str):
        self.values.append(picked)
