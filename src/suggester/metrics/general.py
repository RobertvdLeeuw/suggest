from suggester.objects import Metric, TrainEvent


# class Reward(Metric):
#     name = "Rewards"
#     on_event = TrainEvent.REWARD_OBSERVED

#     arguments = ["reward"]

#     def calc(self, reward: float):
#         self.values.append(reward)

class SongsPicked(Metric):
    name = "Songs picked during trajectory."
    on_event = TrainEvent.STEP_END

    arguments = ["picked"]

    def calc_inner(self, picked: list[str]):
        self.values.append(picked)
