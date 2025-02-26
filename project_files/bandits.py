import numpy as np


class BanditArm:
    def __init__(self, mean_reward):
        self.mean_reward = mean_reward
        self.num_uses = 0

    def __repr__(self):
        return f"(Bandit: Mean Reward = {self.mean_reward})"

    def __eq__(self, other):
        if isinstance(self, BanditArm):
            return self.mean_reward == other.mean_reward
        return False

    def sample_reward(self):
        reward = np.random.binomial(n=1, p=self.mean_reward)
        return reward


class BlockingBanditArm(BanditArm):
    def __init__(self, mean_reward, blocking_delay):
        super().__init__(mean_reward)
        self.blocking_delay = blocking_delay
        self.remaining_rounds_before_use = 0
        self.is_available = True

    def __repr__(self):
        return "(Blocking " + (super().__repr__())[1:-1] + f", Delay = {self.blocking_delay})"

    def __eq__(self, other):
        if isinstance(self, BlockingBanditArm):
            return self.mean_reward == other.mean_reward and self.blocking_delay == other.blocking_delay
        return False

    def sample_reward(self):
        reward = super().sample_reward() if self.is_available else 0
        return reward


