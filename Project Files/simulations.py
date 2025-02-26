import itertools
from copy import deepcopy
from math import sqrt, log10
import numpy as np
from bandits import BlockingBanditArm

SMALL_DELAYS = "Small Delays"
LARGE_DELAYS = "Large Delays"
FIXED_DELAYS = "Fixed Delays"


class BlockingBanditSimulation:
    def __init__(self, K, T, sim_type, fixed_delays, seed):
        self.K, self.T = K, T
        self.sim_type = sim_type
        self.fixed_delays = fixed_delays
        np.random.seed(seed)

        self.delta_mean_lower_bound, self.delta_mean_upper_bound = K / 2000, K / 400
        self.small_blocking_delay_lower_bound, self.small_blocking_delay_upper_bound = 1, T // 1000
        self.large_blocking_delay_lower_bound, self.large_blocking_delay_upper_bound = (T // 1000) + 1, T // 500

        self.bandit_arms_oracle, self.bandit_arms_UCB = self.generate_bandits()

        self.estimated_mean_rewards = [0 for _ in range(self.K)]

        self.null_bandit_arm = BlockingBanditArm(0, 0)

    def generate_bandits(self):
        delta_means = sorted(
            list(np.random.uniform(self.delta_mean_lower_bound, self.delta_mean_upper_bound, self.K - 1)))
        mean_rewards = [0] + list(itertools.accumulate(delta_means))

        if self.sim_type == SMALL_DELAYS:
            blocking_delays = list(np.random.choice(np.arange(self.small_blocking_delay_lower_bound,
                                                              self.small_blocking_delay_upper_bound + 1),
                                                    size=self.K, replace=True))

        elif self.sim_type == LARGE_DELAYS:
            blocking_delays = list(np.random.choice(np.arange(self.large_blocking_delay_lower_bound,
                                                              self.large_blocking_delay_upper_bound + 1),
                                                    size=self.K, replace=True))
        else:
            blocking_delays = [self.fixed_delays for _ in range(self.K)]

        bandit_arms_oracle = [BlockingBanditArm(mean_rewards[i], blocking_delays[i]) for i in range(self.K)]
        bandit_arms_UCB = deepcopy(bandit_arms_oracle)

        return tuple(reversed(bandit_arms_oracle)), tuple(reversed(bandit_arms_UCB))

    def oracle_greedy_algorithm(self):
        available_arms = [arm for arm in self.bandit_arms_oracle if arm.is_available]

        if not available_arms:
            return 0, self.null_bandit_arm

        mean_rewards = [(arm.mean_reward, i) for i, arm in enumerate(available_arms)]
        mean_rewards = sorted(mean_rewards, key=lambda pair: pair[0], reverse=True)

        best_arm, best_arm_idx = available_arms[mean_rewards[0][1]], mean_rewards[0][1]
        reward = best_arm.sample_reward()

        best_arm.is_available = False
        self.bandit_arms_oracle[best_arm_idx].num_uses += 1
        best_arm.remaining_rounds_before_use = best_arm.blocking_delay

        return reward, best_arm

    def UCB_greedy_algorithm(self, timestamp):
        if timestamp < self.K:
            reward = self.bandit_arms_UCB[timestamp].sample_reward()
            self.bandit_arms_UCB[timestamp].is_available = False
            self.bandit_arms_UCB[timestamp].remaining_rounds_before_use = self.bandit_arms_UCB[timestamp].blocking_delay
            self.bandit_arms_UCB[timestamp].num_uses += 1
            self.estimated_mean_rewards[timestamp] = reward
            return reward, self.bandit_arms_UCB[timestamp]

        bandits_arms_with_indices = [(arm, i) for i, arm in enumerate(self.bandit_arms_UCB)]
        available_arms = [(arm, i) for (arm, i) in bandits_arms_with_indices if arm.is_available]

        if not available_arms:
            return 0, self.null_bandit_arm

        UCB_arms_indices = [(self.estimated_mean_rewards[i] + sqrt((8 * log10(timestamp)) / arm.num_uses), i)
                            for (arm, i) in available_arms]
        UCB_arms_indices = sorted(UCB_arms_indices, key=lambda pair: pair[0], reverse=True)

        best_arm, best_idx = self.bandit_arms_UCB[UCB_arms_indices[0][1]], UCB_arms_indices[0][1]
        reward = best_arm.sample_reward()

        self.estimated_mean_rewards[best_idx] = (self.estimated_mean_rewards[best_idx] * self.bandit_arms_UCB[best_idx].
                                                 num_uses + reward) / (self.bandit_arms_UCB[best_idx].num_uses + 1)
        best_arm.is_available = False
        best_arm.remaining_rounds_before_use = best_arm.blocking_delay
        self.bandit_arms_UCB[best_idx].num_uses += 1
        return reward, best_arm

    def update_unavailable_arms(self):
        unavailable_arms_oracle = [arm for arm in self.bandit_arms_oracle if not arm.is_available]
        unavailable_arms_UCB = [arm for arm in self.bandit_arms_UCB if not arm.is_available]

        for unavailable_list in [unavailable_arms_oracle, unavailable_arms_UCB]:
            for arm in unavailable_list:
                arm.remaining_rounds_before_use -= 1

                if arm.remaining_rounds_before_use == 0:
                    arm.is_available = True

    def simulate(self):
        oracle_cumulative_rewards, UCB_cumulative_rewards = [0], [0]
        oracle_chosen_arms_per_timeslots, UCB_chosen_arms_per_timeslots = [], []

        for timeslot in range(self.T):
            oracle_reward, oracle_chosen_arm = self.oracle_greedy_algorithm()
            UCB_reward, UCB_chosen_arm = self.UCB_greedy_algorithm(timeslot)
            self.update_unavailable_arms()

            oracle_cumulative_rewards.append(oracle_cumulative_rewards[-1] + oracle_reward)
            UCB_cumulative_rewards.append(UCB_cumulative_rewards[-1] + UCB_reward)

            oracle_chosen_arms_per_timeslots.append(oracle_chosen_arm)
            UCB_chosen_arms_per_timeslots.append(UCB_chosen_arm)

        return oracle_cumulative_rewards[1:], UCB_cumulative_rewards[1:], oracle_chosen_arms_per_timeslots, \
               UCB_chosen_arms_per_timeslots

    def calculate_cumulative_regret(self, ucb_arms):
        self.reset_simulation()
        cumulative_regret_ucb = [0]

        for timeslot in range(self.T):
            available_arms_ucb = [arm for arm in self.bandit_arms_UCB if arm.is_available]
            sorted_mean_reward_available_arms_ucb = sorted(available_arms_ucb, key=lambda arm: arm.mean_reward,
                                                           reverse=True)
            best_arm_ucb = sorted_mean_reward_available_arms_ucb[0]

            regret_ucb = best_arm_ucb.mean_reward - ucb_arms[timeslot].mean_reward
            cumulative_regret_ucb.append(cumulative_regret_ucb[-1] + regret_ucb)

            best_arm_ucb.is_available = False
            best_arm_ucb.remaining_rounds_before_use = best_arm_ucb.blocking_delay
            best_arm_ucb.num_uses += 1

            self.update_unavailable_arms()

        return cumulative_regret_ucb[1:]

    def reset_simulation(self):
        for arm in self.bandit_arms_UCB:
            arm.num_uses = 0
            arm.remaining_rounds_before_use = 0
            arm.is_available = True

        for arm in self.bandit_arms_oracle:
            arm.num_uses = 0
            arm.remaining_rounds_before_use = 0
            arm.is_available = True

    def calculate_k_star(self):
        sum_inverted_delays = 0
        k_star = 0

        while sum_inverted_delays < 1:
            sum_inverted_delays += 1 / self.bandit_arms_UCB[k_star].blocking_delay
            k_star += 1

        return k_star

    def calculate_k_g(self, oracle_chosen_arms):
        filtered_sorted_arms = sorted(filter(lambda arm: arm.mean_reward > 0, oracle_chosen_arms),
                                      key=lambda arm: arm.mean_reward)
        worst_chosen_arm = filtered_sorted_arms[0]

        for i, arm in enumerate(self.bandit_arms_UCB):
            if arm == worst_chosen_arm:
                return i + 1
