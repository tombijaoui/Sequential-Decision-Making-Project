import argparse
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from simulations import BlockingBanditSimulation
from tqdm import tqdm

matplotlib.use('TkAgg')

SMALL_DELAYS = "Small Delays"
LARGE_DELAYS = "Large Delays"
FIXED_DELAYS = "Fixed Delays"


def parse_arguments():
    parser = argparse.ArgumentParser(description="Simulations for Blocking Bandits Algorithms")

    parser.add_argument("-K", type=int, help="Number of arms", required=True)
    parser.add_argument("-T", type=int, help="Number of rounds", required=True)
    parser.add_argument("-seed", type=int, help="Seed", required=False)
    parser.add_argument("-fixed_delay", type=int, help="Fixed delay for all arms", required=True)
    parser.add_argument("-num_sim", type=int, help="Number of Simulations", required=True)

    args = parser.parse_args()

    return args


def plot_graph_cumulative_rewards(simulation_types, results_values, algorithms, T, k_stars_types, k_g_types):
    for sim_type in simulation_types:
        for alg in algorithms:
            plt.plot(list(range(T)), results_values[sim_type][alg], marker='o', linestyle='-',
                     label=f"{alg}, {sim_type.lower()}, K*={k_stars_types[sim_type]}, Kg={k_g_types[sim_type]}",
                     markersize=1)

    plt.title("Cumulative Reward by simulation types and algorithms")
    plt.legend()

    plt.show()


def plot_graph_cumulative_regrets(simulation_types, regrets_values, T, k_stars_types, k_g_types):
    for sim_type in simulation_types:
        plt.plot(list(range(T)), regrets_values[sim_type], marker='o', linestyle='',
                 label=f"UCB, {sim_type.lower()}, K*={k_stars_types[sim_type]}, Kg={k_g_types[sim_type]}", markersize=1)

    plt.title("Cumulative UCB Algorithm Regret by simulation type")
    plt.legend()

    plt.show()


def deterministic_delays_simulations(simulation_types, K, T, fixed_delays, num_sims, seed):
    results = {sim_type: {"UCB": 0, "Oracle": 0} for sim_type in simulation_types}
    regrets = {sim_type: 0 for sim_type in simulation_types}

    k_stars_types = {sim_type: 0 for sim_type in simulation_types}
    k_g_types = {sim_type: 0 for sim_type in simulation_types}

    for sim_type in simulation_types:
        expected_oracle, expected_ucb = [], []
        expected_cumulative_regret_ucb = []

        for sim_num in tqdm(range(num_sims), desc=f"Simulate with {sim_type.lower()}"):
            blocking_bandit_simulation = BlockingBanditSimulation(K, T, sim_type, fixed_delays, seed)
            oracle, ucb, oracle_arms, ucb_arms = blocking_bandit_simulation.simulate()

            ucb_cumulative_regret = blocking_bandit_simulation.calculate_cumulative_regret(ucb_arms)

            expected_oracle.append(oracle)
            expected_ucb.append(ucb)

            expected_cumulative_regret_ucb.append(ucb_cumulative_regret)

            if sim_num == 0:
                k_stars_types[sim_type] = blocking_bandit_simulation.calculate_k_star()
                k_g_types[sim_type] = blocking_bandit_simulation.calculate_k_g(oracle_arms)

        expected_oracle = [sum(values) / num_sims for values in zip(*expected_oracle)]
        expected_ucb = [sum(values) / num_sims for values in zip(*expected_ucb)]

        expected_cumulative_regret_ucb = [sum(values) / num_sims for values in zip(*expected_cumulative_regret_ucb)]

        results[sim_type]["Oracle"] = expected_oracle
        results[sim_type]["UCB"] = expected_ucb

        regrets[sim_type] = expected_cumulative_regret_ucb

    plot_graph_cumulative_rewards(simulation_types, results, ["UCB", "Oracle"], T, k_stars_types, k_g_types)
    plot_graph_cumulative_regrets(simulation_types, regrets, T, k_stars_types, k_g_types)


if __name__ == '__main__':
    simulation_args = parse_arguments()
    K, T, fixed_delays, num_sims, seed = simulation_args.K, simulation_args.T, simulation_args.fixed_delay, \
                                         simulation_args.num_sim, simulation_args.seed
    np.random.seed(seed)
    simulation_types = [SMALL_DELAYS, LARGE_DELAYS, FIXED_DELAYS]

    deterministic_delays_simulations(simulation_types, K, T, fixed_delays, num_sims, seed)


