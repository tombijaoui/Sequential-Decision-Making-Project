<h1 align="center">
  <img src="https://upload.wikimedia.org/wikipedia/commons/b/b7/Technion_logo.svg" alt="Technion Logo" height="100">
  <br>
  Sequential Decisions Making Learning - Final Project
</h1>

<p align="center">
  <em>Blocking Bandits</em>
</p>

<p align="center">
  <strong>Technion - Israel Institute of Technology</strong> <br>
  Faculty of Data Science and Decisions
</p>

<h1 align="center">
  <img src="https://github.com/tombijaoui/Sequential-Decision-Making-Project/blob/main/Pictures%20MAB/Crazy%20Squid%20MAB.png" alt="Crazy Squid MAB" height="200">
  <img src="https://github.com/tombijaoui/Sequential-Decision-Making-Project/blob/main/Pictures%20MAB/MAB%20Theory.png" alt="MAB Theory" height="200">
</h1>

---

<details open>
<summary><strong>Table of Contents</strong> ⚙️</summary>

1. [Link of the Article](#link-of-the-article)
2. [Project Overview](#project-overview)  
3. [About the code](#about-the-code)  
4. [Implemented Classes](#implemented--classes)
5. [Running Instructions](#running-instructions)  
6. [Results & Comments](#results-and-comments)  

</details>

---

## Link of the article
Our project and implementation is based on the following article:
<blockquote>
  <a href="https://proceedings.neurips.cc/paper_files/paper/2019/hash/88fee0421317424e4469f33a48f50cb0-Abstract.html">Blocking Bandits</a> by Soumya Basu, Rajat Sen, Sujay Sanghavi and Sanjay Shakkottai.</blockquote>

## Project Overview
Blocking Bandits represent a new variation in the multi-armed bandit (MAB) problem, where each arm, once played, becomes unavailable or blocked for a certain number of time slots before it can be played again. This models real-world scenarios such as job scheduling, resource allocation where actions or resources cannot be reused immediately due to limitations. Blocking bandits introduce significant complexity into decision-making algorithms, particularly in balancing exploration and exploitation. Traditional MAB problems focus on maximizing cumulative rewards while learning the best arm to play. However, with blocking constraints, the problem shifts—decisions made now affect future availability, making it harder to maintain an optimal strategy over time​.


## About the code
The code replicates the experimental results from the paper. The goal is to create a virtual environment for blocking bandits by reproducing the different simulations described in Section V of the paper. We replicated three types of simulations: the first involves arms with uniformly sampled short delays, the second with arms having uniformly sampled long delays, and the third with fixed delays. The code runs the two algorithms presented in the paper, namely Oracle Greedy and UCB Greedy, across these different simulations. It then generates graphs showing the evolution of cumulative rewards and cumulative regret for each algorithm depending on the type of simulation.


## Implemented Classes
- **bandits.py**
  - ### BanditArm Class:
    - `__init__(self, mean_reward)`: Initializes the bandit arm with the given mean reward and tracks the number of uses.
    - `__repr__(self)`: Returns a string representation of the bandit arm's mean reward.
    - `__eq__(self, other)`: Checks if two bandit arms have the same mean reward.
    - `sample_reward(self)`: Samples a binary reward from a binomial distribution based on the arm’s mean reward.

  - ### BlockingBanditArm Class (inherits from BanditArm):
    - `__init__(self, mean_reward, blocking_delay)`: Initializes a blocking bandit arm with a mean reward and a blocking delay.
    - `__repr__(self)`: Returns a string representation of the blocking bandit arm, including its blocking delay.
    - `__eq__(self, other)`: Checks if two blocking bandit arms have the same mean reward and blocking delay.
    - `sample_reward(self)`: Samples a reward if the arm is available, otherwise returns 0.

- **simulations.py**
  - ### BlockingBanditSimulation Class:

    - `__init__(self, K, T, sim_type, fixed_delays, seed)`: Initializes the simulation with the number of arms, horizon, type of simulation, fixed delays, and a random seed.
    - `generate_bandits(self)`: Generates bandit arms based on the type of simulation (small delays, large delays, or fixed delays).
    - `oracle_greedy_algorithm(self)`: Selects the best available arm based on the oracle greedy algorithm and updates its state.
    - `UCB_greedy_algorithm(self, timestamp)`: Implements the UCB greedy algorithm, selecting arms and updating their state based on the UCB index.
    - `update_unavailable_arms(self)`: Decreases the blocking delay countdown for unavailable arms and makes them available when the delay ends.
    - `simulate(self)`: Runs the simulation for T time steps, returning the cumulative rewards and the arms chosen by both algorithms.
    - `calculate_cumulative_regret(self, ucb_arms)`: Computes the cumulative regret for the UCB algorithm.
    - `reset_simulation(self)`: Resets the state of all arms, making them available for reuse.
    - `calculate_k_star(self)`: Calculates the optimal number of arms, `K*`, based on the sum of inverse delays.
    - `calculate_k_g(self, oracle_chosen_arms)`: Computes `K_g`, the number of arms used by the Oracle Greedy algorithm.

- **main.py**
  - `parse_arguments()`: Parses the command-line arguments such as the number of arms, rounds, fixed delay, number of simulations, and seed.
  - `plot_graph_cumulative_rewards(simulation_types, results_values, algorithms, T, k_stars_types, k_g_types)`: Plots the cumulative rewards for different algorithms and simulation types.
  - `plot_graph_cumulative_regrets(simulation_types, regrets_values, T, k_stars_types, k_g_types)`: Plots the cumulative regrets for the UCB algorithm across different simulation types.
  - `deterministic_delays_simulations(simulation_types, K, T, fixed_delays, num_sims, seed)`: Runs the simulations for different types of delays and collects results on rewards and regrets.
  - `__main__`: Parses the arguments, sets up the seed, and runs the deterministic delays simulations.


## Running Instructions
This repository contains the code and instructions to run the experiment presented in the associated paper. Please follow the steps below to set up and run the experiment.

### Prerequisites
Before running the experiment, you need to install all the required Python packages.

1. Download and install the dependencies by running the following command in your terminal:
`pip install -r requirements.txt`

2. Run the following command in your terminal to execute the experiment:
`python "your_local_path"/project_files/main.py -K 20 -T 10000 -seed 42 -fixed_delay 10 -num_sim 250`
Those hyperparameters match exactly those of the paper's experiment.

4. You can modify the values of the following parameters to generate different simulations:

- `-K`: The number of simulations to run. For example, `-K 20` will run 20 simulations.
- `-T`: The time duration for each simulation in arbitrary units. For example, `-T 10000` will run each simulation for 10,000 units of time.
- `-seed`: The random seed for reproducibility. For example, `-seed 42` will set the seed to 42.
- `-fixed_delay`: A fixed delay between events in the simulation. For example, `-fixed_delay 10` sets a fixed delay of 10 units.
- `-num_sim`: The number of individual simulations to run. For example, `-num_sim 250` will run 250 simulations.

Feel free to adjust these values to explore different scenarios or to generate alternative results based on your preferences.


## Results & Comments

- **Commented Parts**  
  - Some sections are commented out for *testing purposes*, mainly to sample a small portion of datasets for **faster results**.  
  - Other commented parts are *time-consuming* and *not critical* for our inferences.  
  - **PART 5 – VERSION 2 (Word Embeddings Extension)** is not necessary as it did not yield satisfying results.

- **Databricks & DBFS Usage**  
  - We *write engineered datasets* in **DBFS (Databricks File System)** to avoid re-running code each time.  
  - These datasets **cannot be uploaded here**.  
  - **DBFS write and load cells can be ignored**, as they are inaccessible without our environment.

- **Hugging Face API Requirements**  
  - The code uses *Hugging Face API* to deploy **pretrained models**.  
  - You must *log in with your own valid personal token*.  
  - Any token appearing in the notebook has already been *disabled*.

- **Support & Questions**  
  - If you encounter *any issues running the notebook*, feel free to **contact us**.  
  - We’ll be *glad to help*! 🎯

---

## Model Interpretability
The **Model Interpretability Notebook**:
- Trains **Random Forest** models for **binary classification** per company.
- Measures **Feature Importance** and displays how each feature influences hiring decisions.

Make sure the following prerequisites are met:

- **Databricks Account**: A Databricks account is required to run this project.  
- **Databricks Cluster**: A cluster must be configured and started before running the code.

Then, start the cluster and run the code.

You may need to access patterns_df to run the notebook. We could not load it here.

---

## Scraping
We used **BrightData** for large-scale scraping of:
- **Company Websites**: 'about us' section or similar.
- **Comparably**: references of companies, revealing informations including company's culture.

This file is about the scraping methods we used to scrape the relevant data from the companies' websites and from Comparably. For the scraping mission, we used the BrightData application that allows us to do high-scaling scraping without being blocked. 


### Running the Scraping Code
Before you begin, make sure you have a BrightData account. Copy paste the following lines of codes by replacing "username" and "password" by your own username password.

1. **BrightData Account**  
   Replace `"username"` and `"password"`:
   ```python
   AUTH = 'username:password'
   SBR_WEBDRIVER = f'https://{AUTH}@brd.superproxy.io:9515'


---

## Links

Inception alert 🚨 : You may check our Linkedin post about [On Target](https://www.linkedin.com/posts/tom-bijaoui-2799402ab_machinelearning-bigdata-nlp-activity-7293316200053248000-um9R?utm_source=share&utm_medium=member_ios&rcm=ACoAAEq2IX0Bx9yjkh8KcKEaqRrj5e5HWYojE1c) based on Linkedin Big Data!
