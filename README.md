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

1. [Project Overview](#project-overview)  
2. [About the code](#about-the-code)  
3. [Implemented Classes](#implemented--classes)
4. [Running Instructions](#running-instructions)  
5. [Results & Comments](#results-and-comments)  

</details>

---

## Project Overview
Our project and implementation is based on the article of <blockquote><a href="https://proceedings.neurips.cc/paper_files/paper/2019/hash/88fee0421317424e4469f33a48f50cb0-Abstract.html">Blocking Bandits</a> by Soumya Basu, Rajat Sen, Sujay Sanghavi and Sanjay Shakkottai.</blockquote> This sub-domain represents a new variation in the multi-armed bandit (MAB) problem, where each arm, once played, becomes unavailable or blocked for a certain number of time slots before it can be played again. This models real-world scenarios such as job scheduling, recommendation systems, or resource allocation where actions or resources cannot be reused immediately due to limitations. Blocking bandits introduce significant complexity into decision-making algorithms, particularly in balancing exploration and exploitation. Traditional MAB problems focus on maximizing cumulative rewards while learning the best arm to play. However, with blocking constraints, the problem shifts—decisions made now affect future availability, making it harder to maintain an optimal strategy over time​.


## About the code

* Data Preprocessing
* Feature Engineering
* Pre-trained Models from Hugging Face
* Statistical Tests for Significance Inference
* Machine Learning Model for Model Interpretability
* NLP Keyword Extraction Techniques
* LLM to Generate Instructions

---

## Implemented Classes
- **Model Interpretability**
  - `Model_Interpretability_Notebook`: Analyzes feature importance via Random Forest Model Interpretability Techniques.

- **On Target tool**
  - `On_Target_Notebook`: Core notebook generating guidelines.
  - `example_generated_instructions`: output example showing how the tool provides instructions.

- **Scraping**
  - `Scraping_Comparably`: Code to scrape data from [Comparably](https://www.comparably.com).
  - `Scraping_Company_Websites`: Methods for company website scraping.
  - `Sample_CSV_Scraped_Data`: Example CSV containing raw scraped data.

---

## Running Instructions
Here, we were able to learn the key values for each company, the inherent values of each profile, and understand their leel of importance for each feature. Finally, the system generates the instructions the candidates has to follow to enhance his profil towards a specific company that he targeted.

Example Instruction:
> “Highlight your team collaboration skills more prominently; emphasize your adaptability and willingness to learn.”

---

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
