# Personalized Recommendation System

## Project Overview
This project implements a personalized recommendation system designed to suggest relevant products to users in a hypothetical e-commerce platform. By utilizing collaborative filtering techniques, the system analyzes user interactions to provide tailored product recommendations, enhancing user engagement and satisfaction.

## Table of Contents
- [Features](#features)
- [Technologies Used](#technologies-used)
- [Installation](#installation)
- [Usage](#usage)
- [Data](#data)
- [Methodology](#methodology)
- [Evaluation Metrics](#evaluation-metrics)
- [License](#license)
- [Acknowledgments](#acknowledgments)

## Features
- Collaborative filtering approach for product recommendations
- Evaluation of model performance using metrics such as MAP, NDCG, and MRR
- Modular structure for easy updates and maintenance
- User-friendly input for dynamic recommendations

## Technologies Used
- Python 3.x
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Jupyter Notebook (for exploratory analysis)

## Installation
To run this project locally, follow these steps:

1. Clone the repository:
   ```bash
   git clone https://github.com/sushantkrsky/personalized-recommendation-system.git
2. Navigate to the project directory:
   ```bash
   cd personalized-recommendation-system
3. Install the required packages:
   ```bash   
   pip install -r requirements.txt

Usage
1. Load your dataset into the data folder.
Run the main script:
2. ```bash
   python src/main.py
3. Follow the prompts to input user ID for which you want to see product recommendations

Data
The project uses a sample dataset consisting of user interactions with products. Ensure your dataset is structured correctly as per the project requirements.

Dataset Structure
products.csv: Contains product information including product ID and name.
users.csv: Contains user details including user ID.
interactions.csv: Contains user interaction data (ratings or purchase history).
Methodology
Data Preprocessing: Cleaned and prepared the dataset for analysis.
Feature Engineering: Created relevant features from interaction and product data.
Model Development: Implemented a collaborative filtering recommendation approach.
Evaluation: Used MAP, NDCG, and MRR to assess model accuracy.
Evaluation Metrics
Mean Average Precision (MAP)
Normalized Discounted Cumulative Gain (NDCG)
Mean Reciprocal Rank (MRR)
License
This project is licensed under the MIT License. See the LICENSE file for details.

Acknowledgments
Special thanks to Kaushik Sheet for project management and guidance.
Thanks to the contributors and resources that helped shape this project.
