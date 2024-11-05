import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

def collaborative_filtering_recommendation(user_item_matrix, user_id, top_n=5):
    # Compute the cosine similarity between users
    user_similarity = cosine_similarity(user_item_matrix.fillna(0))
    
    # Get the similarity scores for the target user
    user_index = user_id - 1  # Adjust index (assuming user_id starts from 1)
    similarity_scores = user_similarity[user_index]
    
    # Predict ratings for the target user
    user_ratings = user_item_matrix.iloc[user_index]
    weighted_ratings = user_item_matrix.fillna(0).T.dot(similarity_scores)
    norm_factor = np.array([np.abs(similarity_scores).sum()])
    predicted_ratings = weighted_ratings / norm_factor
    
    # Exclude products already rated by the user
    already_rated = user_ratings[user_ratings.notna()].index
    predicted_ratings[already_rated] = -np.inf  # Set to negative infinity to exclude
    
    # Recommend products with the highest predicted ratings
    recommended_products = np.argsort(predicted_ratings)[-top_n:][::-1]  # Top N products
    return recommended_products + 1  # Adjust back to product IDs

if __name__ == "__main__":
    from data_preprocessing import load_data, preprocess_data
    
    products, users, interactions = load_data()
    user_item_matrix = preprocess_data(interactions)
    
    # Get refined recommendations for a sample user (user_id=1)
    recommendations = collaborative_filtering_recommendation(user_item_matrix, user_id=1)
    print(f"Recommended product IDs for user 1: {recommendations}")
