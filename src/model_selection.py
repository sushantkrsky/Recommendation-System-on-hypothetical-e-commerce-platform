import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

class CollaborativeFilteringModel:
    def __init__(self, user_item_matrix):
        self.user_item_matrix = user_item_matrix.fillna(0)  # Fill NaN with 0 for similarity calculation
        self.user_similarity = cosine_similarity(self.user_item_matrix)

    def predict(self, user_vector):
        # Compute predicted ratings based on the user vector and similarities
        similarity_scores = self.user_similarity[user_vector]
        predicted_ratings = self.user_item_matrix.dot(similarity_scores) / np.array([np.abs(similarity_scores).sum()])
        return predicted_ratings

    def recommend(self, user_id, top_n=5):
        user_index = user_id - 1  # Adjust index for user_id
        predicted_ratings = self.predict(user_index)
        recommended_products = np.argsort(predicted_ratings)[::-1][:top_n]
        return recommended_products + 1  # Adjust to product IDs

def collaborative_filtering(user_item_matrix, user_id, top_n=5):
    model = CollaborativeFilteringModel(user_item_matrix)
    return model.recommend(user_id, top_n)

if __name__ == "__main__":
    from feature_engineering import generate_features
    from data_preprocessing import load_data, clean_data

    products, users, interactions = load_data()
    interactions = clean_data(interactions)
    user_item_matrix = interactions.pivot_table(index='user_id', columns='product_id', values='rating')

    recommendations = collaborative_filtering(user_item_matrix, user_id=1)
    print(f"Recommended products for user 1: {recommendations}")
