import sys 
import os
import pandas as pd  # Import pandas for data manipulation

# Add the root directory to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.helper_functions import load_csv, calculate_average_rating, recommend_top_n_products

from src.data_preprocessing import load_data, clean_data
from src.feature_engineering import generate_features
from src.model_selection import collaborative_filtering
from src.evaluation import evaluate_model

def main():
    products, users, interactions = load_data()
    interactions = clean_data(interactions)
    feature_data = generate_features(products, users, interactions)
    
    user_item_matrix = feature_data.pivot_table(index='user_id', columns='product_id', values='rating')

    # Get user input for user ID or recommend for all users
    user_input = input("Enter user ID for recommendations (or type 'all' for all users): ")
    
    products_df = pd.read_csv("data/products.csv")  # Load product data
    if user_input.lower() == 'all':
        # Iterate over all user IDs and get recommendations
        for user_id in user_item_matrix.index:
            recommendations = collaborative_filtering(user_item_matrix, user_id=user_id)
            print(f"\nTop 3 recommended products for User ID {user_id}:")
            for product_id in recommendations[:3]:  # Limit to top 3 products
                product_name = products_df.loc[products_df['product_id'] == product_id, 'product_name'].values[0]
                print(f"Product ID: {product_id}, Product Name: {product_name}")
    else:
        try:
            user_id = int(user_input)  # Convert input to integer
            if user_id in user_item_matrix.index:
                recommendations = collaborative_filtering(user_item_matrix, user_id=user_id)
                print(f"\nTop 3 recommended products for User ID {user_id}:")
                for product_id in recommendations[:3]:  # Limit to top 3 products
                    product_name = products_df.loc[products_df['product_id'] == product_id, 'product_name'].values[0]
                    print(f"Product ID: {product_id}, Product Name: {product_name}")
            else:
                print("User ID not found.")
        except ValueError:
            print("Invalid input. Please enter a valid user ID or 'all'.")

    # Example evaluation (you would replace with actual data)
    predictions = [0.8, 0.6, 0.4]
    ground_truth = [1, 0, 1]
    metrics = evaluate_model(predictions, ground_truth)
    print("Evaluation Metrics:", metrics)

if __name__ == "__main__":
    main()
