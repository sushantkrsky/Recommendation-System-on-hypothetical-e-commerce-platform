import pandas as pd
import numpy as np

def load_csv(file_path):
    """Load a CSV file into a DataFrame."""
    try:
        return pd.read_csv(file_path)
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None

def calculate_average_rating(interactions):
    """Calculate the average rating for each product."""
    return interactions.groupby('product_id')['rating'].mean().reset_index()

def recommend_top_n_products(recommendations, n=3):
    """Return the top n recommended products."""
    return recommendations[:n]

def get_product_names(products_df, product_ids):
    """Get product names from product IDs."""
    return products_df[products_df['product_id'].isin(product_ids)][['product_id', 'product_name']]
