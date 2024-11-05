import pandas as pd

def generate_features(products, users, interactions):
    # Example: Merge datasets to create a feature-rich data frame
    merged_data = pd.merge(interactions, products, on='product_id', how='left')
    merged_data = pd.merge(merged_data, users, on='user_id', how='left')
    # Add additional engineered features here if necessary
    return merged_data

if __name__ == "__main__":
    from data_preprocessing import load_data, clean_data
    products, users, interactions = load_data()
    interactions = clean_data(interactions)
    feature_data = generate_features(products, users, interactions)
    print(feature_data.head())
