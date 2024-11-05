import pandas as pd

def load_data():
    products = pd.read_csv('data/products.csv')
    users = pd.read_csv('data/users.csv')
    interactions = pd.read_csv('data/interactions.csv')
    return products, users, interactions

def clean_data(interactions):
    # Handle missing values and remove duplicates
    interactions = interactions.dropna().drop_duplicates()
    return interactions

if __name__ == "__main__":
    products, users, interactions = load_data()
    interactions = clean_data(interactions)
    print(interactions.head())
