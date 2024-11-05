from sklearn.model_selection import GridSearchCV

def tune_hyperparameters(model, param_grid, X_train, y_train):
    grid_search = GridSearchCV(model, param_grid, scoring='neg_mean_squared_error', cv=5)
    grid_search.fit(X_train, y_train)
    return grid_search.best_params_

if __name__ == "__main__":
    # Example usage (for a simple model like KNN)
    from sklearn.neighbors import KNeighborsRegressor
    model = KNeighborsRegressor()
    param_grid = {'n_neighbors': [3, 5, 7]}
    # Assume X_train and y_train are already defined
    best_params = tune_hyperparameters(model, param_grid, X_train, y_train)
    print(f"Best Hyperparameters: {best_params}")
