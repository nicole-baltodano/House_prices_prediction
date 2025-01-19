### main.py
from data_processing import load_data, split_data
from feature_engineering import transform_cyclical_features, prepare_pipeline
from model_training import train_model, evaluate_model
from visualization import plot_results


def main():
    # Load and split data
    data, X, y = load_data("data/train.csv")
    X_train, X_test, y_train, y_test = split_data(X, y)

    # Transform features
    X_train = transform_cyclical_features(X_train)
    X_test = transform_cyclical_features(X_test)

    # Prepare pipeline
    preproc_pipeline = prepare_pipeline(X_train)

    # Train and tune model
    best_model, y_train_log = train_model(preproc_pipeline, X_train, y_train)

    # Evaluate model
    rmsle, predictions = evaluate_model(best_model, X_test, y_test, y_train_log)

    # Visualize results
    plot_results(y_test, predictions, rmsle)


if __name__ == "__main__":
    main()
