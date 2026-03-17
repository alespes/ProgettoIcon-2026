import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score, KFold, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder, MinMaxScaler


class PricePredictionTask:
    def __init__(self, data: pd.DataFrame, target_column: str, model: any = RandomForestRegressor(n_estimators=100, random_state=42)):
        """
        Initializes the PricePredictionTask class with data, target column and selected model.

        Parameters:
        - data (pd.DataFrame): The dataset containing features and target
        - target_column (str): The name of the column to predict
        - model (any): The model to be used for prediction
        """
        self.data = data
        self.target_column = target_column
        self.model = model
        self.trained = False
        self.X_train, self.X_test, self.Y_train, self.Y_test, self.Y_pred = None, None, None, None, None

    def preprocess_data(self, test_size: float = 0.2, random_state: int = 42):
        Y = pd.DataFrame(self.data['price'])

        columns_to_drop = ['price',
                           'host_identity_verified',
                           'neighbourhood group',
                         # 'neighbourhood',
                           'lat',
                           'long',
                           'last review',
                           'cancellation_policy',
                           'availability 365',
                           'instant_bookable'
                          ]
        X = self.data.drop(columns=columns_to_drop, axis=1)

        """
        le = LabelEncoder()
        X['neighbourhood'] = le.fit_transform(X['neighbourhood'])
        X['room type'] = le.fit_transform(X['room type'])
        """
        X = pd.get_dummies( X, columns=[# 'host_identity_verified',
                                      # 'neighbourhood group',
                                        'neighbourhood',
                                      #  'cancellation_policy',
                                        'room type'
                                       ], drop_first=True)

        # self.data.to_csv('../data/test.csv' ,index=False)
        # Initialize a MinMaxScaler
        # scaler = MinMaxScaler()
        # Fit and transform the data
        # X.scaled = scaler.fit_transform(X)
        # Y.scaled = scaler.fit_transform(Y)

        # Convert the scaled data back to a DataFrame
        X_scaled = X  # pds.DataFrame(scaler.fit_transform(X), columns=X.columns)
        Y_scaled = Y  # pds.DataFrame(scaler.fit_transform(Y), columns=Y.columns)

        self.X_train, self.X_test, self.Y_train, self.Y_test = train_test_split(X_scaled, Y_scaled['price'],
                                                                                test_size=test_size, random_state=random_state)
        print("Data processing successfully done.")

    def validate(self, num_folds: int = 10, show_results: bool = False, show_individual_results: bool = False   ):
        # Define the parameter grid
        param_grid = {
            'n_estimators': [100, 200, 300],
            'max_depth': [5, 10, 15],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'max_features': ['auto', 'sqrt', 'log2']
        }

        scoring_values = [ 'r2', 'neg_mean_squared_error']

        # Create the GridSearchCV object
        grid_search = GridSearchCV(
            estimator=self.model,
            param_grid=param_grid,
            cv=num_folds,
            scoring=scoring_values,
            refit='r2',
            n_jobs=-1,
            verbose=3
        )

        grid_search.fit(self.X_train, self.Y_train)

        print(" Results from Cross Validation ")
        print("\n The best estimator across ALL searched params:\n", grid_search.best_estimator_)
        print("\n The best score across ALL searched params:\n", grid_search.best_score_)
        print("\n The best parameters across ALL searched params:\n", grid_search.best_params_)
        print("\n The MSE error score for the best estimator:\n", -grid_search.error_score['neg_mean_squared_error'])
        print("\n The R2 error score for the best estimator:\n", grid_search.error_score['r2'])

        self.model = grid_search.best_estimator_

        '''
        print("K-fold set up, starting evaluating the MSE scores")


        mse_scores = cross_val_score(self.model, self.X_train, self.Y_train, cv=kf, scoring='neg_mean_squared_error')

        print("Starting evaluating the R2 scores")
        r2_scores = cross_val_score(self.model, self.X_train, self.Y_train, cv=kf, scoring='r2')
        # Convert MSE scores to positive values
        mse_scores = -mse_scores

        if show_results:
            # Print results
            print(f"K-Fold Cross-Validation Results (k={num_folds}):")
            print(f"Mean Squared Error: {mse_scores.mean():.4f} (+/- {mse_scores.std() * 2:.4f})")
            print(f"R-squared Score: {r2_scores.mean():.4f} (+/- {r2_scores.std() * 2:.4f})")
        
        '''

    def train(self):
        print("Starting the training process")
        # Train final model on entire training set
        self.model.fit(self.X_train, self.Y_train)
        self.trained = True

    def generate_prediction(self, show_results: bool = False):
        print("Starting the testing process")
        if not self.trained:
            print("Model is not trained yet!")
        else :
            # Make predictions
            self.Y_pred = self.model.predict(self.X_test)

            # Inverse transform
            # self.Y_test = pds.DataFrame(scaler.inverse_transform(Y), columns=Y.columns)
            # Y_pred = pds.DataFrame(scaler.inverse_transform(Y), columns=Y.columns)

            if show_results:
                # Calculate final metrics
                final_mse = mean_squared_error(self.Y_test, self.Y_pred)
                final_r2 = r2_score(self.Y_test, self.Y_pred)

                print("\nFinal Model Performance (trained on entire dataset):")
                print(f"Mean Squared Error: {final_mse:.4f}")
                print(f"R-squared Score: {final_r2:.4f}")

                # Calculate standard deviation of the results
                pred_std_dev = np.std(self.Y_pred)
                actual_std_dev = np.std(self.Y_test)

                print(f"Standard Deviation of Predicted Values: {pred_std_dev:.4f}")
                print(f"Standard Deviation of Actual Values: {actual_std_dev:.4f}")

                # Generate the results' plot
                plt.figure(figsize=(10, 6))

                plt.scatter(self.Y_pred, self.Y_test, alpha=0.6)

                # Customize the plot
                plt.title('Actual Price vs Predicted Price', fontsize=14)
                plt.xlabel('Actual Price', fontsize=12)
                plt.ylabel('Predicted Price', fontsize=12)

                # Set the axis limits
                plt.xlim(0, 1300)
                plt.ylim(0, 1300)

                # Add gridlines
                plt.grid(True, linestyle='--', alpha=0.7)

                # Show the plot
                plt.tight_layout()
                plt.show()

    def call(self, preprocessing: bool = True, validation: bool = False, train: bool = True, show_results: bool = False, show_individual_results: bool = False ):
        if preprocessing:
            self.preprocess_data()
        if validation:
            self.validate(num_folds = 10, show_results=show_results, show_individual_results=show_individual_results)
        if train:
            self.train()
        self.generate_prediction(show_results=show_results)

