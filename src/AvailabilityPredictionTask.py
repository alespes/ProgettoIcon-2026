import numpy as np
import pandas as pd
import xgboost as xgb
from matplotlib import pyplot as plt
from scipy.stats import chi2_contingency
from xgboost import XGBClassifier, plot_importance
from xgboost import cv
from sklearn.model_selection import train_test_split
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, \
    mean_squared_error, r2_score, roc_auc_score, roc_curve


class AvailabilityPredictionTask:
    def __init__(self, data: pd.DataFrame, target_column: str,
                 model: any = XGBClassifier(  alpha=0, base_score=0.5, booster='gbtree', colsample_bylevel=1,
                                              colsample_bynode=1, colsample_bytree=0.6, eval_metric='logloss',
                                              gamma=0.2, learning_rate=0.1, max_delta_step=0, max_depth=7,
                                              min_child_weight=5, missing=np.nan, n_estimators=300, n_jobs=1,
                                              nthread=None, objective='binary:logistic', random_state=42,
                                              reg_alpha=0.5, reg_lambda=1, scale_pos_weight=1, seed=None,
                                              silent=None, subsample=0.8, verbosity=1)):
        """
        Initializes the PricePredictionTask class with data, target column and selected model.

        Parameters:
        - data (pd.DataFrame): The dataset containing features and target
        - target_column (str): The name of the column to predict
        - model (any): The model to be used for prediction
        """

        self.data = data
        self.data_dmatrix = None
        self.target_column = target_column
        self.model = model
        self.model_name = model.__class__.__name__
        self.trained = False
        self.X_train, self.X_test, self.Y_train, self.Y_test, self.Y_pred = None, None, None, None, None

    def preprocess_data(self, test_size: float = 0.2, random_state: int = 42):

        self.data[self.target_column] = self.data[self.target_column].astype(int)

        Y = pd.DataFrame(self.data[self.target_column])

        columns_to_drop = ['instant_bookable',
                         # 'price',
                           'host id',
                           'id',
                         # 'host_identity_verified',
                         # 'neighbourhood group',
                           'neighbourhood',
                           'lat',
                           'long',
                         # 'last review',
                         # 'cancellation_policy',
                         # 'availability 365',
                           ]
        X = self.data.drop(columns=columns_to_drop, axis=1)

        X = pd.get_dummies(X, columns=['host_identity_verified',
                                       'neighbourhood group',
                                     # 'neighbourhood',
                                       'cancellation_policy',
                                       'room type'
                                      ], drop_first=True)

        reference_date = pd.to_datetime('2012-01-01')
        X['last review'] = pd.to_datetime(X['last review'])
        X['last review'] = (X['last review'] - reference_date).dt.days

        # define data_dmatrix
        self.data_dmatrix = xgb.DMatrix(data=X, label=Y)

        self.X_train, self.X_test, self.Y_train, self.Y_test = train_test_split(X, Y[self.target_column], test_size=test_size, random_state=random_state)

        # print("X_train shape:", self.X_train.shape)
        # print("y_train shape:", self.Y_train.shape)
        # print("Unique classes in y_train:", self.Y_train.unique())
        # print(self.Y_train.value_counts(normalize=True))

        '''
        for feature in self.data.columns:
            # Create a contingency table for categorical features
            contingency_table = pd.crosstab(self.data[feature], self.data['instant_bookable'])
            chi2, _, _, _ = chi2_contingency(contingency_table)
            n = contingency_table.sum().sum()
            r, k = contingency_table.shape
            cramers_v = np.sqrt(chi2 / (n * (min(r, k) - 1)))
            print(f"Cramér's V for {feature} :", cramers_v)
        '''
        print("Data processing successfully done.")

    def train(self):
        print("Starting the training process")
        # Train final model on entire training set
        self.model.fit(self.X_train, self.Y_train)
        self.trained = True

    def validate(self):
        params = {"objective": "binary:logistic", 'colsample_bytree': 0.3, 'learning_rate': 0.1,
                  'max_depth': 5, 'alpha': 10}

        xgb_cv = cv(dtrain=self.data_dmatrix, params=params, nfold=3,
                    num_boost_round=50, early_stopping_rounds=10, metrics="auc", as_pandas=True, seed=123)

    def tune(self):
        param_grid = {
            'n_estimators': [100, 200, 300, 400],
            'learning_rate': [0.01, 0.05, 0.1, 0.2],
            'max_depth': [3, 5, 7, 10],
            'min_child_weight': [1, 3, 5, 7],
            'subsample': [0.6, 0.8, 1.0],
            'colsample_bytree': [0.6, 0.8, 1.0],
            'gamma': [0, 0.1, 0.2, 0.5]
        }

        # Instantiate the XGBClassifier
        model = XGBClassifier(objective='binary:logistic', random_state=42)

        # Set up the RandomizedSearchCV
        random_search = RandomizedSearchCV(
            estimator=model,
            param_distributions=param_grid,
            n_iter=50,  # Number of parameter settings sampled (adjust for time/accuracy balance)
            scoring='f1',  # Use 'f1' or any other metric based on your goals
            cv=10,  # 5-fold cross-validation
            verbose=1,
            random_state=42,
            n_jobs=-1  # Use all available CPU cores
        )

        # Fit RandomizedSearchCV to the data
        random_search.fit(self.X_train, self.Y_train)

        # Best parameters and score
        print("Best parameters found: ", random_search.best_params_)
        print("Best F1 score: ", random_search.best_score_)

        self.model = random_search.best_estimator_

    def generate_prediction(self, show_results: bool = False):
        print("Starting the prediction process")
        # print(self.X_test.isnull().sum())
        # self.X_test.dropna(inplace=True)
        if not self.trained:
            print("Model is not trained yet!")
        else :
            # Make predictions
            self.Y_pred = self.model.predict(self.X_test)

            if show_results:
                # Calculate final metrics

                print(self.model_name + ' model scores:' )
                print(classification_report(self.Y_test, self.Y_pred))

                final_mse = mean_squared_error(self.Y_test, self.Y_pred)
                final_r2 = r2_score(self.Y_test, self.Y_pred)

                print("\n")
                print(f"Mean Squared Error: {final_mse:.4f}")
                print(f"R-squared Score: {final_r2:.4f}")

                # Calculate standard deviation of the results
                pred_std_dev = np.std(self.Y_pred)
                actual_std_dev = np.std(self.Y_test)

                print(f"Standard Deviation of Predicted Values: {pred_std_dev:.4f}")
                print(f"Standard Deviation of Actual Values: {actual_std_dev:.4f}")

                # AUC-ROC Score
                auc_score = roc_auc_score(self.Y_test, self.Y_pred)
                print(f"AUC-ROC Score: {auc_score:.4f}")

                # Plot ROC Curve
                fpr, tpr, thresholds = roc_curve(self.Y_test, self.Y_pred)
                plt.figure(figsize=(10, 6))
                plt.plot(fpr, tpr, color='blue', label=f'ROC Curve (AUC = {auc_score:.4f})')
                plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
                plt.xlabel('False Positive Rate')
                plt.ylabel('True Positive Rate')
                plt.title('ROC Curve')
                plt.legend()
                plt.show()

                if isinstance( self.model, XGBClassifier ):
                    # Generate the results' plot
                    # Set the figure size and plot the importance
                    fig, ax = plt.subplots(1,1,figsize=(20, 8))  # Specify the size of the figure here
                    plot_importance(self.model, ax=ax, max_num_features=10)  # Plot top 10 features
                    plt.show()

    def call(self, preprocessing: bool = True, validation: bool = False, train: bool = True, show_results: bool = True, show_individual_results: bool = False ):
        if preprocessing:
            self.preprocess_data()
        if validation:
            self.tune()
            # self.validate(num_folds = 10, random_state = 42, show_results=show_results, show_individual_results=show_individual_results)
        if train:
            self.train()
        self.generate_prediction(show_results=show_results)