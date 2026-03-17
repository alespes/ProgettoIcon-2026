import os

import pandas as pd
import numpy as np
from scipy.stats import zscore
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns

class GuestPreferenceSegmentationTask:
    def __init__(self, data: pd.DataFrame):
        self.X = None
        self.featured_data = None
        self.data = data

    def preproccessing(self):
        # Selecting relevant features for clustering
        self.featured_data = self.data[['neighbourhood group', 'host_identity_verified', 'room type',
                                        'price',
                                        'minimum nights', 'instant_bookable',
                         'cancellation_policy', 'availability 365', 'reviews per month']]

        # Checking for missing values and handling them (if necessary)
        self.featured_data = self.featured_data.dropna()  # Simple approach; could use imputation if needed

        # Define categorical and numerical features
        categorical_features = ['room type', 'instant_bookable', 'cancellation_policy', 'neighbourhood group', 'host_identity_verified']
        numerical_features = [ 'price',
                              'minimum nights', 'availability 365', 'reviews per month']


        # Scale numerical variables first
        scaler = StandardScaler()
        scaled_numerics = []
        for n_feature in numerical_features:
            feature_values = self.featured_data[n_feature].values.reshape(-1, 1)
            scaled_column = scaler.fit_transform(feature_values)
            scaled_numerics.append(scaled_column)

        # Combine numerical features and convert to numpy array explicitly
        scaled_numerics = np.asarray(np.hstack(scaled_numerics))
        # print("Scaled numerics shape:", scaled_numerics.shape)

        # Process categorical features with sparse_output=False
        encoder = OneHotEncoder(sparse_output=False)  # Explicitly set sparse_output=False
        # Convert categorical data to the right format
        categorical_data = self.featured_data[categorical_features].values
        # Fit and transform categorical data
        encoded_cats = encoder.fit_transform(categorical_data)

        # Combine encoded and scaled features
        self.X = np.column_stack((scaled_numerics, encoded_cats))

    def apply_Kmeans(self):
        # Define the number of clusters
        kmeans = KMeans(n_clusters=3, random_state=0)
        clusters = kmeans.fit_predict(self.X)

        # Add the cluster labels to the original dataset
        self.featured_data['cluster'] = clusters

        # Apply PCA for dimensionality reduction to a fixed number of components for visualization
        num_components = 2 #2D visualization
        pca = PCA(n_components=num_components)
        X_pca = pca.fit_transform(self.X)

        # Compute the z-scores for each component in PCA
        z_scores = np.abs(zscore(X_pca))
        # Define a threshold for outliers, e.g., 3 standard deviations
        z_threshold = 3
        # Filter out rows where any component exceeds the threshold
        non_outliers = (z_scores < z_threshold).all(axis=1)
        # Apply the outlier mask to both X_pca and self.features to ensure length consistency
        X_pca = X_pca[non_outliers]
        self.featured_data = self.featured_data[non_outliers].reset_index(drop=True)

        # Create a scatter plot of the clusters
        if num_components == 2:
            # Set up the 2D plot
            plt.figure(figsize=(10, 7))
            sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=self.featured_data['cluster'], palette="viridis", s=60)
            plt.title("Guest Preference Clusters for Airbnb Listings in NYC")
            plt.xlabel("PCA Component 1")
            plt.ylabel("PCA Component 2")
            plt.legend(title='Cluster')

            plt.show()
        elif num_components == 3:
            # Set up the 3D plot
            fig = plt.figure(figsize=(10, 7))
            ax = fig.add_subplot(111, projection='3d')
            ax.scatter(X_pca[:, 0], X_pca[:, 1], X_pca[:, 2], c='b', marker='o', s=50, alpha=0.6)
            ax.set_xlabel('Principal Component 1')
            ax.set_ylabel('Principal Component 2')
            ax.set_zlabel('Principal Component 3')
            plt.title("3D PCA Plot")

            plt.show()

        # Separate numerical and categorical columns
        numerical_features = self.featured_data.select_dtypes(include=['number']).columns
        categorical_features = self.featured_data.select_dtypes(include=['object']).columns

        # Calculate mean for numerical features
        numeric_means = self.featured_data.groupby('cluster')[numerical_features].mean()
        # or the mode alternatively
        '''
        numeric_means = self.features.groupby('cluster')[numerical_features].agg(
            lambda x: x.mode().iloc[0] if not x.mode().empty else np.nan
        ) '''

        # Calculate mode for categorical features
        categorical_modes = self.featured_data.groupby('cluster')[categorical_features].agg(
            lambda x: x.mode().iloc[0] if not x.mode().empty else np.nan
        )

        # Combine numeric means and categorical modes
        cluster_analysis = pd.concat([numeric_means, categorical_modes], axis=1)

        print("Cluster analysis with mean of numeric features and mode of categorical features:")
        print(cluster_analysis)
        cluster_analysis.to_csv(os.path.join('data', 'Hard_clustering_analysis.csv'), index=False)

    def call(self):
        self.preproccessing()
        self.apply_Kmeans()