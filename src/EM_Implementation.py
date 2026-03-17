import os

import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from scipy.stats import zscore
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler, OneHotEncoder


class EM_Implementation:
    def __init__(self, data: pd.DataFrame):
        self.data = data
        self.featured_data = None

    def call(self):
        # Selecting relevant features for clustering
        self.featured_data = self.data[
            ['neighbourhood group', 'host_identity_verified', 'room type',
             # 'price',
             'minimum nights',
             'instant_bookable',
             'cancellation_policy', 'availability 365', 'reviews per month']]

        # Checking for missing values and handling them (if necessary)
        self.featured_data = self.featured_data.dropna()  # Simple approach; could use imputation if needed

        # Define categorical and numerical features
        categorical_features = ['room type', 'instant_bookable', 'cancellation_policy', 'neighbourhood group',
                                'host_identity_verified']
        numerical_features = [# 'price',
                              'minimum nights', 'availability 365', 'reviews per month']

        scaler = StandardScaler()
        scaled_numerics = scaler.fit_transform(self.featured_data.select_dtypes(include=['float64', 'int']))

        encoder = OneHotEncoder(sparse_output=False)
        encoded_cats = encoder.fit_transform(self.featured_data.select_dtypes(include=['object']))

        # Combine scaled numerical and encoded categorical features
        X_prepared = np.hstack((scaled_numerics, encoded_cats))

        # Define the GMM model with a chosen number of components
        num_components = 3  # You can tune this based on your needs
        gmm = GaussianMixture(n_components=num_components, covariance_type='full', random_state=42)

        # Fit the GMM model on the preprocessed data
        gmm.fit(X_prepared)

        # Predict the cluster probabilities for each data point
        cluster_probs = gmm.predict_proba(X_prepared)
        cluster_labels = gmm.predict(X_prepared)  # Hard assignments based on maximum probability

        # Calculate BIC and AIC to evaluate the model
        bic = gmm.bic(X_prepared)
        aic = gmm.aic(X_prepared)
        print("BIC:", bic)
        print("AIC:", aic)

        self.featured_data['cluster'] = cluster_labels

        # Cluster analysis (mean for numerical and mode for categorical features)
        numerical_features = self.featured_data.select_dtypes(include=['number']).columns
        categorical_features = self.featured_data.select_dtypes(include=['object']).columns

        numeric_means = self.featured_data.groupby('cluster')[numerical_features].mean()
        categorical_modes = self.featured_data.groupby('cluster')[categorical_features].agg(
            lambda x: x.mode().iloc[0] if not x.mode().empty else np.nan
        )

        # Combine analysis results
        cluster_analysis = pd.concat([numeric_means, categorical_modes], axis=1)
        print("Cluster analysis with GMM:")
        print(cluster_analysis)
        cluster_analysis.to_csv(os.path.join('data', 'Soft_clustering_analysis.csv'), index=False)

        # Apply PCA for dimensionality reduction (2 or 3 components for visualization)
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X_prepared)
        # Calculate z-scores across PCA components
        z_scores = np.abs(zscore(X_pca))
        # Define a z-score threshold (e.g., 3 standard deviations) and filter
        z_threshold = 3
        non_outliers = (z_scores < z_threshold).all(axis=1)

        # Calculate certainty for each point as the max probability of belonging to any cluster
        certainty = cluster_probs.max(axis=1)
        # Filter out outliers in both X_prepared and cluster assignments
        X_prepared_filtered = X_prepared[non_outliers]
        cluster_labels_filtered = cluster_labels[non_outliers]
        certainty_filtered = certainty[non_outliers]

        # Scatter plot with certainty as a color scale (after outlier removal)
        plt.figure(figsize=(10, 7))
        sns.scatterplot(x=X_pca[non_outliers][:, 0], y=X_pca[non_outliers][:, 1],
                        hue=certainty_filtered, palette="coolwarm", s=50) # legend="full"
        plt.title("GMM Clustering with Certainty Levels")
        plt.xlabel("PCA Component 1")
        plt.ylabel("PCA Component 2")
        plt.legend(title='Cluster')
        plt.show()
