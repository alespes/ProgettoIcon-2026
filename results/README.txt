Airbnb KBS — Risultati Pipeline
Generato: 2026-03-17 21:10

results/
├── plots/
│   ├── clustering/
│   │   ├── kmeans_pca_clusters.png     ← K-Means: scatter PCA 2D con label cluster
│   │   └── gmm_certainty.png           ← GMM: scatter PCA 2D con livelli di certezza
│   ├── regression/
│   │   ├── regression_scatter_*.png    ← Actual vs Predicted per ogni modello
│   └── classification/
│       ├── roc_curve_*.png             ← Curva ROC con AUC
│       ├── feature_importance_*.png    ← Top-15 feature (verde = KB-derived)
│       └── cv_barplot_*.png            ← Mean ± Std per Accuracy/Precision/Recall/F1/AUC
└── metrics/
    ├── cv_results_*.csv                ← Risultati per fold + Mean + Std
    ├── clustering_analysis_kmeans.csv  ← Centroidi K-Means
    ├── clustering_analysis_gmm.csv     ← Centroidi GMM
    └── summary_regression.csv         ← MSE, R² per ogni modello di regressione
