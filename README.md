Predicting Emotion in Music (Valence and Energy)

A statistical and machine learning project that predicts two emotional dimensions of music valence and energy from Spotify audio features, 
using the Russell (1980) Circumplex Model of Affect as a theoretical framework.

```
PredictionEmotioninMusic/
├── data/
│   └── dataset.csv          # Spotify track-level audio features (source data)
├── docs/
│   ├── Final Report.pdf   # Full written report
│   └── Presentation.pptx    # Project presentation slides
├── src/
│   └── code.R               # Complete analysis script 
└── README.md
```

Pipeline (`src/code.R`)

The script runs end to end in five stages:

1. Data Loading & Preparation
- Reads the CSV, cleans column names, and encodes categorical variables.
- Subsets the data into the three genre groups (G1, G2, G3).

2. Descriptive Statistics
- Summary statistics (mean, SD, median, skewness, kurtosis) for all audio features.
- Per-group and per-genre breakdowns for valence and energy.
- Missing value and outlier diagnostics.
- Pairwise Pearson correlations between features and each target, per group.
- Kruskal–Wallis tests for group differences in valence and energy.

3. Exploratory Data Analysis
- Histograms of valence and energy by group.
- Circumplex scatter plots (valence vs energy) with 2-D density contours.
- Box plots and violin plots comparing groups.
- Correlation heatmaps (one per group).
- Normalised feature mean bar charts across groups.
- Within group genre-level violin plots.

4. Modelling
Each target (valence, energy) is modelled independently within each group using a 75/25 stratified train/test split with 5 fold Cross Validation. Five model families are evaluated:

- Linear Regression 
- ElasticNet 
- KNearest Neighbours
- Random Forest
- XGBoost
- Stacked Ensemble

5. Evaluation
Models are compared on RMSE, MAE, and R² across all group and target combinations.
