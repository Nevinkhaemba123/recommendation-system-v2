# Movie Recommendation Systems

## Acknowledgments
This project was developed by Group 2:

Nevin Khaemba
Reuben Mutisya
Esther Kabura
Billy Ogonji
Leah Njoroge
Arlene Stanley

## 1. Project Overview & Objective

**Objective:** Build a personalized recommendation engine that delivers the top 5 movie recommendations for a user based on their historical ratings and interaction data.

**Dataset:** MovieLens Small Dataset (`ml-latest-small`) from the GroupLens Research Lab at the University of Minnesota. The dataset contains:

- 100,836 ratings
- 9,742 movies
- 610 unique users

**Core Requirements Addressed:**
- Item-based collaborative filtering
- Cold-start problem handling
- Model evaluation using Root Mean Squared Error (RMSE)

## 2. Dataset Ingestion & Exploratory Data Analysis (EDA)

### Data Ingestion

Extracted ratings and movie metadata using Python's `zipfile` and `os` libraries, then loaded the data into Pandas DataFrames for analysis.

### Data Quality Check

- Checked for missing values in both ratings and movie datasets.
- Confirmed that there were zero missing values across the two tables.

### Sparsity & Long-Tail Behaviour

- Constructed a user-item matrix with dimensions of `(610, 9742)`.
- Identified significant sparsity, as most users rated only a small fraction of the available movies.
- Observed a classic long-tail distribution, where a small number of popular movies received thousands of ratings, while many niche movies received only a single rating.

## 3. Item-Based Collaborative Filtering (KNN)

### User-Item Interaction Matrix

Constructed a pivot table mapping user IDs to movie IDs, with ratings representing user preferences.

- Filled unrated entries with zeros.
- Converted the matrix into a sparse Compressed Sparse Row (CSR) matrix using `scipy.sparse.csr_matrix` to improve memory efficiency.

### Model Configuration

Implemented a K-Nearest Neighbors model using `sklearn.neighbors.NearestNeighbors`.

The model was configured with:

- **Similarity metric:** Cosine similarity
- **Algorithm:** Brute-force search
- **Approach:** Item-based collaborative filtering

The user-item matrix was transposed so that movies could be compared based on their rating patterns across users.

### Recommendation Pipeline

For a target user, the system:

1. Identifies movies the user has rated highly (rating ≥ 4.0).
2. Queries the KNN model to identify similar movies.
3. Calculates cumulative similarity scores for candidate movies.
4. Filters out movies the user has already watched.
5. Returns the top 5 highest-ranked recommendations.

## 4. Cold-Start Problem & Hybrid Strategy

### The Challenge

The cold-start problem occurs when a new user joins the system or an existing user has very limited rating history. In such cases, collaborative filtering may struggle to generate meaningful recommendations.

### Hybrid Fallback Mechanism

To address this challenge, a popularity-based fallback mechanism was implemented.

- If a user is not found or has fewer than 3 ratings, the system switches to the fallback recommendation approach.
- Movies are filtered using a minimum review threshold (e.g., ≥ 50 ratings) to improve reliability.
- Movies are ranked using a Bayesian or weighted average rating score.
- Genre information can also be incorporated to improve recommendation relevance.

This approach ensures that users with limited interaction history can still receive meaningful movie recommendations.

## 5. Model-Based Collaborative Filtering (SVD) & Evaluation

### Baseline Evaluation

A global mean baseline model was used to establish a benchmark for prediction accuracy.

- **Baseline RMSE:** 1.0447

### Advanced SVD Model

Implemented matrix factorization using the `scikit-surprise` library and its Singular Value Decomposition (SVD) algorithm.

**Model Configuration:**

| Parameter | Value |
|---|---:|
| Number of factors (`n_factors`) | 100 |
| Number of epochs (`n_epochs`) | 20 |
| Learning rate (`lr_all`) | 0.005 |
| Regularization (`reg_all`) | 0.02 |

### Final Model Performance

- **SVD Test RMSE:** 0.8830
- **Baseline RMSE:** 1.0447
- **RMSE Reduction:** Approximately 15.5%

The SVD model achieved a lower RMSE than the global mean baseline, indicating improved rating prediction accuracy.

This improvement demonstrates the value of matrix factorization in capturing latent relationships between users and movies.

## 6. Conclusion

The project successfully developed an end-to-end movie recommendation system using collaborative filtering and matrix factorization techniques.

The item-based KNN approach identifies movies with similar rating patterns, while the popularity-based fallback mechanism addresses the cold-start problem. The SVD model further improves rating prediction performance, achieving a test RMSE of 0.8830 compared to the baseline RMSE of 1.0447.

Overall, the project demonstrates practical applications of Python, data preprocessing, exploratory data analysis, sparse matrices, machine learning, and recommendation systems in building personalized recommendation engines.
