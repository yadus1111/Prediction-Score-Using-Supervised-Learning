# Task 1
# Student Score Prediction

This project is a simple data science project that uses supervised learning to predict students' scores based on the number of study hours. It includes data loading, data visualization, model building, and evaluation.

## Table of Contents

- [Introduction](#introduction)
- [Dataset](#dataset)
- [Results](#results)

## Introduction

The goal of this project is to create a predictive model that can estimate a student's score based on the number of hours they study. We used a linear regression model to establish the relationship between study hours and scores.

## Dataset

 find the dataset [here](https://raw.githubusercontent.com/AdiPersonalWorks/Random/master/student_scores%20-%20student_scores.csv).
## Results

The project includes data visualization, model training, prediction, and evaluation. Here are some key results:

- The linear regression model predicts student scores based on study hours.
- The Mean Absolute Error (MAE) and Mean Squared Error (MSE) are used to evaluate the model's performance.
- A scatter plot visualizes the actual vs. predicted scores.

  
### We made predictions using the trained model, including predicting the score for a specific case where a student studies for 9.25 hours, which resulted in a predicted score of approximately 92.39.

# Task 2
# Prediction Using Unsupervised ML
### Author
- Yadu Sharma

### Objective
The primary goal of this project is to apply unsupervised learning, specifically K-means clustering, to the Iris dataset. The objective is to determine the optimal number of clusters in the dataset and then visually represent these clusters.

### Prerequisites
Before running the code in this repository, ensure you have the following libraries installed:
- numpy
- pandas
- matplotlib
- scikit-learn
- seaborn

### Dataset
The dataset used for this project is 'You can access the dataset [iris.csv](https://bit.ly/3kXTdox).'. It contains information about iris flowers, including sepal and petal measurements.

### Code Overview
- Importing Libraries: The necessary libraries are imported for data manipulation, visualization, and K-means clustering.
- Loading the Dataset: The Iris dataset is loaded from the 'iris.csv' file.
- Data Exploration: Basic exploration of the dataset, including checking for missing values, displaying statistics, and analyzing class distribution.
- Determining the Optimal Number of Clusters: The Elbow Method is used to find the optimal number of clusters (K) for K-means clustering.
- Performing K-Means Clustering: K-means clustering is performed using the optimal K value determined in the previous step.
- Analysis & Visualization: Various analyses and visualizations are conducted to understand the clusters, including cluster size analysis, cluster statistics, scatter plots, 3D scatter plot, silhouette plot, and parallel coordinate plot.
- Conclusion: A summary of the project's findings and the power of unsupervised learning for pattern discovery in data.
### Conclusion
In this project, we applied K-means clustering to the Iris dataset to determine the optimal number of clusters (K=3) and visually represent these clusters. We conducted data exploration, found the optimal K using the Elbow Method, performed K-means clustering, and visualized the results through scatter plots, Silhouette Plot, Parallel Coordinate Plot of Clusters. This project showcases the power of unsupervised learning for pattern discovery in data.


