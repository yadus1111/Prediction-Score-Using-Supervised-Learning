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


# Stock Market Prediction using Numerical and Textual Analysis

## Objective:
To create a hybrid model for stock price/performance prediction using numerical analysis of historical stock prices and sentimental analysis of news headlines.

## Author: 
Yadu Sharma
Data Science & Business Analytics Intern

### Project Overview:

This project combines two different data sources, numerical stock price data and textual news headlines data, to create a hybrid model for predicting stock price performance. The objective is to leverage both numerical and textual data to improve the accuracy of stock price predictions.

### Data Sources:

1. **Numerical Data (Stock Price Data):**
   - Data Source: Yahoo Finance
   - Date Range: January 2, 2001, to March 31, 2022
   - Data Includes: Open, High, Low, Volume of BSE Sensex
   
2. **Textual Data (News Headlines):**
   - Data Source: [India News Headlines Dataset](https://bit.ly/36fFPI6)
   - Information: News headlines data with dates

### Project Workflow:

1. **Data Collection:**
   - Numerical data is collected from Yahoo Finance and stored in a CSV file.
   - Textual news headlines data is downloaded from the provided source.

2. **Data Preprocessing:**
   - Missing values are handled.
   - Duplicates are removed from the news headlines dataset.
   - The 'Date' column is converted to a datetime format.

3. **Sentiment Analysis:**
   - TextBlob is used to perform sentiment analysis on the news headlines, categorizing them as negative, neutral, or positive based on polarity scores.
   - Overall sentiment distribution is visualized using a pie chart.

4. **Hybrid Model Creation:**
   - Numerical and textual datasets are merged based on the 'Date' column.
   - Sentiment labels are one-hot encoded and added to the dataset.
   - Subjectivity and polarity scores are calculated for the news headlines and added to the textual data.
   - Sentiment scores (Compound, Negative, Neutral, Positive) are added to the hybrid dataset.
   - Relevant columns (Open, High, Low, Volume, Compound, Negative, Neutral, Positive) are selected for analysis.

5. **Model Training and Testing:**
   - The data is split into features (X) and the target (y).
   - Machine learning models, including Logistic Regression, Random Forest, AdaBoost, Gradient Boosting, Linear Discriminant Analysis, and Decision Tree models, are trained and tested.
   - Data quality is checked, and missing values are handled.

6. **Model Evaluation:**
   - Model accuracy is calculated for each trained model.
   - The Gradient Boosting model is identified as the best-performing model.

7. **Visualization:**
   - A line chart is created to visualize actual vs. predicted labels for the best model (Gradient Boosting).
   - A confusion matrix is computed and visualized as a heatmap to further evaluate model performance.

### Conclusion:

In conclusion, this hybrid model combining numerical analysis and sentiment analysis proved to be effective in predicting stock price/performance. The Gradient Boosting model outperformed other models, achieving an accuracy of approximately 81.56%. This analysis demonstrates the potential of leveraging both numerical and textual data for stock market prediction.

### Instructions for Running the Code:


Enjoy exploring the world of stock market prediction!
