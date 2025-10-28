# Iris Flower Classification

This project demonstrates the classification of Iris flowers into three species—Setosa, Versicolor, and Virginica—using machine learning techniques. The goal is to predict the species of an Iris flower based on its sepal and petal measurements.

## Dataset Overview

The dataset used in this project is the Iris dataset, which consists of 150 samples from each of three species of Iris flowers. Each sample is consists of four features:

* Sepal Length (cm)
* Sepal Width (cm)
* Petal Length (cm)
* Petal Width (cm)

These features are used to train different machine learning models to classify the species of Iris flowers.

## Methodology

The project employs a **Logistic-Regression, Decision-Tree and K-Nearest Neighbors (KNN)** classifier to predict the species of Iris flowers. The steps involved are:

1. **Data Preprocessing**:
   * Data Wrangling was done to change all data to integer or float type.
   * After data wrangling, Exploratory Data Analysis (EDA) is done to understand about data distrubution and relationship between features.
   * For EDA matplotlib and seaborn library were used to visualize the data. The skewness of data was observed. Then correlation between features were seen. Outliers were handled using Inter-Quartile Range (IQR).
   * Finally train.csv and train2.csv were created.
3. **Model Training**:
    * Logistic Regression model was trained by splitting the data into train(0.6), validation(0.2) and test(0.2). As 3 classes needed to be classified, the logistic regression model was trained for individual classes. The model with highest BCE Loss was ignored and other two models were used to create the final prediction.
    * Decision Tree model was used from scikit-learn library. The metrics in the library were used for accuracy.
    * KNN model was fit by splitting the data into train(0.8) and test(0.2) dataset. The model was fit in train and prediction for test were returned.
5. **Model Evaluation**: 
    * The accuracy, precision, recall and F1-Score were used as metrics for the models.

## Repository Structure

* `README.md`: Project documentation.
* `datawrangling.ipynb`: Jupyter notebook for data preprocessing.
* `decision_tree.ipynb`: Jupyter notebook for Decision Tree model implementation.
* `eda.ipynb`: Jupyter notebook for Exploratory Data Analysis.
* `knn.ipynb`: Jupyter notebook for KNN model implementation.
* `logistic_regression.ipynb`: Jupyter notebook for Logistic Regression model implementation.
* `train.csv`: Training dataset.
* `train2.csv`: Additional training dataset.
* `wrangled_data.csv`: Cleaned and preprocessed dataset.

## Getting Started

To run the project locally, follow these steps:

1. Clone the repository:

```bash
git clone https://github.com/Helpmep1s-ux/Iris_flower_classification.git
```

2. Navigate into the project directory:

```bash
cd Iris_flower_classification
```

3. Install the required dependencies:

```bash
pip install -r requirements.txt
```

4. Launch Jupyter Notebook:

```bash
jupyter notebook
```

5. Open the desired notebook (e.g., `knn.ipynb`) to begin.

## Results

The KNN model achieved an accuracy of approximately 97% on the test dataset. Detailed performance metrics, including precision, recall, and F1-score, are available in the respective notebooks.
