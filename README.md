
# Housing Price Prediction

This project involves performing regression analysis to predict housing prices using various machine learning and statistical techniques, specifically linear regression. The dataset is loaded from a CSV file, and the project includes steps for data preprocessing, training, and evaluation.


## Installation

To run this project, you will need to install the following Python libraries:

- NumPy
- pandas 
- Matplotlib 
- seaborn 
- scikit-learn 
- statsmodels.


## Features

- Regression analysis using linear regression.
- Data exploration and visualization.
- Evaluation of model performance.



## Files


- main.py: The main script that runs the regression analysis and evaluation.
- Housing.csv: The dataset used for the analysis (ensure it's available in the correct path).
## Project Steps

    1 Data Loading:

     . The dataset is loaded using Pandas from the CSV file: Housing.csv.

    2 Data Exploration:

     . The script prints the dataset shape, data types, and summary statistics.
     . It checks for missing values and provides basic visualizations to explore   relationships between features.

    3 Data Preprocessing:

     . Data is split into training and test sets using train_test_split.
     . Feature scaling is done using MinMaxScaler.

    4 Analysis:

     . Multiple regression analysis is conducted using linear regression models from statsmodels.
     . VIF (Variance Inflation Factor) is calculated to assess multicollinearity.
    
    5 Evaluation:
    
     . The performance is evaluated using the R-squared metric (r2_score).

## Usage

- To run the project, execute the main.py script:

    bash
    python main.py
## Result

The regression analysis yields valuable insights into housing prices based on various features:

    1 Model Performance: 
  
     . The linear regression model achieved an R-squared value of Z, indicating that Z% of the variance in housing prices is explained by the model.

    2 Key Findings:

     . Properties with more bedrooms significantly correlate with higher prices.

     . Semi-furnished properties generally show a notable price increase.

    3 Visual Insights:

     . The scatter plot of area vs. price illustrates a positive correlation.
  
     . A correlation matrix reveals strong relationships among key features.