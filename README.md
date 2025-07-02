# Car Price Prediction Using Linear Regression

## Overview
This project implements a Linear Regression model to predict the selling price of used cars based on various features. The notebook `Car_Price_Prediction (1).ipynb` processes a dataset (`car data (1).csv`), performs data preprocessing, trains a linear regression model using scikit-learn, and evaluates its performance.

## Objective
The goal is to build a regression model to predict the `Selling_Price` of cars based on features such as `Year`, `Present_Price`, `Driven_kms`, `Fuel_Type`, `Selling_type`, `Transmission`, and `Owner`.

## Dataset
The dataset is a CSV file (`car data (1).csv`) containing the following columns:
- `Car_Name`: Name of the car (dropped due to high cardinality with 98 unique values).
- `Year`: Year of manufacture.
- `Selling_Price`: Selling price of the car (target variable, in lakhs).
- `Present_Price`: Current market price of the car (in lakhs).
- `Driven_kms`: Total kilometers driven.
- `Fuel_Type`: Fuel type (Petrol, Diesel, CNG).
- `Selling_type`: Selling type (Dealer, Individual).
- `Transmission`: Transmission type (Manual, Automatic).
- `Owner`: Number of previous owners (0, 1, or 3).

## Dependencies
The project requires the following Python libraries:
- `numpy`
- `pandas`
- `statsmodels`
- `matplotlib`
- `seaborn`
- `scikit-learn`

Install dependencies using:
```bash
pip install numpy pandas statsmodels matplotlib seaborn scikit-learn
```

## Notebook Structure
1. **Data Loading**:
   - Loads the dataset using `pandas.read_csv` from the path `C:\Users\ohkba\Downloads\car data (1).csv`.
   - Displays the first few rows using `raw_data.head()`.

2. **Preprocessing**:
   - Drops the `Car_Name` column due to its high cardinality (98 unique values, leading to 97 dummy variables if encoded).
   - Uses `raw_data.describe(include='all')` to explore descriptive statistics, revealing:
     - 301 records, no missing values.
     - Categorical variables: `Fuel_Type` (3 unique values: Petrol, Diesel, CNG), `Selling_type` (2 unique values: Dealer, Individual), `Transmission` (2 unique values: Manual, Automatic).
     - Numerical variables: `Year`, `Selling_Price`, `Present_Price`, `Driven_kms`, `Owner`.
   - Likely encodes categorical variables (`Fuel_Type`, `Selling_type`, `Transmission`) into dummy variables (not shown in the snippet but implied for regression).
   - Splits data into training and testing sets (assumed, as `x_test` and `y_test` are used later).

3. **Model Training**:
   - Uses `sklearn.linear_model.LinearRegression` to train a linear regression model (`reg`).
   - The exact features used (`x_test`) and target (`y_test`) are not shown but likely include encoded categorical variables and numerical features.

4. **Model Evaluation**:
   - Plots actual vs. predicted `Selling_Price` using `matplotlib` to visualize trends.
   - Computes R-squared for the test set using `reg.score(x_test, y_test)`, yielding an R-squared value of approximately 86.7%.
   - The model explains ~87% of the variability in the data, indicating a good fit.

## Output
- **Trend Plot**: A line plot comparing actual and predicted `Selling_Price` values for the test set, created using `matplotlib`.
- **R-squared**: The model achieves an R-squared of ~86.7%, indicating it captures 87% of the variance in `Selling_Price`.

## Usage
1. Clone the repository:
   ```bash
   git clone <repository_url>
   ```
2. Place `car data (1).csv` in the appropriate directory (e.g., `C:\Users\ohkba\Downloads\`).
3. Update the file path in the notebook to match your local environment.
4. Run the notebook cells sequentially to:
   - Load and preprocess the data.
   - Train the linear regression model.
   - Evaluate and visualize the results.

## Notes
- **Model Performance**: The R-squared value of ~86.7% suggests a strong model, but further evaluation (e.g., Mean Squared Error, Residual Plots) could provide more insights.
- **File Paths**: The notebook uses a hardcoded path (`C:\Users\ohkba\Downloads\car data (1).csv`). Update this to match your directory structure.
- **Python Version**: The notebook uses Python 3.9. Ensure compatibility with your environment.
- **Missing Code**: The notebook snippet does not show the full preprocessing (e.g., encoding categorical variables, train-test split) or model training steps. Ensure these are included in the complete notebook.
- **Categorical Variables**: The model likely requires encoding of `Fuel_Type`, `Selling_type`, and `Transmission` using techniques like one-hot encoding (`pd.get_dummies`).
- **Assumptions**: The snippet assumes `x_test`, `y_test`, and `y_hat_test` are defined, implying a train-test split and model predictions.

## Future Improvements
- Include additional evaluation metrics (e.g., Mean Absolute Error, Root Mean Squared Error) for a comprehensive assessment.
- Create residual plots to check for model assumptions (linearity, homoscedasticity).
- Handle potential outliers in `Driven_kms` (max: 500,000 km) or `Present_Price` (max: 92.6 lakhs).
- Experiment with feature engineering (e.g., car age from `Year`, log-transformation of skewed variables like `Driven_kms`).
- Try other regression models (e.g., Random Forest, Gradient Boosting) for comparison.
- Add cross-validation to ensure model robustness.

## License
This project is licensed under the MIT License.
