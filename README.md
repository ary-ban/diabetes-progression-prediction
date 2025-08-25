## Diabetes Progression Prediction

This project focuses on predicting the progression of diabetes in patients using the Diabetes dataset from scikit‑learn. Understanding how the disease progresses helps clinicians adjust treatment and monitor patient outcomes.

### Dataset

The Diabetes dataset contains 442 samples with ten baseline variables (age, sex, body mass index, blood pressure and six blood serum measurements) and a target score representing disease progression one year after baseline. Because the dataset ships with scikit‑learn, it does not require an internet connection.

### Objective

Train models to estimate the diabetes progression score based on a patient’s baseline measurements.

### Methodology

* **Data loading** – Use `sklearn.datasets.load_diabetes` to load the dataset.
* **Preprocessing** – Standardise features with `StandardScaler` to ensure each variable contributes equally.
* **Models** – Fit a `RandomForestRegressor` to capture nonlinear patterns and a `LinearRegression` model as a simple baseline.
* **Evaluation** – Split the data into 80% training and 20% testing. Compute the coefficient of determination (R²) and root‑mean‑squared error (RMSE) for both models. Report the mean target value as a naive baseline for comparison.

### Usage

To run the script:

```bash
python diabetes_prediction.py
```

The script prints the R² and RMSE for the Random Forest and linear regression models, along with the mean target value. Feel free to experiment with other regression algorithms or tune the hyperparameters for improved performance.

### Possible improvements

* **Hyperparameter tuning** – Adjust the number of trees, maximum depth and other parameters of the random forest to optimise performance.
* **Feature importance** – Examine feature importance scores to see which variables most influence disease progression.
* **Visualisation** – Plot predicted vs. actual progression scores and residuals to assess model fit.
