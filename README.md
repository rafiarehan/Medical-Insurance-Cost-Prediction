# Medical Insurance Cost Prediction

## Live Demo

https://medical-insurance-cost-predictions.streamlit.app/

## Project Overview

This project predicts medical insurance charges for an individual based on personal attributes such as age, sex, BMI, number of children, smoking status and region. It covers the full workflow from data exploration and cleaning to model training, evaluation and deployment as an interactive web app.

## Problem Statement

Medical insurance providers need a way to estimate how much a customer is likely to cost them based on that customer's profile. Charges vary widely across individuals, and factors like smoking status and BMI have a large effect on cost, so a simple average is not a useful estimate.

## Objective

Build a regression model that predicts insurance charges from a person's demographic and health related information, evaluate its accuracy, and deploy it as a web app so anyone can enter their own details and get an instant estimate.

## Dataset

The project uses the Medical Cost Personal Dataset (insurance.csv), which contains 1338 records with the following columns.

| Column | Description |
| ------ | ----------- |
| age | Age of the primary beneficiary |
| sex | Female or male |
| bmi | Body mass index |
| children | Number of children or dependents covered |
| smoker | Whether the person smokes (yes or no) |
| region | Residential area in the US (northeast, northwest, southeast, southwest) |
| charges | Individual medical costs billed by insurance (target variable) |

## Features Used

The model is trained on all columns except charges: age, sex, bmi, children, smoker and region. The categorical columns (sex, smoker, region) are converted into numeric form through one hot encoding before training.

## Data Preprocessing

The following steps were applied before training.

1. Checked the dataset for missing values (none found) and duplicate rows
2. Removed duplicate rows
3. Applied one hot encoding to the categorical columns sex, smoker and region using pandas get_dummies
4. Separated the data into features (x) and the target variable charges (y)
5. Split the data into training and test sets using an 80/20 split with a fixed random state for reproducibility

Exploratory analysis included a boxplot of charges by smoking status, scatter plots of charges against age and BMI, and a correlation heatmap of the numeric features.

## Machine Learning Model

A Linear Regression model from scikit learn was trained on the preprocessed features. Linear regression was chosen as a simple, interpretable baseline that works well when relationships between features and the target are roughly linear, which fits this dataset reasonably well, particularly the strong relationship between smoking status and charges.

The trained model is saved as model.pkl using pickle and loaded by the app for inference.

## Model Evaluation

Performance was measured on the held out test set using the following metrics.

| Metric | Value |
| ------ | ----- |
| MAE | 4177.05 |
| MSE | 35478020.68 |
| RMSE | 5956.34 |
| R2 Score | 0.807 |

An R2 score of about 0.81 means the model explains roughly 81 percent of the variance in insurance charges. The average prediction is off by about 4177 in charges (MAE), which is reasonable given the wide spread of the target variable.

## Prediction Examples

A sample comparison of actual versus predicted charges on the test set.

| Actual Charges | Predicted Charges |
| --------------- | ------------------ |
| 8688.86 | 8143.69 |
| 5708.87 | 5737.12 |
| 11436.74 | 14369.31 |
| 38746.36 | 31745.51 |
| 4463.21 | 8962.39 |

## Technologies Used

* Python
* pandas and numpy for data handling
* matplotlib and seaborn for visualization
* scikit learn for model training and evaluation
* pickle for saving the trained model
* Streamlit for the web application and deployment

## Project Structure

```
medical_insurance_cost_prediction/
    insurance_app.py     application used to make predictions
    insurance.csv         dataset used for training and evaluation
    main.ipynb            notebook covering exploration, preprocessing, training and evaluation
    model.pkl              trained linear regression model
    requirements.txt      python packages required to run the project
    README.md             project documentation
    LICENSE                 license for this repository
```

## How to Run Locally

```
git clone <repository url>
cd medical_insurance_cost_prediction
pip install -r requirements.txt
streamlit run insurance_app.py
```

The app will open in your browser, typically at http://localhost:8501.

## Deployment

The application is deployed on Streamlit Community Cloud, which builds and hosts the app directly from the GitHub repository.

Live app: https://medical-insurance-cost-predictions.streamlit.app/

## Limitations

* Linear regression assumes roughly linear relationships between features and charges, so it may underpredict or overpredict for unusual combinations of inputs
* The dataset is relatively small (about 1300 records) and comes from a limited set of regions, so the model may not generalize well to other populations or countries
* No hyperparameter tuning or comparison against other model types (such as random forest or gradient boosting) was performed
* Categorical features are limited to the values present in the training data, so new categories at prediction time cannot be handled
* The model does not account for factors known to affect real insurance pricing, such as preexisting conditions, occupation or detailed medical history

## Future Improvements

* Compare linear regression against other models such as random forest or gradient boosting to see if accuracy improves
* Add cross validation and hyperparameter tuning
* Include additional features if available, such as preexisting conditions or occupation
* Add input validation and error handling to the app
* Add confidence intervals or a range alongside the point prediction

## Author

<Rafia Rehan>
GitHub: <github.com/rafiarehan>