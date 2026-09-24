# Medical Insurance Cost Prediction

A machine learning project that predicts individual medical insurance charges based on personal attributes such as age, sex, BMI, number of children, smoking status and region. The project includes data exploration, preprocessing, a trained regression model and a simple web app for making live predictions.

## Problem Statement

Medical insurance providers need a way to estimate how much a customer is likely to cost them based on that customer's profile. This project builds a regression model that predicts insurance charges from a small set of demographic and health related features, then wraps the model in an interactive app so a user can enter their own details and get an instant estimate.

## Dataset

The project uses the Medical Cost Personal Dataset (insurance.csv), which contains 1338 records with the following columns.

* age: age of the primary beneficiary
* sex: female or male
* bmi: body mass index
* children: number of children or dependents covered by the insurance plan
* smoker: whether the person smokes (yes or no)
* region: residential area in the US (northeast, northwest, southeast, southwest)
* charges: individual medical costs billed by the insurance (target variable)

## Preprocessing

The following steps were applied before training.

1. Checked the dataset for missing values (none found) and duplicate rows
2. Removed duplicate rows
3. Applied one hot encoding to the categorical columns sex, smoker and region using pandas get_dummies
4. Separated the data into features (x) and the target variable charges (y)
5. Split the data into training and test sets using an 80/20 split with a fixed random state for reproducibility

Exploratory analysis included boxplots of charges by smoking status, scatter plots of charges against age and BMI, and a correlation heatmap of the numeric features.

## Model

A Linear Regression model from scikit learn was trained on the preprocessed features. Linear regression was chosen as a simple, interpretable baseline that works well when relationships between features and the target are roughly linear, which fits this dataset reasonably well (particularly the strong relationship between smoking status and charges).

The trained model is saved as model.pkl using pickle, and loaded by the app for inference.

## Evaluation

Performance was measured on the held out test set using the following metrics.

| Metric | Value |
| ------ | ----- |
| MAE | 4177.05 |
| MSE | 35478020.68 |
| RMSE | 5956.34 |
| R2 Score | 0.807 |

An R2 score of about 0.81 means the model explains roughly 81 percent of the variance in insurance charges. The average prediction is off by about 4177 in charges (MAE), which is reasonable given the wide spread of the target variable.

## Application

The prediction app (insurance_app.py) provides a simple interface where a user enters age, sex, BMI, number of children, smoking status and region, and receives a predicted insurance charge from the trained model.

### Running locally

```
git clone <repository url>
cd medical_insurance_cost_prediction
pip install -r requirements.txt
streamlit run insurance_app.py
```

### Deployed application

Live app link: <add your deployed app URL here>

## Screenshots

### GitHub Repository
<add screenshot here>

### README
<add screenshot here>

### Deployed Application
<add screenshot here>

### Working Prediction
<add screenshot here>

### Project Structure
<add screenshot here>

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

## Limitations

* Linear regression assumes roughly linear relationships between features and charges, so it may underpredict or overpredict for unusual combinations of inputs
* The dataset is relatively small (about 1300 records) and comes from a limited set of regions, so the model may not generalize well to other populations or countries
* No hyperparameter tuning or comparison against other model types (such as random forest or gradient boosting) was performed
* Categorical features are limited to the values present in the training data, so new categories at prediction time cannot be handled
* The model does not account for factors known to affect real insurance pricing, such as preexisting conditions, occupation or detailed medical history

## License

This project is licensed under the terms described in the LICENSE file.