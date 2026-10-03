# California House Price Prediction

A Machine Learning project that predicts California house prices using different regression algorithms and provides a simple web interface using Flask.

## Project Overview

The goal of this project is to build a machine learning model capable of predicting house prices based on various features of California housing data.

Multiple regression algorithms were explored and compared, including:

* Linear Regression
* Decision Tree Regressor
* Random Forest Regressor

After comparing the models, **Random Forest Regressor** was selected as the final model because it provided better predictive performance for this dataset.

The trained model is integrated with a **Flask web application**, allowing users to enter housing information and receive a predicted house price.

## Features

* California house price prediction
* Data preprocessing and analysis
* Comparison of multiple regression models
* Random Forest based prediction
* Flask web application
* User-friendly prediction interface
* Separate input and output workflow

## Technologies Used

### Programming

* Python

### Machine Learning

* Scikit-learn
* NumPy
* Pandas

### Web Development

* Flask
* HTML
* CSS

### Development Tools

* VS Code
* Git
* GitHub

## Machine Learning Workflow

```text
California Housing Dataset
          ↓
Data Loading
          ↓
Data Preprocessing
          ↓
Exploratory Data Analysis
          ↓
Train-Test Split
          ↓
Model Training
          ↓
Linear Regression
          ↓
Decision Tree Regressor
          ↓
Random Forest Regressor
          ↓
Model Comparison
          ↓
Final Prediction Model
          ↓
Flask Web Application
```

## Dataset

The project uses California housing data containing information related to housing characteristics and their corresponding house prices.

The dataset is used to train and evaluate regression models for predicting house prices.

## Project Structure

```text
California-House-Price-Prediction/
│
├── flask_app/
│   ├── app.py
│   ├── static/
│   │   └── style.css
│   └── templates/
│       └── index.html
│
├── housing.csv
├── input.csv
├── input - Copy.csv
├── output.csv
├── main.py
├── main_old.py
├── .gitignore
└── README.md
```

## Model Development

Three regression algorithms were explored:

### 1. Linear Regression

Used as a baseline regression model to understand the relationship between the housing features and target price.

### 2. Decision Tree Regressor

A tree-based regression algorithm capable of capturing nonlinear relationships in the dataset.

### 3. Random Forest Regressor

An ensemble learning algorithm that combines multiple decision trees to improve prediction performance and reduce overfitting compared with a single decision tree.

**Random Forest Regressor was used as the final prediction model.**

## Flask Application

The machine learning model is connected to a Flask backend.

The application allows users to:

1. Enter housing-related information.
2. Submit the input through the web interface.
3. Process the input using the trained machine learning pipeline.
4. Generate a predicted house price.
5. Display the prediction to the user.

## How to Run the Project

### 1. Clone the repository

```bash
git clone https://github.com/vivekrahangdale07/California-House-Price-Prediction.git
```

### 2. Navigate to the project

```bash
cd California-House-Price-Prediction
```

### 3. Install dependencies

```bash
pip install pandas numpy scikit-learn flask
```

### 4. Run the Flask application

```bash
python flask_app/app.py
```

### 5. Open the application

Open the local Flask URL shown in the terminal, usually:

```text
http://127.0.0.1:5000/
```

## Trained Model Files

The trained model files (`model.pkl` and `pipelen.pkl`) are intentionally excluded from this GitHub repository because of their large file size.

They can be generated locally by running the model training workflow.

## Future Improvements

* Improve model performance through hyperparameter tuning
* Add additional regression algorithms
* Add interactive data visualizations
* Improve frontend design
* Deploy the Flask application online
* Add automated model retraining
* Add model performance metrics to the web interface

## Author

**Vivek Rahangdale**

B.Tech – Artificial Intelligence & Data Science

GitHub: [vivekrahangdale07](https://github.com/vivekrahangdale07)

## License

This project is created for educational and portfolio purposes.

