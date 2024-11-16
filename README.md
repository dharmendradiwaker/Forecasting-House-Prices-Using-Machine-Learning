# Forecasting House Prices Using Machine Learning 🏡📊

## Overview
This project focuses on predicting house prices using machine learning techniques. The dataset consists of over **1,000,000+ rows** and **12 columns** containing information about various house attributes. The goal is to build predictive models to estimate house prices based on these attributes. The project explores different machine learning models, including **Linear Regression**, **Decision Trees**, and **Random Forests** using **scikit-learn**.

## Key Highlights 🔑
- **Data Cleaning & Analysis**: Processed and analyzed a large dataset with over **1,000,000 rows** to prepare it for modeling.
- **Models Trained**:
  - **Linear Regression**: Simple but effective for baseline prediction.
  - **Decision Tree**: Captures non-linear relationships between features and target.
  - **Random Forests**: Ensemble method that improved accuracy by combining multiple decision trees.
- **Model Performance**: Achieved a **Root Mean Square Error (RMSE)** of **866,152** after **hyperparameter tuning**. 🏆

## Dataset 📁
The dataset contains more than **1,000,000 rows** and **12 columns**, including features like:
- **Price of the house** 💰 (target variable)
- **Date of Transfer** 📅
- **Property Type** 🏠 (e.g., detached, semi-detached, terraced, flat)
- **Old/New** 🏡 (indicates whether the property is newly built or existing)
- **Duration** ⏳ (e.g., freehold or leasehold)
- **Town/City** 📍
- **District** 🏢
- **County** 🌍
- **PPDCategory Type** 🔖 (indicates if the property was a full or partial sale)
- **Record Status** 🗂️ (applicable to monthly file updates)

## Requirements 📋
- Python (version 3.6 or higher recommended)
- Required Python libraries:
  - scikit-learn (for machine learning algorithms) 🤖
  - Pandas (for data manipulation) 📊
  - NumPy (for numerical operations) 🔢
  - Matplotlib & Seaborn (for data visualization) 🎨

You can install these dependencies using pip:
```bash
pip install scikit-learn pandas numpy matplotlib seaborn
```

## Workflow ⚙️
1. **Data Cleaning**: Missing values were handled, and categorical variables were encoded.
2. **Feature Engineering**: Created new features or transformed existing ones to improve model accuracy.
3. **Model Training**: Trained multiple machine learning models, including:
   - Linear Regression (baseline)
   - Decision Tree Regressor
   - Random Forest Regressor
4. **Model Evaluation**: Evaluated model performance using **Root Mean Square Error (RMSE)**.
5. **Hyperparameter Tuning**: Used grid search to tune the hyperparameters and improve model performance.

## Setup 🧑‍💻
1. Clone this repository to your local machine:
   ```bash
   git clone https://github.com/your_username/forecasting-house-prices-using-machine-learning.git
   ```

2. Navigate to the project folder:
   ```bash
   cd forecasting-house-prices-using-machine-learning
   ```

3. Install the required libraries:
   ```bash
   pip install -r requirements.txt
   ```

4. Run the Jupyter Notebook or Python script:
   ```bash
   jupyter notebook House_Price_Prediction.ipynb
   ```

## How to Use 🧑‍💻
1. Load the house price dataset from the provided `data/` folder.
2. Follow the steps in the notebook or script to clean, preprocess, and train the models.
3. Explore the performance metrics of each model and see the final predictions.
4. You can adjust hyperparameters or add new features to improve the model's accuracy.

## Conclusion 📌
This project demonstrates how to predict house prices using machine learning algorithms, offering insights into the key factors that influence the price of a property. The achieved **Root Mean Square Error (RMSE)** of **866,152** indicates that the model is performing fairly well after fine-tuning, although there is still room for improvement with further feature engineering or advanced techniques.

## Contributors 🙋‍♂️
- @Dharmendradiwaker12

---
