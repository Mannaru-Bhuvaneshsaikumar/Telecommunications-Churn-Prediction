# Telecommunications-Churn-Prediction

# 📞 Telecommunications Churn Prediction

## 📁 Project Overview
This project is focused on analyzing customer churn in the telecommunications industry. The main objective is to develop a predictive model that identifies customers who are likely to discontinue their service, enabling proactive retention strategies.

## 👨‍💻 Team
- Group 4

## 🧠 Problem Statement
Customer churn is a critical metric in the telecom industry. Losing customers directly impacts revenue. The goal of this project is to use historical customer data to build a machine learning model that can predict the likelihood of churn, so that the business can take necessary action to retain valuable customers.

## 📊 Dataset
The dataset contains information on:
- Customer demographics
- Service usage
- Account information
- Churn labels

It includes features such as:
- `customerID`
- `gender`
- `SeniorCitizen`
- `Partner`, `Dependents`
- `tenure`, `MonthlyCharges`, `TotalCharges`
- `Contract`, `PaymentMethod`
- `Churn` (Target Variable)

## 🛠️ Tools & Technologies
- Python
- Jupyter Notebook
- Pandas, NumPy
- Matplotlib, Seaborn
- Scikit-learn
- XGBoost

## 🚀 Key Steps
1. **Data Preprocessing**
   - Handling missing values
   - Encoding categorical variables
   - Feature scaling

2. **Exploratory Data Analysis (EDA)**
   - Visualizing relationships between variables
   - Identifying churn patterns

3. **Model Building**
   - Logistic Regression
   - Random Forest
   - XGBoost
   - Model evaluation using accuracy, precision, recall, F1-score, ROC-AUC

4. **Hyperparameter Tuning**
   - GridSearchCV for optimal performance

## 📈 Model Performance
- The best-performing model is **XGBoost**, achieving high accuracy and recall on the test data.

## ✅ Conclusion
This model can help telecom companies reduce churn by identifying high-risk customers early. The insights can be used to guide customer retention efforts and improve service offerings.

