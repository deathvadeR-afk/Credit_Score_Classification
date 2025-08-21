# PaisaBazaar Credit Score Classification Project

## Project Overview

This project aims to develop a predictive model that estimates an individual's credit score based on key financial indicators such as income, credit card usage, and payment behavior. Credit scores are critical in determining a person's creditworthiness and influence decisions related to loan approvals, interest rates, and financial risk assessments.

## Problem Statement

The goal is to leverage machine learning techniques to provide a data-driven approach to credit scoring, enabling more efficient and transparent financial evaluations. The model predicts credit scores into three categories: Good, Standard, and Poor.

## Dataset

The project uses the `dataset-2.csv` file containing 100,000 records with 28 features including:
- Customer demographics (Age, Occupation)
- Financial information (Annual Income, Monthly Salary)
- Credit history (Credit Mix, Payment Behavior, Outstanding Debt)
- Banking details (Number of Bank Accounts, Credit Cards)

## Project Structure

```
PaisaBazaar Project Credit Score Classification/
├── README.md
├── requirements.txt
├── Notebook.ipynb
├── dataset-2.csv
├── best_xgb_model.pkl
└── Paisabazaar.pptx
```

## Installation

1. Clone or download this repository
2. Install required dependencies:
```bash
pip install -r requirements.txt
```

## Dependencies

- pandas >= 1.5.0
- numpy >= 1.21.0
- scikit-learn >= 1.1.0
- xgboost >= 1.6.0
- imbalanced-learn >= 0.9.0
- matplotlib >= 3.5.0
- seaborn >= 0.11.0
- scipy >= 1.9.0
- joblib >= 1.1.0
- jupyter >= 1.0.0

## Methodology

### 1. Data Understanding and Cleaning
- Thorough analysis of the dataset
- Standardization of missing value representations
- Data type corrections
- Removal of irrelevant columns (SSN, Name)

### 2. Feature Engineering
- Created Debt-to-Income Ratio feature
- Transformed Type_of_Loan column into binary features
- Age binning for non-linear relationships
- Feature scaling and encoding

### 3. Exploratory Data Analysis (EDA)
- Univariate, Bivariate, and Multivariate analysis
- Statistical hypothesis testing
- 15+ meaningful visualizations with business insights

### 4. Model Development
Three models were implemented and compared:
- **Logistic Regression** (Baseline)
- **Random Forest Classifier**
- **XGBoost Classifier** (Best performing)

### 5. Class Imbalance Handling
- Applied SMOTE (Synthetic Minority Oversampling Technique)
- Ensured models are not biased towards majority class

### 6. Model Optimization
- Hyperparameter tuning using RandomizedSearchCV
- Cross-validation for robust performance evaluation
- Feature importance analysis

## Key Findings

1. **Income Impact**: Statistically significant relationship between monthly income and credit score
2. **Credit Mix Importance**: Diverse credit portfolio significantly associated with better credit scores
3. **Payment Behavior**: Number of delayed payments is a powerful predictor of credit risk

## Model Performance

The final XGBoost Classifier achieved the best performance after hyperparameter tuning and was selected as the production model. The trained model is saved as `best_xgb_model.pkl`.

## Usage

1. Open the Jupyter notebook:
```bash
jupyter notebook Sample_ML_Submission_Template.ipynb
```

2. Run all cells to reproduce the analysis

3. Load the pre-trained model:
```python
import joblib
model = joblib.load('best_xgb_model.pkl')
```

## File Descriptions

- `Sample_ML_Submission_Template.ipynb`: Main analysis notebook with complete workflow
- `dataset-2.csv`: Training dataset with customer financial information
- `best_xgb_model.pkl`: Trained XGBoost model ready for predictions
- `Paisabazaar.pptx`: Project presentation slides
- `requirements.txt`: Python package dependencies

## Business Impact

The model provides:
- Automated credit scoring reducing manual evaluation time
- Data-driven decisions for loan approvals
- Risk assessment for financial institutions
- Transparent credit evaluation process

## Future Enhancements

- Real-time prediction API development
- Model monitoring and drift detection
- Integration with banking systems
- Advanced feature engineering techniques

## Contributing

This project was developed as part of a Codecademy machine learning course. For questions or improvements, please refer to the project guidelines.

## License

This project is for educational purposes as part of the Codecademy curriculum.
