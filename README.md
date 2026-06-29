# Salary_Prediction_Analysis_Python
Overview

This project investigates the key factors that influence salary in the modern workforce using multiple linear regression. Using a dataset of 1,000 observations sourced from Kaggle, the analysis explores how education, experience, job title, location, gender, and age affect annual compensation — and whether the impact of experience on salary differs between genders.


🛠️ Tools & Technologies


Python
Pandas
Statsmodels
Google Colab



📂 Dataset


Source: Salary Prediction Data – Kaggle
Size: 1,000 observations
Dependent Variable: Annual Salary
Independent Variables:



VariableTypeDescriptionEducationCategoricalHigh School, Bachelor's (ref), Master's, PhDExperienceContinuousYears of professional experienceGenderCategoricalMale / FemaleLocationCategoricalUrban, Suburban, Rural (ref)Job TitleCategoricalDirector, Engineer, Manager, Analyst (ref)AgeContinuousAge in years


🗄️ Setup

```python
pythonfrom google.colab import files
uploaded = files.upload()

import pandas as pd
df = pd.read_csv('salary_prediction_data.csv')

print(df.head(6))
```

🔍 Analysis & Solutions

1️⃣ Base Model — Salary Predicted by Education, Experience, Gender, Location, Job Title, and Age

Fits a multiple linear regression model using dummy variables for all categorical predictors.

pythonimport statsmodels.api as sm
from statsmodels.formula.api import ols

```python
model_dummies_only = ols(
    "Salary ~ Education + Experience + C(Gender) + C(Location) + C(Job_Title) + Age",
    data=df
).fit()

print("Model with dummy variables only:")
print(model_dummies_only.summary())
```

2️⃣ Interaction Model — Does Experience Affect Salary Differently by Gender?

Adds an interaction term between Experience and Gender to test whether returns to experience differ across genders.

```python
pythonmodel_with_interaction = ols(
    "Salary ~ Education + Experience + C(Gender) + C(Location) + C(Job_Title) + Experience*C(Gender) + Age",
    data=df
).fit()

print("\nModel with interaction term:")
print(model_with_interaction.summary())
```

📊 Key Findings

🎓 Education

Education has a strong and statistically significant effect on salary (all p < 0.001):


PhD: +$40,070 vs. Bachelor's (reference)
Master's: +$21,000
High School: −$18,040


📍 Location


Urban: +$9,280 over Rural (reference)
Suburban: +$3,910 over Rural


💼 Job Title


Director: +$25,330 over Analyst (reference)
Manager: +$15,750
Engineer: +$4,800


📈 Experience

Each additional year of professional experience adds approximately $1,000–$1,030 to annual salary.

👥 Gender & Age

Neither gender nor age showed a statistically significant direct effect on salary after controlling for all other variables (p = 0.420 and p = 0.670, respectively). The interaction between experience and gender was also non-significant (p = 0.440), suggesting that returns to experience are consistent across genders.


📈 Model Performance

Both models achieved an R² of 0.878, meaning the variables collectively explain approximately 88% of salary variance. The interaction term added no meaningful explanatory power.


Conclusion

Education, job title, location, and years of experience are the strongest predictors of salary. Gender and age had no significant direct impact on compensation in this dataset. The consistent returns to experience across genders suggest that career progression is rewarded equitably regardless of gender — though further causal research would be needed to draw firm conclusions.


References

MrSimple. (2024, February 5). Salary prediction data. Kaggle. https://www.kaggle.com/datasets/mrsimple07/salary-prediction-data
