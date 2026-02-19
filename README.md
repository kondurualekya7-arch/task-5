# 🚢 Titanic: Exploratory Data Analysis (EDA)

## 📌 Project Overview
The goal of this project is to perform a comprehensive Exploratory Data Analysis (EDA) on the classic Titanic dataset. By leveraging visual and statistical exploration, we aim to identify key patterns and factors—such as gender, age, and socio-economic status—that influenced passenger survival rates.

## 🛠️ Tools & Libraries
This project utilizes the standard Python data science stack:
* **Pandas:** For data manipulation, cleaning, and structural analysis.
* **Matplotlib:** For low-level plotting control and figure labeling.
* **Seaborn:** For high-level statistical visualizations (Heatmaps, Countplots, Boxplots).

---

## 🧹 Data Preprocessing
To ensure the integrity of the analysis, the following data-cleaning steps were implemented:
* **Age Imputation:** Missing values (~20%) were filled using **Median Imputation** to maintain a realistic distribution while mitigating the influence of outliers.
* **Feature Dropping:** The `Cabin` column was removed due to excessive missing data (approx. 77%), which would have introduced significant noise.
* **Embarked:** Missing entries were filled using the **Mode** (most frequent port of departure).

---

## 📊 Key Visual Insights

### 1. Survival Distribution
The majority of passengers did not survive, highlighting a significant class imbalance in the target variable.

### 2. The "Women and Children First" Rule
Bivariate analysis confirmed that **Gender** was a primary factor. Females had a significantly higher survival probability compared to males.

### 3. Socio-Economic Impact (Pclass)
Passengers in **1st Class** had the highest survival rate. Conversely, those in **3rd Class** (the largest group) faced the highest mortality rate.

### 4. Correlation Analysis
A **Heatmap** was used to identify relationships between numerical variables. A strong **negative correlation (-0.55)** was found between `Pclass` and `Fare`, confirming that higher fares (1st Class) were directly linked to better survival outcomes.

---

## 📝 Summary of Findings
Survival on the Titanic was not random; it was heavily dictated by **socio-economic status (Pclass)** and **gender**. The EDA process successfully identified these trends and handled data anomalies (missing values) to prepare the dataset for future predictive modeling.

---

## 🎓 Interview Quick-Ref
**Q: What is the difference between a Heatmap and a Pairplot?**
* **Heatmap:** A summarized numerical grid showing correlation coefficients.
* **Pairplot:** A detailed grid of scatterplots for every possible variable pair to show relationship shapes.

**Q: How do you handle skewed data?**
* By applying **Log Transformations** or utilizing robust statistical measures like the **Median** instead of the Mean.

---

## 💻 Sample Code Snippet
```python
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load and Clean
df = pd.read_csv('train.csv')
df['Age'] = df['Age'].fillna(df['Age'].median())
df.drop(columns=['Cabin'], inplace=True)

# Visualize Correlation
plt.figure(figsize=(10, 8))
sns.heatmap(df.select_dtypes(include=['number']).corr(), annot=True, cmap='coolwarm')
plt.show()
