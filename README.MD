# 📊 Codveda Data Analytics Internship Project

## 📌 Project Overview
This project was completed as part of the **Data Analytics Internship at Codveda Technologies**.  
The objective of this project is to perform **data cleaning, exploratory data analysis (EDA), data visualization, and machine learning modeling** using the **Iris dataset**.

It demonstrates a complete **end-to-end data analytics workflow**, starting from raw data processing to predictive modeling and evaluation.

---

## 📂 Project Structure
Codveda_Data_Analytics/
│
├── task1_data_cleaning.py
├── task2_eda.py
├── task3_visualization.py
├── task4_ml_model.py
├── iris_cleaned.csv
│
└── outputs/
├── histograms.png
├── boxplot.png
├── scatter_plot.png
└── heatmap.png


---

## 🧪 Dataset Used
**Dataset:** Iris Dataset  

**Features:**
- Sepal Length  
- Sepal Width  
- Petal Length  
- Petal Width  

**Target Variable:**  
- Species (Setosa, Versicolor, Virginica)

---

## ✅ Task 1: Data Cleaning
**File:** `task1_data_cleaning.py`

### Steps Performed:
- Loaded the dataset  
- Checked for missing values  
- Removed duplicate rows  
- Standardized column names  
- Saved the cleaned dataset  

**Output:**
- Cleaned dataset with **147 rows and 5 columns**
- Saved as `iris_cleaned.csv`

---

## 📊 Task 2: Exploratory Data Analysis (EDA)
**File:** `task2_eda.py`

### Analysis Included:
- Summary statistics (mean, standard deviation, min, max)  
- Species distribution analysis  
- Initial data insights  

**Key Insights:**
- Dataset is well-balanced across species  
- Petal features show strong variation among different species  

---

## 📈 Task 3: Data Visualization
**File:** `task3_visualization.py`

### Visualizations Generated:
- **Histograms** – Feature distributions  
- **Boxplots** – Outlier and spread analysis  
- **Scatter Plot** – Petal Length vs Petal Width (species-wise separation)  
- **Correlation Heatmap** – Feature relationships  

📁 All visualizations are saved in the `outputs/` folder.

---

## 🤖 Task 4: Machine Learning Model
**File:** `task4_ml_model.py`

### Model Used:
- Logistic Regression  

### Workflow:
- Train-test split (80% training, 20% testing)  
- Model training  
- Prediction and evaluation  

### 📌 Results:
- **Accuracy:** 93%  
- **Setosa** classified with **100% precision and recall**  
- Strong overall performance across all species  

---

## 🏁 Conclusion
This project successfully demonstrates:
- Data cleaning and preprocessing  
- Exploratory data analysis  
- Insightful data visualization  
- Machine learning classification with high accuracy  

The results highlight the importance of **petal features** in Iris species classification and showcase a complete data analytics pipeline.

---

## 🛠️ Technologies Used
- Python  
- Pandas  
- Matplotlib  
- Seaborn  
- Scikit-learn  

---

## 👤 Author
**Shubham Raju Nagpure**  
Data Analytics Intern – Codveda Technologies  

---

## 🔗 Acknowledgement
Thanks to **Codveda Technologies** for providing this internship opportunity and dataset for hands-on learning.
