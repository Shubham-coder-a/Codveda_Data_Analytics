📊 Codveda Data Analytics Internship Project
📌 Project Overview

This project is part of the Data Analytics Internship at Codveda Technologies.
The objective of this project is to perform data cleaning, exploratory data analysis (EDA), data visualization, and machine learning modeling using the Iris dataset.

The project demonstrates a complete end-to-end data analytics workflow, from raw data to predictive modeling.

📂 Project Structure
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

🧪 Dataset Used

Dataset: Iris Dataset

Features:

Sepal Length

Sepal Width

Petal Length

Petal Width

Target Variable: Species (Setosa, Versicolor, Virginica)

✅ Task 1: Data Cleaning

File: task1_data_cleaning.py

Steps Performed:

Loaded dataset

Checked missing values

Removed duplicate rows

Standardized column names

Saved cleaned dataset

Output:

Cleaned dataset with 147 rows and 5 columns

File saved as iris_cleaned.csv

📊 Task 2: Exploratory Data Analysis (EDA)

File: task2_eda.py

Analysis Included:

Summary statistics (mean, std, min, max)

Species distribution analysis

Initial data insights

Key Insight:

Dataset is well-balanced across species

Petal features show strong variation across species

📈 Task 3: Data Visualization

File: task3_visualization.py

Visualizations Generated:

Histograms – Feature distributions

Boxplot – Outlier and spread analysis

Scatter Plot – Petal Length vs Petal Width (species-wise separation)

Correlation Heatmap – Feature relationships

📁 All visualizations are saved in the outputs/ folder.

🤖 Task 4: Machine Learning Model

File: task4_ml_model.py

Model Used:

Logistic Regression

Workflow:

Train-test split (80%-20%)

Model training

Prediction and evaluation

📌 Results:

Accuracy: 93%

Setosa classified with 100% precision and recall

Strong overall performance across all species

🏁 Conclusion

This project successfully demonstrates:

Data cleaning and preprocessing

Exploratory data analysis

Insightful data visualization

Machine learning classification with high accuracy

The results highlight the importance of petal features in Iris species classification and showcase a complete data analytics pipeline.

🛠️ Technologies Used

Python

Pandas

Matplotlib

Seaborn

Scikit-learn

👤 Author

Shubham Raju Nagpure
Data Analytics Intern – Codveda Technologies

🔗 Acknowledgement

Thanks to Codveda Technologies for providing this internship opportunity and dataset for hands-on learning.