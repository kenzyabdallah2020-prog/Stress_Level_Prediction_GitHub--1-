# Predicting Student Stress Levels with Machine Learning

## Project Overview

This project explores student well-being data and builds a machine
learning model to predict student stress levels. It demonstrates a data
science workflow covering data cleaning, exploratory data analysis,
feature engineering, classification, and model evaluation.

## Objectives

-   Explore the distribution of student stress levels.
-   Check missing values, duplicate records, and potential outliers.
-   Clean and prepare data for modeling.
-   Engineer features related to study habits and well-being.
-   Train a Random Forest classifier.
-   Evaluate the model with accuracy, a classification report, a
    confusion matrix, and ROC curves.

## Dataset

The code expects a CSV file named `student_wellbeing.csv` with a target
column named `stress_level`. Features referenced in the code include: -
`sleep_hours` - `study_hours` - `screen_time` - `attendance` -
`anxiety_level` - `academic_pressure` - `social_support` -
`physical_activity`

**Important:** The dataset itself was not included in the provided PDF.
Add the dataset you used to the repository and update the filename in
the code if needed. Only publish data you have permission to share.

## Workflow

1.  Load the dataset with Pandas.
2.  Explore stress-level distribution, missing values, and outliers.
3.  Clean duplicate rows and handle missing values.
4.  Cap selected numerical values to reduce the influence of extreme
    values.
5.  Create features such as `screen_study_ratio`, `pressure_index`, and
    `health_score`.
6.  Encode the target labels.
7.  Split the data into training and test sets.
8.  Train a Random Forest classifier.
9.  Evaluate results with classification metrics, a confusion matrix,
    and ROC curves.
10. Save the trained model and label encoder as pickle files.

## Tools & Technologies

Python, Pandas, NumPy, Matplotlib, and Scikit-learn.

## How to Run

1.  Install Python.

2.  Place `student_wellbeing.csv` in the same directory as the Python
    script.

3.  Install dependencies:

    ``` bash
    python -m pip install pandas numpy matplotlib scikit-learn
    ```

4.  Run the script:

    ``` bash
    python your_script_name.py
    ```

Replace `your_script_name.py` with the actual name of your Python file.

## Expected Outputs

-   Data exploration visualizations.
-   Classification report and confusion matrix.
-   ROC curve plot(s).
-   Saved model files such as `model.pkl` and `encoder.pkl`.

## Project Status

This README summarizes the workflow visible in the provided project
materials. No verified model metrics were available in the PDF, so
performance values are intentionally not claimed here.

## Disclaimer

This project is for educational and portfolio purposes only. It is not a
clinical or psychological diagnostic tool.
