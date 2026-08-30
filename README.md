# CMP7005 Google Colab + Streamlit App

This folder keeps the full project in one place.

## Folder structure

```text
CMP7005_Colab_With_App/
│
├── notebook/
│   └── CMP7005_Google_Colab_Full_Workflow.ipynb
│
├── app.py
├── data/
│   ├── Loan_Approval_Data_Set_1.csv
│   └── Loan_Approval_Data_Set_2.csv
│
├── models/
│   ├── loan_approval_model.joblib
│   ├── kmeans_customer_segmentation.joblib
│   └── pca_privacy_model.joblib
│
├── output/
│   ├── cleaned_loan_data.csv
│   ├── customer_segments.csv
│   ├── cluster_profile.csv
│   ├── pca_components.csv
│   ├── metadata.json
│   └── model_metrics.json
│
├── requirements.txt
├── setup_windows.bat
└── run_app.bat
```

## How the two parts work

### Google Colab notebook

Use the notebook to show the academic/data-analysis workflow:

**raw CSV → preprocessing → merging → feature engineering → EDA → model development → K-Means → PCA**

### Streamlit app

Use the app as the software demonstration during your presentation.

The app contains:

- Dashboard
- Cleaned Data
- EDA
- Loan Prediction
- Customer Segmentation
- Model Performance

## Run the app on Windows

First time:

1. Extract the ZIP.
2. Double-click `setup_windows.bat`.

Then:

3. Double-click `run_app.bat`.

Your browser should open automatically.

You can also run:

```bash
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

## Presentation explanation

> I used one Google Colab notebook for the complete data-analysis and machine-learning workflow. The cleaned data and exported models are then connected to a simple Streamlit application. This separates the analytical work from the user-facing software while keeping both parts in the same project folder.

## Updated GUI style

The Streamlit application has been redesigned with:

- top navigation tabs instead of sidebar radio navigation;
- blue, green and white banking-style colours;
- a new header and dashboard cards;
- updated fonts and spacing;
- styled buttons, metrics, tables and form elements;
- a select box instead of a radio button for the prediction mode.

Run the app with:

```bash
python -m streamlit run app.py
```
