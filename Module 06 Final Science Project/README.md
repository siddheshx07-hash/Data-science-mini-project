# Wine Quality Prediction Using Machine Learning

## Project overview
This end-to-end Data Science project analyzes the UCI red wine quality dataset and trains a Random Forest Regressor to estimate wine quality from physicochemical measurements.

## Project objectives
1. Load and inspect the dataset.
2. Check missing values and remove duplicate rows.
3. Perform exploratory data analysis (EDA).
4. Visualize distributions and feature correlations.
5. Train a Random Forest Regression model.
6. Evaluate it using MAE, RMSE, and R².

## Tools and technologies
- Python
- Pandas and NumPy
- Matplotlib and Seaborn
- Scikit-learn
- Jupyter Notebook / Google Colab

## Files
- `Final_Data_Science_Project.ipynb` — complete, step-by-step notebook.
- `wine_quality_analysis.py` — runnable Python script version.
- `requirements.txt` — required Python packages.

## How to run in Google Colab
1. Open https://colab.research.google.com/
2. Choose **File → Upload notebook**.
3. Upload `Final_Data_Science_Project.ipynb`.
4. Choose **Runtime → Run all**.
5. The notebook downloads the dataset from UCI, so an internet connection is required.

## How to run locally
```bash
pip install -r requirements.txt
python wine_quality_analysis.py
```

## Dataset
UCI Machine Learning Repository — Wine Quality:
https://archive.ics.uci.edu/dataset/186/wine+quality

The notebook reads the red wine CSV directly from the UCI public dataset URL. Internet access is required.

## Model and evaluation
- Model: Random Forest Regressor
- Metrics: Mean Absolute Error (MAE), Root Mean Squared Error (RMSE), and R²

The actual metric values are generated when the notebook or script is executed; they are intentionally not pre-filled.

## Important note
This is an educational project. Wine quality ratings are subjective, and model predictions should not be treated as professional wine assessments.
