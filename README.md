# Indian IPO Analysis & Prediction Dashboard

This is a full-stack data science project that analyzes historical Indian IPO data to predict listing-day gains. The project uses a machine learning model to provide real-time predictions through an interactive web dashboard built with Streamlit.

Check out the live app here: https://ipopredictiondashboard.streamlit.app

## Purpose

The goal of this project is to help investors and analysts make data-driven decisions about upcoming IPOs. By analyzing factors like issue size and subscription rates (QIB, HNI, RII), the predictive model forecasts the "pop" or listing gain, offering a quantitative estimate of risk and reward.

## Features

Interactive Prediction: A sidebar allows users to input values for an IPO (Issue Size, QIB, HNI, and RII subscription rates) and receive an instant prediction for the listing gain.

## Exploratory Data Analysis (EDA): The main page features a scatter plot showing the historical relationship between Issue Size and Listing Gains.

Live Data Table: A filterable, sortable sample of the cleaned data used for the analysis.

Robust & Fast: The app uses Streamlit's caching features (@st.cache_resource, @st.cache_data) for high performance and includes robust error handling.

# Tech Stack

This project was built using the following technologies:

Language: Python

Data Analysis: Pandas

Machine Learning: Scikit-learn (RandomForestRegressor)

Data Visualization: Matplotlib, Seaborn

Web Framework & Deployment: Streamlit, Streamlit Cloud

Model Persistence: Joblib

# How to Run Locally

1. Prerequisites

Python: This project requires Python 3.10 or newer. Some dependencies may not be compatible with older versions.

Git: You will need Git installed to clone the repository.

2. Clone the repository

git clone [https://github.com/asifmddev/pythonprojekts.git](https://github.com/asifmddev/pythonprojekts.git)
cd pythonprojekts


3. Create a virtual environment

python -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`


4. Install the required libraries

It is highly recommended to install streamlit and joblib first, as they can sometimes run into dependency conflicts when bundled.

pip install streamlit joblib
pip install -r requirements.txt


5. Run the Streamlit app

streamlit run ipo_app.py


Your browser will automatically open to the app.

# Common Issues & Troubleshooting

This project involved several real-world deployment challenges. Here are the solutions to the most common issues:

Error (Streamlit Cloud): ModuleNotFoundError: No module named 'joblib'

Problem: The Streamlit Cloud environment doesn't have the joblib library installed.

Solution: This means the requirements.txt file was not pushed to GitHub or Streamlit is stuck on an old commit.

# Fix:

Ensure joblib is listed in your requirements.txt file.

Push the file to your GitHub repository (git add requirements.txt, git commit ..., git push).

If the error persists, force a redeploy by adding a new comment to ipo_app.py, saving, and pushing the change. This forces Streamlit to do a fresh clone of your repository.

As a last resort, delete and re-deploy the app on Streamlit Cloud.

Error (Local or Cloud): App hangs with a blank screen.

Problem: This is a matplotlib backend issue. Streamlit can't render the chart using the default interactive backend.

Solution: We must force matplotlib to use a non-interactive, web-safe backend.

Fix: Add these three lines to the top of your ipo_app.py file, before importing matplotlib.pyplot:

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


Error (Git): failed to push some refs... (fetch first)

Problem: The GitHub repository contains work (like a README.md file) that you don't have locally. You can't overwrite it.

Solution: You must pull the remote changes and combine the histories.

Fix: Run git pull origin main --rebase --allow-unrelated-histories before trying to push again.

Error (Git): Password authentication is not supported

Problem: GitHub no longer accepts passwords for command-line operations.

Solution: You must use a Personal Access Token (PAT).

Fix:

Go to your GitHub Settings > Developer settings > Personal access tokens (classic).

Generate a new token with the repo scope.

Copy the token (ghp_...).

When you run git push, use your username and paste the token as your password.
