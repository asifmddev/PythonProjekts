# Forcing. a redeploy
import streamlit as st
import pandas as pd
import joblib
import seaborn as sns

# --- Matplotlib Fix ---
import matplotlib 
matplotlib.use('Agg') # <--- THIS IS THE FIX
import matplotlib.pyplot as plt
# ----------------------

import os # Good, keep this

# --- File Paths ---
MODEL_PATH = 'ipo_model.pkl'
DATA_PATH = 'ipo_cleaned.csv'

# --- Caching Functions (Combining Your Logic + Caching) ---

@st.cache_resource  # Caches the model
def load_model():
    """Loads the model, with error handling."""
    try:
        model = joblib.load(MODEL_PATH)
        return model
    except FileNotFoundError:
        st.error(f"Error: Model file not found at {MODEL_PATH}. Make sure 'ipo_model.pkl' is in the same directory.")
        st.stop()
    except Exception as e:
        st.error(f"An error occurred loading the model: {e}")
        st.stop()

@st.cache_data  # Caches the dataframe
def load_data():
    """Loads the data, with error handling and column checks."""
    try:
        df = pd.read_csv(DATA_PATH)
        
        # Your robust column check
        required_cols = ['Issue_Size_Cr', 'Listing_Gain_%', 'QIB', 'HNI', 'RII']
        if not all(col in df.columns for col in required_cols):
            missing = [col for col in required_cols if col not in df.columns]
            st.error(f"Error: The loaded CSV is missing required columns: {', '.join(missing)}")
            st.stop()

        # Your robust type conversion
        if 'Issue_Size_Cr' in df.columns:
             df['Issue_Size_Cr'] = pd.to_numeric(df['Issue_Size_Cr'], errors='coerce')
             df.dropna(subset=['Issue_Size_Cr'], inplace=True)

        return df
        
    except FileNotFoundError:
        st.error(f"Error: Data file not found at {DATA_PATH}. Make sure 'ipo_cleaned.csv' is in the same directory.")
        st.stop()
    except Exception as e:
        st.error(f"An error occurred loading the data: {e}")
        st.stop()

# --- Load data and model using the cached functions ---
model = load_model()
df = load_data()

# --- App Layout ---
# (The rest of your code is perfect and remains unchanged)

st.set_page_config(layout="wide")
st.title('📈 Indian IPO Analysis & Listing Gain Predictor')

# --- Sidebar for Prediction ---
st.sidebar.header('🔮 Predict Listing Gains')
st.sidebar.write('Tune these parameters to see the predicted gain.')

# Your robust slider logic
if not df.empty and 'Issue_Size_Cr' in df.columns:
    issue_size_min = float(df['Issue_Size_Cr'].min())
    issue_size_max = float(df['Issue_Size_Cr'].max())
    issue_size_mean = float(df['Issue_Size_Cr'].mean())
    issue_size_mean = max(issue_size_min, min(issue_size_mean, issue_size_max))
    issue_size = st.sidebar.slider(
        'Issue Size (Crores)',
        issue_size_min,
        issue_size_max,
        issue_size_mean
    )
else:
    st.sidebar.warning("Could not load Issue Size data for slider.")
    issue_size = st.sidebar.number_input('Issue Size (Crores)', value=1000.0)

# Your robust defaults
qib_default, qib_max = (50.0, 250.0) if 'QIB' in df.columns else (50.0, 250.0)
hni_default, hni_max = (100.0, 400.0) if 'HNI' in df.columns else (100.0, 400.0)
rii_default, rii_max = (30.0, 100.0) if 'RII' in df.columns else (30.0, 100.0)

qib = st.sidebar.slider('QIB Subscription (times)', 0.0, qib_max, qib_default)
hni = st.sidebar.slider('HNI Subscription (times)', 0.0, hni_max, hni_default)
rii = st.sidebar.slider('RII Subscription (times)', 0.0, rii_max, rii_default)

# Your robust prediction logic
try:
    feature_names = ['Issue_Size_Cr', 'QIB', 'HNI', 'RII']
    input_df = pd.DataFrame([[issue_size, qib, hni, rii]], columns=feature_names)
    prediction = model.predict(input_df)
    st.sidebar.subheader(f'Predicted Listing Gain: **{prediction[0]:.2f}%**')
except Exception as e:
    st.sidebar.error(f"Could not make prediction: {e}")

# --- Main Page for Analysis ---
st.header('IPO Data Analysis')

# Your robust plotting logic
if not df.empty and 'Issue_Size_Cr' in df.columns and 'Listing_Gain_%' in df.columns:
    st.subheader('Listing Gains vs. Issue Size')
    fig, ax = plt.subplots()
    try:
        sns.scatterplot(data=df, x='Issue_Size_Cr', y='Listing_Gain_%', alpha=0.5, ax=ax)
        ax.set_xlabel('Issue Size (Crores)')
        ax.set_ylabel('Listing Gain %')
        st.pyplot(fig)
    except Exception as e:
        st.warning(f"Could not plot chart: {e}")
else:
    st.warning("Insufficient data to plot Listing Gains vs. Issue Size.")

# Your robust data display
st.subheader('Cleaned IPO Data Sample (Last 10 Rows)')
if not df.empty:
    st.dataframe(df.tail(10))
else:
    st.warning("Could not display data table.")