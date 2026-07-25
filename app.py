import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import time

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, classification_report
from scipy.spatial.distance import cdist

st.set_page_config(
    page_title="AI Multi-Tool Hub", 
    page_icon="Multi_task/logo.png", 
    layout="wide"
)

# -----------------------------------------------------------------------------
# 🔥 FUTURE-PROOF AUTOMATIC UPGRADE DETECTOR (भविष्य के लिए पूरी तरह ऑटोमैटिक)
# -----------------------------------------------------------------------------
# जब भी आप भविष्य में कोई नया फीचर जोड़कर फाइल सेव करेंगे, यह अपने आप एक्टिव हो जाएगा।
UPGRADE_COOL_DOWN_MINUTES = 3  # कोड बदलने के बाद ऐप कितनी देर लॉक रहेगा

try:
    # अपनी मुख्य कोडिंग फाइल का नाम यहाँ सुनिश्चित करें
    current_file = "app.py" 
    
    if os.path.exists(current_file):
        file_last_modified = os.path.getmtime(current_file)
        server_current_time = time.time()
        diff_seconds = server_current_time - file_last_modified
        diff_minutes = diff_seconds / 60
        
        # अगर फाइल को सेव किए हुए तय मिनट से कम समय हुआ है, तो ऑटो-लॉक स्क्रीन दिखेगी
        if diff_minutes < UPGRADE_COOL_DOWN_MINUTES:
            time_left_seconds = int((UPGRADE_COOL_DOWN_MINUTES * 60) - diff_seconds)
            
            st.markdown("<h1 style='text-align: center; color: #F59E0B;'>🚧 App is Upgrading & Modifying 🚧</h1>", unsafe_allow_html=True)
            st.markdown("<p style='text-align: center; font-size: 20px;'>The developer is currently deploying new features or upgrading algorithms. The infrastructure is syncing live modifications safely.</p>", unsafe_allow_html=True)
            
            # लाइव सेकंड्स रिवर्स टाइमर जो स्क्रीन पर घटेगा
            st.warning(f"🔄 Real-time compilation active. Auto-unlocking workspace in {time_left_seconds} seconds...")
            st.progress(int((diff_seconds / (UPGRADE_COOL_DOWN_MINUTES * 60)) * 100))
            
            st.image("https://giphy.com", use_container_width=True)
            st.stop() # कोड को यहीं रोक देगा ताकि यूजर्स को क्रैश ऐप न दिखे
except Exception:
    pass # बैकअप के लिए, ताकि किसी वजह से फाइल पाथ मिस होने पर ऐप बंद न हो

# --- एंटी-फ्लड रिक्वेस्ट कंट्रोल ---
if 'last_request_time' not in st.session_state:
    st.session_state['last_request_time'] = 0.0

current_time = time.time()
if current_time - st.session_state['last_request_time'] < 0.5:
    st.error("Too many rapid requests. Please wait a moment.")
    st.stop()
st.session_state['last_request_time'] = current_time

if 'data' not in st.session_state:
    st.session_state['data'] = None

if os.path.exists("Multi_task/logo.png"):
    st.sidebar.image("Multi_task/logo.png", width=120)
st.sidebar.title("🤖 AI Multi-Tools Engine")

# साइडबार में बैकअप के लिए मैन्युअल डेवलपर लॉक
st.sidebar.subheader("🛠️ Developer Control")
under_construction_mode = st.sidebar.checkbox("Force Under-Construction State", value=False)

if under_construction_mode:
    st.markdown("<h1 style='text-align: center; color: #EF4444;'>🚧 System Under Construction 🚧</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; font-size: 20px;'>The developer is applying manual live changes. Please hold on!</p>", unsafe_allow_html=True)
    st.stop()

menu = st.sidebar.selectbox(
    "Navigation Menu",
    [
        "🏠 Dashboard Home",
        "📈 Linear Regression", 
        "🎯 K-Means Clustering", 
        "🏷️ KNN Classification",
        "🚧 Advanced Tools (Under Construction Page)"
    ]
)

st.sidebar.markdown("---")
st.sidebar.subheader("⚙️ Global Configuration")
test_size_user = st.sidebar.slider("Testing Data Size (%)", min_value=10, max_value=50, value=20, step=5) / 100.0

# --- DASHBOARD HOME ---
if menu == "🏠 Dashboard Home":
    st.markdown("<h1 style='text-align: center; color: #4F46E5;'>Automated Advanced ML Engine</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center;'>Upload your dataset to initialize creative data analysis pipelines safely.</p>", unsafe_allow_html=True)
    st.markdown("---")
    
    uploaded_file = st.file_uploader("Drop your secure CSV file here", type=["csv"])
    
    if uploaded_file is not None:
        with st.spinner("🔄 Authenticating and caching dataset..."):
            time.sleep(0.5)
            st.session_state['data'] = pd.read_csv(uploaded_file)
        st.success("Dataset successfully authenticated and loaded into secure cache.")
            
    if st.session_state['data'] is not None:
        df = st.session_state['data']
        st.markdown("### 📊 Live Dataset Overview")
        c1, c2, c3 = st.columns(3)
        # ✅ FIX: .shape को अलग इंडेक्स देकर संख्या निकाली ताकि क्रैश न हो
        c1.metric("Total Rows", int(df.shape[0]))
        c2.metric("Total Columns", int(df.shape[1]))
        c3.metric("Missing Cells Detected", int(df.isna().sum().sum()))
        st.dataframe(df.head(10), use_container_width=True)

# --- LINEAR REGRESSION ---
elif menu == "📈 Linear Regression":
    st.markdown("<h2 style='color: #2563EB;'>📈 Linear Regression Pipeline</h2>", unsafe_allow_html=True)
    st.markdown("---")
    
    if st.session_state['data'] is None:
        st.warning("Please upload a safe CSV file from the 'Dashboard Home' section first.")
    else:
        df = st.session_state['data'].copy()
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        if len(numeric_cols) < 2:
            st.error("Dataset lacks sufficient continuous numeric variables for regression analysis.")
        else:
            y_col = st.selectbox("Select Target Variable (Y)", numeric_cols)
            x_cols = st.multiselect("Select Independent Features (X)", [c for c in numeric_cols if c != y_col])
            
            if st.button("Execute Linear Regression") and x_cols:
                with st.spinner("⚙️ Architecture is Under Construction / Processing... Please Wait."):
                    time.sleep(0.5)
                    working_df = df[x_cols + [y_col]].dropna()
                    X = working_df[x_cols]
                    y = working_df[y_col]
                    
                    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size_user, random_state=0)
                    
                    model = LinearRegression()
                    model.fit(X_train, y_train)
                    preds = model.predict(X_test)
                    
                st.markdown("### 🏆 Performance Matrix")
                col1, col2 = st.columns(2)
                col1.metric("R² Prediction Accuracy", f"{r2_score(y_test, preds):.4f}")
                col2.metric("Mean Squared Error (MSE)", f"{mean_squared_error(y_test, preds):.4f}")
                
                if len(x_cols) == 1:
                    st.markdown("### 📊 Continuous Fitting Trend")
                    fig, ax = plt.subplots(figsize=(10, 5))
                    plt.scatter(X_test, y_test, color='#2563EB', alpha=0.7, label='Actual Values')
                    plt.plot(X_test, preds, color='#EF4444', linewidth=3, label='Optimal Fit Line')
                    plt.xlabel(str(x_cols[0]))
                    plt.ylabel(str(y_col))
                    plt.grid(True, linestyle='--', alpha=0.5)
                    plt.legend()
                    st.pyplot(fig)
        
        st.markdown("---")
        st.markdown("### 💡 Trend Summary & Business Logic")
        st.info("**Ice Cream Sales Trend Example:** Linear Regression helps find a direct link between variables. "
                "For instance, **जैसे-जैसे Temperature (तापमान) बढ़ेगा, Ice Cream की Sales (बिक्री) भी उसी अनुपात में लगातार बढ़ेगी**।")

# --- K-MEANS CLUSTERING ---
elif menu == "🎯 K-Means Clustering":
    st.markdown("<h2 style='color: #7C3AED;'>🎯 K-Means Clustering Core Engine</h2>", unsafe_allow_html=True)
    st.markdown("---")
    
    if st.session_state['data'] is None:
        st.warning("Please upload a safe CSV file from the 'Dashboard Home' section first.")
    else:
        df = st.session_state['data'].copy()
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        x_cols = st.multiselect("Select Processing Features (Min 2)", numeric_cols)
        
        if x_cols and len(x_cols) >= 2:
            working_df = df[x_cols].dropna()
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(working_df)
            
            st.markdown("### 📊 Step 1: Evaluating Distortion Curves")
            distortions = []
            max_k = min(10, len(working_df))
            K_range = range(1, max_k + 1)
            
            for k in K_range:
                kmeanModel = KMeans(n_clusters=k, random_state=42, n_init=10).fit(X_scaled)
                distortions.append(sum(np.min(cdist(X_scaled, kmeanModel.cluster_centers_, 'euclidean'), axis=1)) / X_scaled.shape[0])
                
            fig_elbow, ax_elbow = plt.subplots(figsize=(10, 4))
            plt.plot(K_range, distortions, 'bx-', color='#7C3AED', marker='o', linewidth=2)
            plt.xlabel('Number of Clusters (K)')
            plt.ylabel('Calculated Structural Distortion')
            plt.title('The Elbow Method Optimization Interface')
            plt.grid(True, linestyle='--', alpha=0.5)
            st.pyplot(fig_elbow)
            
            st.markdown("### ⚙️ Step 2: Final Cluster Optimization")
