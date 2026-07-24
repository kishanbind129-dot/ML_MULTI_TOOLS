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

if 'last_request_time' not in st.session_state:
    st.session_state['last_request_time'] = 0.0

current_time = time.time()
if current_time - st.session_state['last_request_time'] < 0.5:
    st.error("Too many rapid requests. Please wait a moment.")
    st.stop()
st.session_state['last_request_time'] = current_time

if 'data' not in st.session_state:
    st.session_state['data'] = None

try:
    if os.path.exists("Multi_task/logo.png"):
        st.sidebar.image("Multi_task/logo.png", width=120)
    st.sidebar.title("🤖 AI Multi-Tools Engine")
except Exception:
    pass

menu = st.sidebar.selectbox(
    "Navigation Menu",
    [
        "🏠 Dashboard Home",
        "📈 Linear Regression", 
        "🎯 K-Means Clustering", 
        "🏷️ KNN Classification"
    ]
)

st.sidebar.markdown("---")
st.sidebar.subheader("⚙️ Global Configuration")
test_size_user = st.sidebar.slider("Testing Data Size (%)", min_value=10, max_value=50, value=20, step=5) / 100.0

if menu == "🏠 Dashboard Home":
    st.markdown("<h1 style='text-align: center; color: #4F46E5;'>Automated Advanced ML Engine</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center;'>Upload your dataset to initialize creative data analysis pipelines safely.</p>", unsafe_allow_html=True)
    st.markdown("---")
    
    uploaded_file = st.file_uploader("Drop your secure CSV file here", type=["csv"])
    
    if uploaded_file is not None:
        try:
            head_check = pd.read_csv(uploaded_file, nrows=5)
            uploaded_file.seek(0)
            st.session_state['data'] = pd.read_csv(uploaded_file)
            st.success("Dataset successfully authenticated and loaded into secure cache.")
        except Exception:
            st.error("Security Halt: Invalid or corrupt data structure inside the file.")
            st.session_state['data'] = None
            
    if st.session_state['data'] is not None:
        df = st.session_state['data']
        st.markdown("### 📊 Live Dataset Overview")
        c1, c2, c3 = st.columns(3)
        c1.metric("Total Rows", df.shape[0])
        c2.metric("Total Columns", df.shape[1])
        c3.metric("Missing Cells Detected", df.isna().sum().sum())
        st.dataframe(df.head(10), use_container_width=True)

elif menu == "📈 Linear Regression":
    st.markdown("<h2 style='color: #2563EB;'>📈 Linear Regression Pipeline</h2>", unsafe_allow_html=True)
    st.markdown("---")
    
    if st.session_state['data'] is None:
        st.warning("Please upload a safe CSV file from the 'Dashboard Home' section first.")
    else:
        try:
            df = st.session_state['data'].copy()
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            
            if len(numeric_cols) < 2:
                st.error("Dataset lacks sufficient continuous numeric variables for regression analysis.")
            else:
                y_col = st.selectbox("Select Target Variable (Y)", numeric_cols)
                x_cols = st.multiselect("Select Independent Features (X)", [c for c in numeric_cols if c != y_col])
                
                if st.button("Execute Linear Regression") and x_cols:
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
                        plt.xlabel(x_cols[0])
                        plt.ylabel(y_col)
                        plt.grid(True, linestyle='--', alpha=0.5)
                        plt.legend()
                        st.pyplot(fig)
        except Exception:
            st.error("Pipeline Interrupted: Failed to solve calculation array bounds securely.")

elif menu == "🎯 K-Means Clustering":
    st.markdown("<h2 style='color: #7C3AED;'>🎯 K-Means Clustering Core Engine</h2>", unsafe_allow_html=True)
    st.markdown("---")
    
    if st.session_state['data'] is None:
        st.warning("Please upload a safe CSV file from the 'Dashboard Home' section first.")
    else:
        try:
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
                k_choice = st.slider("Select Target Hyperparameter (K)", min_value=2, max_value=max_k, value=3)
                
                if st.button("Generate Partition Clusters"):
                    kmeans = KMeans(n_clusters=k_choice, random_state=42, n_init=10)
                    clusters = kmeans.fit_predict(X_scaled)
                    st.success(f"Mathematical structural optimization split completed into {k_choice} separate nodes.")
                    
                    fig_cluster, ax_cluster = plt.subplots(figsize=(10, 5))
                    sns.scatterplot(x=working_df.iloc[:, 0], y=working_df.iloc[:, 1], hue=clusters, palette='viridis', s=120, alpha=0.8)
                    plt.grid(True, linestyle='--', alpha=0.4)
                    st.pyplot(fig_cluster)
        except Exception:
            st.error("Pipeline Interrupted: Computational convergence boundary failure.")

elif menu == "🏷️ KNN Classification":
    st.markdown("<h2 style='color: #059669;'>🏷️ KNN Classification Core Pipeline</h2>", unsafe_allow_html=True)
    st.markdown("---")
    
    if st.session_state['data'] is None:
        st.warning("Please upload a safe CSV file from the 'Dashboard Home' section first.")
    else:
        try:
            df = st.session_state['data'].copy()
            all_cols = df.columns.tolist()
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            
            y_col = st.selectbox("Select Target Categorical Class (Y)", all_cols)
            x_cols = st.multiselect("Select Training Numerical Predictors (X)", [c for c in numeric_cols if c != y_col])
            
            k_neighbors = st.slider("Set Neighbors Concentration Bound (K)", min_value=1, max_value=15, value=5)
            
            if st.button("Execute Vectorized KNN Classifier") and x_cols:
                working_df = df[x_cols + [y_col]].dropna()
                X = working_df[x_cols]
                y = working_df[y_col]
                
                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(X)
                X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=test_size_user, random_state=0)
                
                knn = KNeighborsClassifier(n_neighbors=k_neighbors)
                knn.fit(X_train, y_train)
                preds = knn.predict(X_test)
                
                st.markdown("### 🏆 Prediction Quality Matrix")
                st.metric("Aggregate Evaluation Classification Accuracy", f"{accuracy_score(y_test, preds)*100:.2f}%")
                
                st.markdown("### 📝 Detailed Boundary Distribution Output")
                st.text_area("Vector Evaluation Metrics Matrix", str(classification_report(y_test, preds, output_dict=False)), height=200)
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

if 'last_request_time' not in st.session_state:
    st.session_state['last_request_time'] = 0.0

current_time = time.time()
if current_time - st.session_state['last_request_time'] < 0.5:
    st.error("Too many rapid requests. Please wait a moment.")
    st.stop()
st.session_state['last_request_time'] = current_time

if 'data' not in st.session_state:
    st.session_state['data'] = None

try:
    if os.path.exists("Multi_task/logo.png"):
        st.sidebar.image("Multi_task/logo.png", width=120)
    st.sidebar.title("🤖 AI Multi-Tools Engine")
except Exception:
    pass

menu = st.sidebar.selectbox(
    "Navigation Menu",
    [
        "🏠 Dashboard Home",
        "📈 Linear Regression", 
        "🎯 K-Means Clustering", 
        "🏷️ KNN Classification"
    ]
)

st.sidebar.markdown("---")
st.sidebar.subheader("⚙️ Global Configuration")
test_size_user = st.sidebar.slider("Testing Data Size (%)", min_value=10, max_value=50, value=20, step=5) / 100.0

if menu == "🏠 Dashboard Home":
    st.markdown("<h1 style='text-align: center; color: #4F46E5;'>Automated Advanced ML Engine</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center;'>Upload your dataset to initialize creative data analysis pipelines safely.</p>", unsafe_allow_html=True)
    st.markdown("---")
    
    uploaded_file = st.file_uploader("Drop your secure CSV file here", type=["csv"])
    
    if uploaded_file is not None:
        try:
            head_check = pd.read_csv(uploaded_file, nrows=5)
            uploaded_file.seek(0)
            st.session_state['data'] = pd.read_csv(uploaded_file)
            st.success("Dataset successfully authenticated and loaded into secure cache.")
        except Exception:
            st.error("Security Halt: Invalid or corrupt data structure inside the file.")
            st.session_state['data'] = None
            
    if st.session_state['data'] is not None:
        df = st.session_state['data']
        st.markdown("### 📊 Live Dataset Overview")
        c1, c2, c3 = st.columns(3)
        c1.metric("Total Rows", df.shape[0])
        c2.metric("Total Columns", df.shape[1])
        c3.metric("Missing Cells Detected", df.isna().sum().sum())
        st.dataframe(df.head(10), use_container_width=True)

elif menu == "📈 Linear Regression":
    st.markdown("<h2 style='color: #2563EB;'>📈 Linear Regression Pipeline</h2>", unsafe_allow_html=True)
    st.markdown("---")
    
    if st.session_state['data'] is None:
        st.warning("Please upload a safe CSV file from the 'Dashboard Home' section first.")
    else:
        try:
            df = st.session_state['data'].copy()
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            
            if len(numeric_cols) < 2:
                st.error("Dataset lacks sufficient continuous numeric variables for regression analysis.")
            else:
                y_col = st.selectbox("Select Target Variable (Y)", numeric_cols)
                x_cols = st.multiselect("Select Independent Features (X)", [c for c in numeric_cols if c != y_col])
                
                if st.button("Execute Linear Regression") and x_cols:
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
                        plt.xlabel(x_cols[0])
                        plt.ylabel(y_col)
                        plt.grid(True, linestyle='--', alpha=0.5)
                        plt.legend()
                        st.pyplot(fig)
        except Exception:
            st.error("Pipeline Interrupted: Failed to solve calculation array bounds securely.")

elif menu == "🎯 K-Means Clustering":
    st.markdown("<h2 style='color: #7C3AED;'>🎯 K-Means Clustering Core Engine</h2>", unsafe_allow_html=True)
    st.markdown("---")
    
    if st.session_state['data'] is None:
        st.warning("Please upload a safe CSV file from the 'Dashboard Home' section first.")
    else:
        try:
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
                k_choice = st.slider("Select Target Hyperparameter (K)", min_value=2, max_value=max_k, value=3)
                
                if st.button("Generate Partition Clusters"):
                    kmeans = KMeans(n_clusters=k_choice, random_state=42, n_init=10)
                    clusters = kmeans.fit_predict(X_scaled)
                    st.success(f"Mathematical structural optimization split completed into {k_choice} separate nodes.")
                    
                    fig_cluster, ax_cluster = plt.subplots(figsize=(10, 5))
                    sns.scatterplot(x=working_df.iloc[:, 0], y=working_df.iloc[:, 1], hue=clusters, palette='viridis', s=120, alpha=0.8)
                    plt.grid(True, linestyle='--', alpha=0.4)
                    st.pyplot(fig_cluster)
        except Exception:
            st.error("Pipeline Interrupted: Computational convergence boundary failure.")

elif menu == "🏷️ KNN Classification":
    st.markdown("<h2 style='color: #059669;'>🏷️ KNN Classification Core Pipeline</h2>", unsafe_allow_html=True)
    st.markdown("---")
    
    if st.session_state['data'] is None:
        st.warning("Please upload a safe CSV file from the 'Dashboard Home' section first.")
    else:
        try:
            df = st.session_state['data'].copy()
            all_cols = df.columns.tolist()
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            
            y_col = st.selectbox("Select Target Categorical Class (Y)", all_cols)
            x_cols = st.multiselect("Select Training Numerical Predictors (X)", [c for c in numeric_cols if c != y_col])
            
            k_neighbors = st.slider("Set Neighbors Concentration Bound (K)", min_value=1, max_value=15, value=5)
            
            if st.button("Execute Vectorized KNN Classifier") and x_cols:
                working_df = df[x_cols + [y_col]].dropna()
                X = working_df[x_cols]
                y = working_df[y_col]
                
                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(X)
                X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=test_size_user, random_state=0)
                
                knn = KNeighborsClassifier(n_neighbors=k_neighbors)
                knn.fit(X_train, y_train)
                preds = knn.predict(X_test)
                
                st.markdown("### 🏆 Prediction Quality Matrix")
                st.metric("Aggregate Evaluation Classification Accuracy", f"{accuracy_score(y_test, preds)*100:.2f}%")
                
                st.markdown("### 📝 Detailed Boundary Distribution Output")
                st.text_area("Vector Evaluation Metrics Matrix", str(classification_report(y_test, preds, output_dict=False)), height=200)
