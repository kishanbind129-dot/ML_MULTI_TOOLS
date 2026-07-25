import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, classification_report
from scipy.spatial.distance import cdist
import statsmodels.api as sm

# 🚧 LIVE NOTICE ENGINE (Set True when updating code, Set False when fully done)
UNDER_CONSTRUCTION = False

st.set_page_config(
    page_title="AI Multi-Tool Hub", 
    page_icon="Multi_task/logo.png", 
    layout="wide"
)

# Global Under Construction Intercept Filter
if UNDER_CONSTRUCTION:
    st.warning("🚧 **Notice:** System Maintenance Active. The application is currently under construction. Algorithms are temporarily offline.")
    st.stop()

if 'data' not in st.session_state:
    st.session_state['data'] = None

if os.path.exists("Multi_task/logo.png"):
    st.sidebar.image("Multi_task/logo.png", width=120)
st.sidebar.title("🤖 AI Multi-Tools Engine")

menu = st.sidebar.selectbox(
    "Navigation Menu",
    ["🏠 Dashboard Home", "📈 Linear Regression", "🎯 K-Means Clustering", "🏷️ KNN Core Engine"]
)

st.sidebar.markdown("---")
st.sidebar.subheader("⚙️ Global Configuration")
test_size_user = st.sidebar.slider("Testing Data Size (%)", min_value=10, max_value=50, value=20, step=5) / 100.0

if st.session_state['data'] is not None:
    df = st.session_state['data'].copy()
else:
    df = None

if menu == "🏠 Dashboard Home":
    st.markdown("<h1 style='text-align: center; color: #4F46E5;'>Automated Advanced ML Engine</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center;'>Upload your dataset to initialize creative data analysis pipelines safely.</p>", unsafe_allow_html=True)
    st.markdown("---")
    
    uploaded_file = st.file_uploader("Drop your secure CSV file here", type=["csv"])
    if uploaded_file is not None:
        st.session_state['data'] = pd.read_csv(uploaded_file)
        st.success("Dataset successfully authenticated and loaded into secure cache.")
            
    if st.session_state['data'] is not None:
        st.markdown("### 📊 Live Dataset Overview")
        c1, c2, c3 = st.columns(3)
        c1.metric("Total Rows", int(st.session_state['data'].shape[0]))
        c2.metric("Total Columns", int(st.session_state['data'].shape[1]))
        c3.metric("Missing Cells Detected", int(st.session_state['data'].isna().sum().sum()))
        st.dataframe(st.session_state['data'].head(10), use_container_width=True)

elif menu == "📈 Linear Regression":
    st.markdown("<h2 style='color: #2563EB;'>📈 Linear Regression Pipeline</h2>", unsafe_allow_html=True)
    st.markdown("---")
    if df is None:
        st.warning("Please upload a safe CSV file from the 'Dashboard Home' section first.")
    else:
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if len(numeric_cols) < 2:
            st.error("Dataset lacks sufficient continuous numeric variables for regression analysis.")
        else:
            y_col = st.selectbox("Select Target Variable (Y)", numeric_cols)
            x_cols = st.multiselect("Select Independent Features (X)", [c for c in numeric_cols if c != y_col])
            
            if x_cols:
                working_df = df[x_cols + [y_col]].dropna()
                X = np.array(working_df[x_cols], dtype=np.float64)
                y = working_df[y_col].tolist() # Plain list conversion to bypass PyArrow index bugs
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size_user, random_state=0)
                
                X_train_sm = sm.add_constant(X_train)
                X_test_sm = sm.add_constant(X_test, has_constant='add')
                ols_model = sm.OLS(y_train, X_train_sm).fit()
                preds = ols_model.predict(X_test_sm)
                
                if len(x_cols) == 1:
                    st.markdown("### 🔮 Live Interactive Prediction Calculator")
                    feature_name = x_cols[0]
                    intercept_val = ols_model.params[0]
                    coefficient_val = ols_model.params[1] if len(ols_model.params) > 1 else 0.0
                    
                    user_input_val = st.number_input(f"Set Custom Input Value for {feature_name}:", value=float(np.mean(X_test)))
                    predicted_outcome = (coefficient_val * user_input_val) + intercept_val
                    st.success(f"💡 **Statement:** Agar **{feature_name}** ki value **{user_input_val:.2f}** hogi, toh **{y_col}** ki estimated value **{predicted_outcome:.2f}** hogi!")
                
                if st.button("Execute Detailed Analysis"):
                    st.markdown("### 🏆 Performance Matrix")
                    col1, col2 = st.columns(2)
                    col1.metric("R² Prediction Accuracy", f"{r2_score(y_test, preds):.4f}")
                    col2.metric("Mean Squared Error (MSE)", f"{mean_squared_error(y_test, preds):.4f}")
                    
                    st.markdown("### 📋 Detailed OLS Regression Statistical Summary")
                    st.text_area("OLS Summary Report Table", str(ols_model.summary()), height=350)

elif menu == "🎯 K-Means Clustering":
    st.markdown("<h2 style='color: #7C3AED;'>🎯 K-Means Clustering Core Engine</h2>", unsafe_allow_html=True)
    st.markdown("---")
    if df is None:
        st.warning("Please upload a safe CSV file from the 'Dashboard Home' section first.")
    else:
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        x_cols = st.multiselect("Select Processing Features (Min 2)", numeric_cols)
        
        if x_cols and len(x_cols) >= 2:
            working_df = df[x_cols].dropna()
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(np.array(working_df, dtype=np.float64))
            
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
            st.pyplot(fig_elbow)
            
            k_choice = st.slider("Select Target Hyperparameter (K)", min_value=2, max_value=max_k, value=3)
            if st.button("Generate Partition Clusters"):
                kmeans = KMeans(n_clusters=k_choice, random_state=42, n_init=10)
                clusters = kmeans.fit_predict(X_scaled)
                working_df['Cluster_Output'] = clusters
                st.success(f"Mathematical structural optimization split completed into {k_choice} separate nodes.")

elif menu == "🏷️ KNN Core Engine":
    st.markdown("<h2 style='color: #059669;'>🏷️ KNN Predictive Modeling Engine</h2>", unsafe_allow_html=True)
    st.markdown("---")
    if df is None:
        st.warning("Please upload a safe CSV file from the 'Dashboard Home' section first.")
    else:
        all_cols = df.columns.tolist()
        y_col = st.selectbox("Select Target Column (Y)", all_cols)
        x_cols = st.multiselect("Select Predictor Features (X)", [c for c in all_cols if c != y_col])
        k_neighbors = st.slider("Set K-Neighbors Concentration Bound", min_value=1, max_value=15, value=5)
        
        if st.button("Run KNN Model Pipeline") and x_cols:
            working_df = df[x_cols + [y_col]].dropna()
            X_raw = working_df[x_cols].select_dtypes(include=[np.number])
            
            if X_raw.empty:
                st.error("Please ensure you select at least one numeric feature for training.")
            else:
                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(np.array(X_raw, dtype=np.float64))
                y_list = working_df[y_col].tolist() # 👈 Extracted to plain Python list to permanently avoid pyarrow index boundaries crash
                
                is_classification = (working_df[y_col].dtype == 'object') or (len(np.unique(y_list)) <= 10)
                X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_list, test_size=test_size_user, random_state=0)
                
                if is_classification:
                    st.info("ℹ️ Target variable detected as **Categorical**. Running **KNN Classification Pipeline**.")
                    knn = KNeighborsClassifier(n_neighbors=k_neighbors)
                    knn.fit(X_train, y_train)
                    preds = knn.predict(X_test)
                    st.metric("Model Classification Accuracy", f"{accuracy_score(y_test, preds)*100:.2f}%")
                    st.text_area("Detailed Report", str(classification_report(y_test, preds)))
                else:
                    st.info("ℹ️ Target variable detected as **Continuous Numbers**. Running **KNN Regression Pipeline**.")
                    knn = KNeighborsRegressor(n_neighbors=k_neighbors)
                    knn.fit(X_train, y_train)
                    preds = knn.predict(X_test)
                    st.metric("Model R² Prediction Score", f"{r2_score(y_test, preds):.4f}")
                    st.metric("Mean Squared Error (MSE)", f"{mean_squared_error(y_test, preds):.4f}")
