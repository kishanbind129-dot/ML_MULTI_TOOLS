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

if os.path.exists("Multi_task/logo.png"):
    st.sidebar.image("Multi_task/logo.png", width=120)
st.sidebar.title("🤖 AI Multi-Tools Engine")

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
        MAX_BIG_FILE_SIZE = 300 * 1024 * 1024  # 300MB upper limit
        if uploaded_file.size > MAX_BIG_FILE_SIZE:
            st.error("Security Threat Alert: Upload blocked. File exceeds the 300MB safety limit.")
            st.stop()
        
        try:
            # Chunking process to read huge files up to 100k rows without system crash
            chunks = []
            for chunk in pd.read_csv(uploaded_file, chunksize=20000):
                chunks.append(chunk)
                if len(chunks) * 20000 >= 100000:
                    break
            
            raw_df = pd.concat(chunks, ignore_index=True)
            if raw_df.empty:
                st.error("Invalid Target: File contains no readable structural elements.")
                st.stop()
                
            sanitized_cols = [str(col).replace('{','').replace('}','').replace('[','').replace(']','').replace('<','').replace('>','')[:32] for col in raw_df.columns]
            raw_df.columns = sanitized_cols
            
            st.session_state['data'] = raw_df
            st.success("Large Dataset authenticated, optimized via chunking partitions, and loaded successfully.")
        except Exception:
            st.error("Parser Error: Malformed file data engine rejected.")
            st.stop()
            
    if st.session_state['data'] is not None:
        df = st.session_state['data']
        st.markdown("### 📊 Live Dataset Overview")
        c1, c2, c3 = st.columns(3)
        c1.metric("Total Rows", df.shape[0])
        c2.metric("Total Columns", df.shape[1])
        c3.metric("Missing Cells Detected", df.isna().sum().sum())
        st.dataframe(df.head(10), use_container_width=True)
        st.markdown("---")
        st.markdown("### 🔍 Live Data Quality Audit Report")
        col_q1, col_q2 = st.columns(2)
        
        with col_q1:
            st.markdown("**🚨 Columns with Missing Values:**")
            missing_info = df.isna().sum()
            missing_cols = missing_info[missing_info > 0]
            if not missing_cols.empty:
                st.dataframe(pd.DataFrame({"Missing Count": missing_cols}), use_container_width=True)
            else:
                st.success("Perfect Health! No missing values found in any column.")
                
        with col_q2:
            st.markdown("**📂 Row Analytics & Structural Integrity:**")
            duplicate_count = int(df.duplicated().sum())
            if duplicate_count > 0:
                st.warning(f"Detected {duplicate_count} exact duplicate rows. Consider cleaning your data source.")
            else:
                st.success("No duplicate rows found. Data variance is clean.")
            
            # Numeric vs Categorical column distributions
            num_count = len(df.select_dtypes(include=[np.number]).columns)
            cat_count = len(df.select_dtypes(exclude=[np.number]).columns)
            st.info(f"Feature Breakdown: {num_count} Numerical columns | {cat_count} Categorical columns")
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
            
            if x_cols:
                working_df = df[x_cols + [y_col]].dropna()
                X = working_df[x_cols]
                y = working_df[y_col]
                
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size_user, random_state=0)
                
                model = LinearRegression()
                model.fit(X_train, y_train)
                preds = model.predict(X_test)
                
                # 🟢 STABLE TABS STRUCTURAL FIX: Separated execution states to prevent input data flush
                tab1, tab2 = st.tabs(["📊 Model Performance & Analytics", "🔮 Live Value Trend Predictor"])
                
                with tab1:
                    st.markdown("### 🏆 Performance Matrix")
                    col1, col2 = st.columns(2)
                    col1.metric("R² Prediction Accuracy", f"{r2_score(y_test, preds):.4f}")
                    col2.metric("Mean Squared Error (MSE)", f"{mean_squared_error(y_test, preds):.4f}")
                    
                    if len(x_cols) == 1:
                        st.markdown("### 📊 Continuous Fitting Trend")
                        fig, ax = plt.subplots(figsize=(10, 5))
                        plt.scatter(X_test.iloc[:, 0], y_test, color='#2563EB', alpha=0.7, label='Actual Values')
                        plt.plot(X_test.iloc[:, 0], preds, color='#EF4444', linewidth=3, label='Optimal Fit Line')
                        plt.xlabel(x_cols[0])
                        plt.ylabel(y_col)
                        plt.grid(True, linestyle='--', alpha=0.5)
                        plt.legend()
                        st.pyplot(fig)

                    if len(x_cols) > 1:
                        st.markdown("### 📊 Driver Analysis (Feature Importance)")
                        fig_imp, ax_imp = plt.subplots(figsize=(10, max(3, len(x_cols) * 0.6)))
                        importance_df = pd.DataFrame({
                            'Feature': x_cols,
                            'Impact Score (Coefficient)': model.coef_
                        }).sort_values(by='Impact Score (Coefficient)', key=abs, ascending=True)
                        
                        colors = ['#EF4444' if x < 0 else '#2563EB' for x in importance_df['Impact Score (Coefficient)']]
                        plt.barh(importance_df['Feature'], importance_df['Impact Score (Coefficient)'], color=colors, alpha=0.85)
                        plt.axvline(0, color='black', linestyle='--', alpha=0.5)
                        plt.title("Which feature changes impact the trend line most? (Blue=Positive, Red=Negative)", fontsize=10)
                        plt.grid(True, linestyle='--', alpha=0.3)
                        st.pyplot(fig_imp)

                with tab2:
                    st.markdown("### 🔮 Live Interactive Trend Predictor")
                    st.info("Enter values for features below to see the calculated live trend output instantly.")
                    
                    user_inputs = {}
                    for col in x_cols:
                        min_val = float(X[col].min())
                        max_val = float(X[col].max())
                        mean_val = float(X[col].mean())
                        user_inputs[col] = st.number_input(f"Enter Value for '{col}' (Range: {min_val:.2f} - {max_val:.2f})", value=mean_val)
                    
                    if st.button("Predict Target Value", key="lr_live_predict_btn"):
                        input_df = pd.DataFrame([user_inputs])
                        
                        if input_df.isin([np.inf, -np.inf, np.nan]).any().any() or (input_df.abs() > 1e12).any().any():
                            st.error("Malicious Input Detected: Non-finite or extreme overflow numbers blocked.")
                        else:
                            predicted_val = model.predict(input_df)
                            st.success(f"🎯 Estimated **{y_col}** value will be: **{predicted_val[0]:.4f}**")


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
            k_choice = st.slider("Select Target Hyperparameter (K)", min_value=2, max_value=max_k, value=3)
            
            if st.button("Generate Partition Clusters"):
                kmeanModel = KMeans(n_clusters=k_choice, random_state=42, n_init=10)
                clusters = kmeanModel.fit_predict(X_scaled)
                st.success(f"Mathematical structural optimization split completed into {k_choice} separate nodes.")
                
                fig_cluster, ax_cluster = plt.subplots(figsize=(10, 5))
                sns.scatterplot(x=working_df.iloc[:, 0], y=working_df.iloc[:, 1], hue=clusters, palette='viridis', s=120, alpha=0.8)
                plt.grid(True, linestyle='--', alpha=0.4)
                st.pyplot(fig_cluster)
                st.markdown("---")
            st.markdown("### 🔮 Live Data Node Cluster Identifier")
            
            user_inputs_cluster = {}
            for col in x_cols:
                min_val = float(working_df[col].min())
                max_val = float(working_df[col].max())
                mean_val = float(working_df[col].mean())
                user_inputs_cluster[col] = st.number_input(f"Enter Value for '{col}'", value=mean_val, key=f"cluster_{col}")
            
            if st.button("Identify Data Cluster"):
                input_df = pd.DataFrame([user_inputs_cluster])
                input_scaled = scaler.transform(input_df)
                predicted_cluster = kmeanModel.predict(input_scaled)[0]
                st.info(f"This specific data trend falls into **Cluster Group / Node: {predicted_cluster}**")
elif menu == "🏷️ KNN Classification":
    st.markdown("<h2 style='color: #059669;'>🏷️ KNN Classification Core Pipeline</h2>", unsafe_allow_html=True)
    st.markdown("---")
    
    if st.session_state['data'] is None:
        st.warning("Please upload a safe CSV file from the 'Dashboard Home' section first.")
    else:
        df = st.session_state['data'].copy()
        all_cols = df.columns.tolist()
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        y_col = st.selectbox("Select Target Categorical Class (Y)", all_cols)
        x_cols = st.multiselect("Select Training Numerical Predictors (X)", [c for c in numeric_cols if c != y_col])
        
        k_neighbors = st.slider("Set Neighbors Concentration Bound (K)", min_value=1, max_value=15, value=5)
        
        if x_cols:
            working_df = df[x_cols + [y_col]].dropna()
            X = working_df[x_cols]
            y = working_df[y_col]
            
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size_user, random_state=0)
            
            scaler_knn = StandardScaler()
            X_train_scaled = scaler_knn.fit_transform(X_train)
            X_test_scaled = scaler_knn.transform(X_test)
            
            model_knn = KNeighborsClassifier(n_neighbors=k_neighbors)
            model_knn.fit(X_train_scaled, y_train)
            preds = model_knn.predict(X_test_scaled)
            
            if st.button("Execute Vectorized KNN Classifier"):
                st.markdown("### 🏆 Performance Matrix")
                st.metric("Model Accuracy Score", f"{accuracy_score(y_test, preds):.4f}")
                
                st.markdown("### 📝 Detailed Classification Report")
                report_dict = classification_report(y_test, preds, output_dict=True)
                report_df = pd.DataFrame(report_dict).transpose()
                st.dataframe(report_df, use_container_width=True)

            # NAYA FEATURE: Interactive Category Class Predictor
            st.markdown("---")
            st.markdown("### 🔮 Live Categorical Class Predictor")
            
            user_inputs_knn = {}
            for col in x_cols:
                min_val = float(X[col].min())
                max_val = float(X[col].max())
                mean_val = float(X[col].mean())
                user_inputs_knn[col] = st.number_input(f"Enter Value for '{col}'", value=mean_val, key=f"knn_{col}")
            
            if st.button("Predict Target Category"):
                input_df = pd.DataFrame([user_inputs_knn])
                input_scaled = scaler_knn.transform(input_df)
                predicted_class = model_knn.predict(input_scaled)[0]
                st.success(f"Predicted **{y_col}** Category is: **{predicted_class}**")

