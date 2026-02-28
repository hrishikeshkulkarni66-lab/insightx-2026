import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score

st.set_page_config(page_title="INSIGHTX 2026", layout="wide")

# ===============================
# 🌌 FUTURISTIC 3D BACKGROUND
# ===============================

st.markdown("""
<style>

body {
    background: black;
    overflow-x: hidden;
}

#particles-js {
    position: fixed;
    width: 100%;
    height: 100%;
    background-color: #000;
    z-index: -1;
}

.title {
    font-size: 70px;
    font-weight: bold;
    text-align: center;
    background: linear-gradient(90deg, #00ffff, #ff00ff);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    animation: glow 2s ease-in-out infinite alternate;
}

@keyframes glow {
    from { text-shadow: 0 0 20px #00ffff; }
    to { text-shadow: 0 0 40px #ff00ff; }
}

.glass {
    background: rgba(255,255,255,0.08);
    backdrop-filter: blur(15px);
    padding: 30px;
    border-radius: 20px;
    box-shadow: 0 0 30px rgba(0,255,255,0.4);
    margin-bottom: 30px;
}

.sidebar .sidebar-content {
    background: #0f0f0f;
}

</style>

<div id="particles-js"></div>

<script src="https://cdn.jsdelivr.net/particles.js/2.0.0/particles.min.js"></script>
<script>
particlesJS("particles-js", {
  "particles": {
    "number": {"value": 100},
    "color": {"value": "#00ffff"},
    "shape": {"type": "circle"},
    "opacity": {"value": 0.5},
    "size": {"value": 3},
    "line_linked": {"enable": true, "distance": 150, "color": "#00ffff"},
    "move": {"enable": true, "speed": 3}
  }
});
</script>

""", unsafe_allow_html=True)

st.markdown('<div class="title">🚀 INSIGHTX 2026</div>', unsafe_allow_html=True)
st.markdown('<div style="text-align:center; color:white;">Smart Student Performance Prediction & Academic Advisory System</div>', unsafe_allow_html=True)

# ===============================
# SIDEBAR NAVIGATION
# ===============================

st.sidebar.title("🧭 Navigation")
page = st.sidebar.radio("Go to", ["🏠 Home", "📊 Data", "🤖 Prediction", "📈 Visualization", "📉 Performance"])

# ===============================
# SYNTHETIC DATA GENERATOR
# ===============================

def generate_data(n=1000):
    df = pd.DataFrame({
        "Attendance": np.random.randint(50, 100, n),
        "StudyHours": np.random.randint(1, 10, n),
        "AssignmentScore": np.random.randint(40, 100, n),
        "PreviousGPA": np.round(np.random.uniform(5, 10, n),2),
        "Participation": np.random.randint(1, 10, n),
        "InternetUsage": np.random.randint(1, 10, n),
        "SleepHours": np.random.randint(4, 9, n),
        "FamilySupport": np.random.randint(1, 10, n),
        "ExtraCurricular": np.random.randint(0, 5, n)
    })

    df["FinalScore"] = (
        df["Attendance"]*0.2 +
        df["StudyHours"]*5 +
        df["AssignmentScore"]*0.3 +
        df["PreviousGPA"]*5 -
        df["InternetUsage"]*1.5 +
        df["SleepHours"]*2
    )

    df["PassFail"] = np.where(df["FinalScore"]>=50,1,0)
    df["PerformanceCategory"] = pd.cut(df["FinalScore"],
                                       bins=[0,50,70,100],
                                       labels=["Low","Medium","High"])
    return df

if "data" not in st.session_state:
    st.session_state.data = None

# ===============================
# HOME PAGE
# ===============================

if page == "🏠 Home":
    st.markdown('<div class="glass">', unsafe_allow_html=True)
    st.header("📌 Project Description")
    st.write("""
    This intelligent system uses:
    - 🔹 Linear Regression → Final Score Prediction
    - 🔹 Decision Tree → Pass/Fail Classification
    - 🔹 KNN → Performance Category Prediction
    - 🔹 K-Means → Academic Risk Clustering
    """)
    st.markdown("### 👨‍💻 Team Members")
    st.write("""
    - Name 1 – USN1  
    - Name 2 – USN2  
    - Name 3 – USN3  
    - Name 4 – USN4  
    """)
    st.markdown('</div>', unsafe_allow_html=True)

# ===============================
# DATA PAGE
# ===============================

elif page == "📊 Data":
    st.markdown('<div class="glass">', unsafe_allow_html=True)
    option = st.radio("Choose Data Option", ["Generate Synthetic Data", "Upload CSV"])

    if option == "Generate Synthetic Data":
        n = st.slider("Number of Records", 1000, 5000, 1000)
        st.session_state.data = generate_data(n)
        st.success("Data Generated Successfully!")
        st.dataframe(st.session_state.data.head())

    else:
        file = st.file_uploader("Upload CSV", type=["csv"])
        if file:
            st.session_state.data = pd.read_csv(file)
            st.success("CSV Uploaded Successfully!")
            st.dataframe(st.session_state.data.head())

    st.markdown('</div>', unsafe_allow_html=True)

# ===============================
# PREDICTION
# ===============================

elif page == "🤖 Prediction":
    if st.session_state.data is None:
        st.warning("Generate or Upload Data First")
    else:
        df = st.session_state.data
        X = df.drop(["FinalScore","PassFail","PerformanceCategory"], axis=1)
        y = df["FinalScore"]

        X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2)
        model = LinearRegression().fit(X_train,y_train)

        st.markdown('<div class="glass">', unsafe_allow_html=True)
        st.subheader("Enter Student Details")

        input_data = []
        for col in X.columns:
            val = st.number_input(col, float(X[col].min()), float(X[col].max()))
            input_data.append(val)

        if st.button("Predict 🚀"):
            pred = model.predict([input_data])[0]
            st.success(f"Predicted Final Score: {round(pred,2)}")
        st.markdown('</div>', unsafe_allow_html=True)

# ===============================
# VISUALIZATION
# ===============================

elif page == "📈 Visualization":
    if st.session_state.data is None:
        st.warning("Generate or Upload Data First")
    else:
        df = st.session_state.data
        st.markdown('<div class="glass">', unsafe_allow_html=True)

        fig = px.scatter_3d(df, x="Attendance", y="StudyHours",
                            z="FinalScore",
                            color="PerformanceCategory")
        st.plotly_chart(fig, use_container_width=True)

        st.markdown('</div>', unsafe_allow_html=True)

# ===============================
# PERFORMANCE
# ===============================

elif page == "📉 Performance":
    if st.session_state.data is None:
        st.warning("Generate or Upload Data First")
    else:
        df = st.session_state.data
        X = df.drop(["FinalScore","PassFail","PerformanceCategory"], axis=1)
        y = df["FinalScore"]

        X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2)
        model = LinearRegression().fit(X_train,y_train)
        y_pred = model.predict(X_test)

        st.markdown('<div class="glass">', unsafe_allow_html=True)
        st.write("MSE:", mean_squared_error(y_test,y_pred))
        st.write("R2 Score:", r2_score(y_test,y_pred))
        st.markdown('</div>', unsafe_allow_html=True)
