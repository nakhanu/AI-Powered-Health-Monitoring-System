import streamlit as st
import pandas as pd
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
import time
import numpy as np

# ----------------- PAGE CONFIG -----------------
st.set_page_config(
    page_title="🧠 AI Health Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ----------------- LOAD DATA -----------------
@st.cache_data
def load_data():
    df = pd.read_csv("full_mock_health_data.csv", parse_dates=["timestamp"])
    return df

df = load_data()

# ----------------- SIDEBAR -----------------
st.sidebar.title("🧭 Filter Options")
selected_user = st.sidebar.selectbox("Choose a user:", sorted(df['user_id'].unique()))
selected_metric = st.sidebar.selectbox("Select vital sign to plot:", [
    'heart_rate', 'temperature', 'spo2', 'respiration_rate', 'step_count'
])

# Dark/Light Mode Toggle
mode = st.sidebar.radio("Select Theme Mode:", options=["Light", "Dark"])
if mode == "Dark":
    st.markdown("""
        <style>
            body, .css-1v3fvcr, .stApp {
                background-color: #1e1e1e;
                color: #fafafa;
            }
        </style>
    """, unsafe_allow_html=True)

# ----------------- USER FILTER -----------------
user_df = df[df['user_id'] == selected_user]

# ----------------- HEADER -----------------
st.markdown("## 🩺 Real-Time Health Monitoring Dashboard")

# ----------------- LOADING SIMULATION -----------------
with st.spinner("🔄 Fetching health data..."):
    time.sleep(1.5)

# ----------------- SUMMARY METRICS -----------------
st.markdown("### 📊 Vital Stats Overview")
col1, col2, col3, col4 = st.columns(4)
col1.metric("💓 Avg HR", f"{user_df['heart_rate'].mean():.1f} bpm")
col2.metric("🌡️ Avg Temp", f"{user_df['temperature'].mean():.2f} °C")
col3.metric("🧪 Avg SpO2", f"{user_df['spo2'].mean():.2f} %")
col4.metric("🫁 Avg Resp. Rate", f"{user_df['respiration_rate'].mean():.1f} br/min")

# ----------------- RISK SCORING MODEL -----------------
def calculate_risk_score(row):
    score = 0
    if row['heart_rate'] > 100 or row['heart_rate'] < 50:
        score += 2
    if row['temperature'] > 38 or row['temperature'] < 35.5:
        score += 2
    if row['spo2'] < 92:
        score += 3
    if row['respiration_rate'] > 22 or row['respiration_rate'] < 10:
        score += 2
    return score

user_df['risk_score'] = user_df.apply(calculate_risk_score, axis=1)
st.markdown("### ⚠️ Health Risk Score")
latest_risk = user_df[['timestamp', 'risk_score']].sort_values(by='timestamp').tail(30)
fig_risk = px.line(latest_risk, x="timestamp", y="risk_score", markers=True)
fig_risk.update_layout(title="Recent Health Risk Score Trend", yaxis_title="Risk Score")
st.plotly_chart(fig_risk, use_container_width=True)

# ----------------- AI RECOMMENDATION ENGINE -----------------
st.markdown("### 🤖 AI Health Recommendations")
last_row = user_df.iloc[-1]
recommendations = []

if last_row['spo2'] < 92:
    recommendations.append("🔴 SpO2 is low: Consider checking for respiratory issues or using supplemental oxygen.")
if last_row['heart_rate'] > 100:
    recommendations.append("⚠️ Elevated heart rate: Avoid physical exertion, consider hydration and rest.")
if last_row['temperature'] > 38:
    recommendations.append("🌡️ High temperature: Possible fever. Monitor closely or consult a physician.")
if not recommendations:
    st.success("✅ All vitals appear normal. Keep monitoring regularly!")
else:
    for rec in recommendations:
        st.warning(rec)

# ----------------- REAL-TIME SIMULATION -----------------
st.markdown("### ⏱️ Simulated Real-Time Vitals Stream")
latest = user_df.sort_values(by='timestamp').tail(20)
fig_rt = px.line(latest, x="timestamp", y=selected_metric, markers=True)
fig_rt.update_layout(title=f"Real-Time {selected_metric.title()} for {selected_user}", xaxis_title="Time", yaxis_title=selected_metric.title())
st.plotly_chart(fig_rt, use_container_width=True)

# ----------------- ACTIVITY BREAKDOWN -----------------
st.markdown("### 🏃 Activity Distribution")
fig_bar = px.histogram(
    user_df,
    x="activity",
    title="Activity Frequency",
    color="activity",
    color_discrete_map={"rest": "#FFD166", "sleep": "#06D6A0", "active": "#EF476F"},
)
st.plotly_chart(fig_bar, use_container_width=True)

# ----------------- ANOMALIES -----------------
st.markdown("### 🚨 Detected Health Anomalies")
anomalies = user_df[user_df["condition_label"] != 0]
if not anomalies.empty:
    st.dataframe(anomalies[['timestamp', 'condition_label', 'heart_rate', 'temperature', 'spo2']])
else:
    st.success("✅ No anomalies detected for this user.")

# ----------------- LEADERBOARD -----------------
st.markdown("### 🏆 Top 10 Most Active Users (by Total Step Count)")
leaderboard = df.groupby("user_id")["step_count"].sum().sort_values(ascending=False).head(10).reset_index()
fig_leader = px.bar(leaderboard, x="user_id", y="step_count", color="step_count", color_continuous_scale="viridis")
st.plotly_chart(fig_leader, use_container_width=True)

# ----------------- USER COMPARISON -----------------
st.markdown("### 🔍 Compare 2 Users")
user_compare = st.multiselect("Select 2 users to compare:", sorted(df['user_id'].unique()), default=sorted(df['user_id'].unique())[:2])
if len(user_compare) == 2:
    compare_df = df[df["user_id"].isin(user_compare)]
    fig_comp = px.line(compare_df, x="timestamp", y=selected_metric, color="user_id", title=f"{selected_metric.title()} Comparison")
    st.plotly_chart(fig_comp, use_container_width=True)

# ----------------- HEATMAP -----------------
st.markdown("### 🔥 Correlation Heatmap Between Vitals")
fig, ax = plt.subplots()
sns.heatmap(
    user_df[['heart_rate', 'temperature', 'spo2', 'respiration_rate', 'step_count', 'heart_rate_variability']].corr(),
    annot=True, cmap="coolwarm", ax=ax
)
st.pyplot(fig)

# ----------------- DOWNLOAD BUTTON -----------------
st.markdown("### ⬇️ Download User Data")
csv = user_df.to_csv(index=False).encode('utf-8')
st.download_button("Download CSV", csv, f"{selected_user}_health_data.csv", "text/csv")

# ----------------- MOBILE MODE TOGGLE -----------------
st.markdown("### 📱 Mobile-Friendly Mode (Beta)")
if st.sidebar.checkbox("Enable Compact View for Mobile"):
    st.info("✅ Compact view enabled. Visuals optimized for mobile experience. (Preview Mode)")

# ----------------- USER HEALTH SUMMARY -----------------
st.markdown("### 📋 User Health Profile Summary")
with st.expander("Click to expand user profile"):
    st.write(f"**Most common activity:** {user_df['activity'].mode()[0]}")
    st.write(f"**Max Heart Rate:** {user_df['heart_rate'].max():.0f} bpm")
    st.write(f"**Min SpO2:** {user_df['spo2'].min():.1f}%")
    st.write(f"**Number of Anomalies:** {anomalies.shape[0]}")
    st.write(f"**Last Reading Timestamp:** {user_df['timestamp'].max().strftime('%Y-%m-%d %H:%M:%S')}")

# ----------------- BASIC HEALTH CHATBOT -----------------
st.markdown("### 💬 Ask Our AI Health Assistant")
chat_query = st.text_input("Ask a health-related question or greeting:")

faq_responses = {
    "hello": "Hi there! 👋 How can I assist you with your health stats today?",
    "hi": "Hello! 👋 Feel free to ask me about your vitals or how to interpret them.",
    "what is a normal heart rate": "A normal resting heart rate for adults ranges from 60 to 100 bpm.",
    "what does spo2 mean": "SpO₂ stands for peripheral capillary oxygen saturation, indicating oxygen levels in the blood. Normal values are usually 95% or higher.",
    "how is risk score calculated": "Risk score is based on thresholds of heart rate, temperature, SpO₂, and respiration rate. The higher the score, the higher the potential health risk.",
    "what is heart rate variability": "Heart Rate Variability (HRV) measures the time variation between heartbeats. Higher HRV is generally healthier and indicates better cardiovascular fitness.",
    "what is a normal temperature": "A typical body temperature ranges from 36.1°C to 37.2°C. Temperatures above 38°C may indicate fever.",
    "what is a normal respiration rate": "The normal respiration rate for an adult at rest is 12 to 20 breaths per minute.",
    "what does a high risk score mean": "A high risk score suggests abnormalities in vitals. This could indicate stress, illness, or need for medical attention.",
    "how to improve heart health": "Stay active, eat a balanced diet, avoid smoking, manage stress, and get regular health screenings to improve heart health.",
    "why is sleep important": "Sleep helps the body repair, restore energy, and regulate vital signs like heart rate and blood pressure. Poor sleep can increase risk of chronic diseases."
}

if chat_query:
    query = chat_query.lower()
    matched = False
    for key in faq_responses:
        if key in query:
            st.success(f"🧠 Assistant: {faq_responses[key]}")
            matched = True
            break
    if not matched:
        st.warning("🤖 I'm still learning! Try asking about vitals like 'heart rate', 'SpO₂', or 'risk score'.")

# ----------------- FOOTER -----------------
st.markdown("---")
st.caption("💡 Built with synthetic health data | Streamlit + Plotly + Seaborn")
