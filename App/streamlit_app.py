import streamlit as st
import pandas as pd
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
import time

# Page Config
st.set_page_config(
    page_title="🧠 AI Health Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load data
@st.cache_data
def load_data():
    df = pd.read_csv("full_mock_health_data.csv", parse_dates=["timestamp"])
    return df

df = load_data()

# Sidebar
st.sidebar.title("🧭 Filter Options")
selected_user = st.sidebar.selectbox("Choose a user:", sorted(df['user_id'].unique()))
selected_metric = st.sidebar.selectbox("Select vital sign to plot:", [
    'heart_rate', 'temperature', 'spo2', 'respiration_rate', 'step_count'
])

# Filter user data
user_df = df[df['user_id'] == selected_user]

# Header
st.markdown("## 🩺 Real-Time Health Monitoring Dashboard")
st.markdown(f"Tracking vitals and detecting early warning signs for **{selected_user}**")

# Animated loading effect
with st.spinner("🔄 Fetching health data..."):
    time.sleep(1.5)

# Summary metrics
st.markdown("### 📊 Vital Stats Overview")
col1, col2, col3, col4 = st.columns(4)
col1.metric("💓 Avg HR", f"{user_df['heart_rate'].mean():.1f} bpm")
col2.metric("🌡️ Avg Temp", f"{user_df['temperature'].mean():.2f} °C")
col3.metric("🧪 Avg SpO2", f"{user_df['spo2'].mean():.2f} %")
col4.metric("🫁 Avg Resp. Rate", f"{user_df['respiration_rate'].mean():.1f} br/min")

# Real-time simulation (last 20 mins)
st.markdown("### ⏱️ Simulated Real-Time Vitals Stream")
latest = user_df.sort_values(by='timestamp').tail(20)
fig_rt = px.line(latest, x="timestamp", y=selected_metric, markers=True)
fig_rt.update_layout(title=f"Real-Time {selected_metric.title()} for {selected_user}", xaxis_title="Time", yaxis_title=selected_metric.title())
st.plotly_chart(fig_rt, use_container_width=True)

# Activity Breakdown
st.markdown("### 🏃 Activity Distribution")
fig_bar = px.histogram(
    user_df,
    x="activity",
    title="Activity Frequency",
    color="activity",
    color_discrete_map={"rest": "#FFD166", "sleep": "#06D6A0", "active": "#EF476F"},
)
st.plotly_chart(fig_bar, use_container_width=True)

# Anomaly Table
st.markdown("### 🚨 Detected Health Anomalies")
anomalies = user_df[user_df["condition_label"] != 0]
if not anomalies.empty:
    st.dataframe(anomalies[['timestamp', 'condition_label', 'heart_rate', 'temperature', 'spo2']])
else:
    st.success("✅ No anomalies detected for this user.")

# Leaderboard - Top 10 Most Active Users
st.markdown("### 🏆 Top 10 Most Active Users (by Total Step Count)")
leaderboard = df.groupby("user_id")["step_count"].sum().sort_values(ascending=False).head(10).reset_index()
fig_leader = px.bar(leaderboard, x="user_id", y="step_count", color="step_count", color_continuous_scale="viridis")
st.plotly_chart(fig_leader, use_container_width=True)

# Compare 2 Users
st.markdown("### 🔍 Compare 2 Users")
user_compare = st.multiselect("Select 2 users to compare:", sorted(df['user_id'].unique()), default=sorted(df['user_id'].unique())[:2])
if len(user_compare) == 2:
    compare_df = df[df["user_id"].isin(user_compare)]
    fig_comp = px.line(compare_df, x="timestamp", y=selected_metric, color="user_id", title=f"{selected_metric.title()} Comparison")
    st.plotly_chart(fig_comp, use_container_width=True)

# Correlation Heatmap
st.markdown("### 🔥 Correlation Heatmap Between Vitals")
fig, ax = plt.subplots()
sns.heatmap(
    user_df[['heart_rate', 'temperature', 'spo2', 'respiration_rate', 'step_count', 'heart_rate_variability']].corr(),
    annot=True, cmap="coolwarm", ax=ax
)
st.pyplot(fig)

# Mobile-friendly mode toggle
st.markdown("### 📱 Mobile-Friendly Mode (Beta)")
if st.sidebar.checkbox("Enable Compact View for Mobile"):
    st.info("✅ Compact view enabled. Visuals optimized for mobile experience. (Preview Mode)")

# User Health Profile Summary
st.markdown("### 📋 User Health Profile Summary")
with st.expander("Click to expand user profile"):
    st.write(f"**Most common activity:** {user_df['activity'].mode()[0]}")
    st.write(f"**Max Heart Rate:** {user_df['heart_rate'].max():.0f} bpm")
    st.write(f"**Min SpO2:** {user_df['spo2'].min():.1f}%")
    st.write(f"**Number of Anomalies:** {anomalies.shape[0]}")
    st.write(f"**Last Reading Timestamp:** {user_df['timestamp'].max().strftime('%Y-%m-%d %H:%M:%S')}")

# Footer
st.markdown("---")
st.markdown("💡 *Built with synthetic health data | Streamlit + Plotly + Seaborn*")
