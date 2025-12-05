import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# -----------------------------------------------------------------------------
# 1. APP CONFIGURATION
# -----------------------------------------------------------------------------
st.set_page_config(page_title="Titanic Analytics", layout="wide")

# -----------------------------------------------------------------------------
# 2. DATA LOADING & CACHING
# -----------------------------------------------------------------------------
@st.cache_data
def load_data():
    # Load the processed data
    data = pd.read_csv('data/processed/titanic_processed.csv')
    return data

df = load_data()

# -----------------------------------------------------------------------------
# 3. HELPER FUNCTIONS (The Magic Sauce 🪄)
# -----------------------------------------------------------------------------
def add_bar_labels(ax, total_count):
    """
    Adds labels to a bar chart showing both the absolute count 
    and the percentage relative to the total count.
    """
    for container in ax.containers:
        # Create custom labels: "Count \n (Percentage%)"
        labels = [
            f'{int(h)}\n({h/total_count*100:.1f}%)' if h > 0 else "" 
            for h in container.datavalues
        ]
        
        # Add labels to the center of the bars
        ax.bar_label(container, labels=labels, label_type='center', 
                     color='white', fontweight='bold', fontsize=10)

# -----------------------------------------------------------------------------
# 4. SIDEBAR FILTERS
# -----------------------------------------------------------------------------
st.sidebar.header("🔍 Filter Options")

gender_filter = st.sidebar.multiselect(
    "Gender:",
    options=df['sex'].unique(),
    default=df['sex'].unique()
)

class_filter = st.sidebar.multiselect(
    "Passenger Class:",
    options=sorted(df['pclass'].unique()),
    default=sorted(df['pclass'].unique())
)

embark_filter = st.sidebar.multiselect(
    "Embarkation Port:",
    options=df['embark_town'].unique(),
    default=df['embark_town'].unique()
)

# Apply Filters
filtered_df = df[
    (df['sex'].isin(gender_filter)) & 
    (df['pclass'].isin(class_filter)) &
    (df['embark_town'].isin(embark_filter))
]

# -----------------------------------------------------------------------------
# 5. MAIN DASHBOARD UI
# -----------------------------------------------------------------------------
st.title("🚢 Titanic Survival Prediction Analysis")
st.markdown("---")

# --- SECTION A: KPI METRICS ---
kpi1, kpi2, kpi3, kpi4 = st.columns(4)

total_passengers = len(filtered_df)
survival_rate = filtered_df['survived'].mean() * 100 if total_passengers > 0 else 0
avg_age = filtered_df['age'].mean() if total_passengers > 0 else 0
avg_fare = filtered_df['fare'].mean() if total_passengers > 0 else 0

kpi1.metric("Total Passengers", total_passengers)
kpi2.metric("Survival Rate", f"{survival_rate:.1f}%")
kpi3.metric("Average Age", f"{avg_age:.1f}")
kpi4.metric("Average Fare", f"£{avg_fare:.2f}")

st.markdown("---")

# --- SECTION B: VISUALIZATIONS ---
tab1, tab2, tab3 = st.tabs(["💀 Survival Overview", "💰 Class & Economics", "👨‍👩‍👧 Family & Demographics"])

# TAB 1: SURVIVAL OVERVIEW
with tab1:
    st.header("Survival Analysis")
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("#### Survival by Gender")
        fig_sex, ax_sex = plt.subplots()
        sns.countplot(data=filtered_df, x='sex', hue='survived', palette=['#e74c3c', '#2ecc71'], ax=ax_sex)
        ax_sex.legend(title='Status', labels=['Died', 'Survived'])
        
        # Apply our new helper function!
        add_bar_labels(ax_sex, total_passengers)
        
        st.pyplot(fig_sex)
        
    with col2:
        st.write("#### Survival by Embarkation")
        fig_emb, ax_emb = plt.subplots()
        sns.countplot(data=filtered_df, x='embark_town', hue='survived', palette='viridis', ax=ax_emb)
        
        # Apply helper function
        add_bar_labels(ax_emb, total_passengers)
        
        st.pyplot(fig_emb)

# TAB 2: CLASS & ECONOMICS
with tab2:
    st.header("Socioeconomic Factors")
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("#### Survival by Passenger Class")
        fig_class, ax_class = plt.subplots()
        sns.countplot(data=filtered_df, x='pclass', hue='survived', palette='viridis', ax=ax_class)
        
        # Apply helper function
        add_bar_labels(ax_class, total_passengers)
        
        st.pyplot(fig_class)
        
    with col2:
        st.write("#### Fare Distribution")
        fig_fare, ax_fare = plt.subplots()
        sns.histplot(data=filtered_df, x='fare', hue='survived', kde=True, element="step", ax=ax_fare)
        ax_fare.set_xlim(0, 300)
        st.pyplot(fig_fare)

# TAB 3: FAMILY & DEMOGRAPHICS
with tab3:
    st.header("Family Size Impact")
    st.write("Analysis of your custom feature: `family_size`")
    
    fig_fam, ax_fam = plt.subplots(figsize=(10, 4))
    sns.kdeplot(data=filtered_df[filtered_df['survived']==0], x='family_size', fill=True, color='red', label='Died', ax=ax_fam)
    sns.kdeplot(data=filtered_df[filtered_df['survived']==1], x='family_size', fill=True, color='green', label='Survived', ax=ax_fam)
    ax_fam.legend()
    st.pyplot(fig_fam)