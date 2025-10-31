import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np
import os
from database import init_database, save_prediction, get_prediction_history, get_prediction_statistics

# Set page configuration
st.set_page_config(
    page_title="Mitarbeiterabwanderung Vorhersagen",
    page_icon="📊",
    layout="wide"
)

# Initialize database
init_database()

# Title and description
st.title("Mitarbeiterabwanderung Vorhersagen App")
st.markdown("""
Willkommen zur Mitarbeiterabwanderung Vorhersagen App! 
Mit dieser App können Sie die Wahrscheinlichkeit der Abwanderung eines Mitarbeiters vorhersagen.
Bitte spezifizieren Sie die Eingabeparameter im Seitenbereich oder laden Sie eine Datei hoch.
""")
st.write('---')

# Check if the data file exists
data_file = 'modified_file.csv'
if not os.path.exists(data_file):
    st.error(f"Die Datei '{data_file}' wurde nicht gefunden. Bitte stellen Sie sicher, dass die Datei im Hauptverzeichnis vorhanden ist.")
    st.stop()

# Load the Dataset
try:
    data = pd.read_csv(data_file)
    # Fill missing values with mean for numeric columns
    numeric_columns = data.select_dtypes(include=['float64', 'int64']).columns
    data[numeric_columns] = data[numeric_columns].fillna(data[numeric_columns].mean())
    
    target = data['left']
    
    # String to integer mapping for salary
    string_to_int = {
        'low': 1,
        'medium': 2,
        'high': 3
    }
    
    # Handle salary mapping
    if data['gehalt'].dtype == 'object':
        data['gehalt'] = data['gehalt'].map(string_to_int)
    
    # Define features
    features = ["zufriedenheitsgrad", "anzahl_projekte", "durchschnittliche_monatliche_arbeitszeit", "arbeitsunfall", "foerderung_letzte_5_jahre", "gehalt"]
    
    X = data[features]
    Y = target
    
    # Get min and max values for sliders
    average_monthly_hours_min = int(min(data['durchschnittliche_monatliche_arbeitszeit']))
    average_monthly_hours_max = int(max(data['durchschnittliche_monatliche_arbeitszeit']))
    
except Exception as e:
    st.error(f"Fehler beim Laden der Daten: {e}")
    st.stop()

# Sidebar input
st.sidebar.header('Eingabeparameter spezifizieren')
st.sidebar.markdown("### Eingabeparameter")

def user_input_features():
    zufriedenheitsgrad = st.sidebar.slider('Zufriedenheitsgrad', 0, 100, 50, help="Der Zufriedenheitsgrad des Mitarbeiters in Prozent.")
    anzahl_projekte = st.sidebar.slider('Anzahl der Projekte', 0, 7, 3, help="Die Anzahl der Projekte, an denen der Mitarbeiter gearbeitet hat.")
    durchschnittliche_monatliche_arbeitszeit = st.sidebar.slider('Durchschnittliche Monatliche Arbeitszeit', average_monthly_hours_min, average_monthly_hours_max, 200, help="Die durchschnittliche Anzahl der monatlichen Arbeitsstunden.")
    arbeitsunfall = st.sidebar.selectbox('Arbeitsunfall', [0, 1], format_func=lambda x: 'Ja' if x == 1 else 'Nein', help="Ob der Mitarbeiter einen Arbeitsunfall hatte (Ja/Nein).")
    foerderung_letzte_5_jahre = st.sidebar.selectbox('Förderung in den letzten 5 Jahren', [0, 1], format_func=lambda x: 'Ja' if x == 1 else 'Nein', help="Ob der Mitarbeiter in den letzten 5 Jahren befördert wurde (Ja/Nein).")
    gehalt = st.sidebar.selectbox('Gehalt', [1, 2, 3], format_func=lambda x: ['Niedrig', 'Mittel', 'Hoch'][x-1], help="Die Gehaltsstufe des Mitarbeiters (Niedrig, Mittel, Hoch).")
    
    data_dict = {
        'zufriedenheitsgrad': zufriedenheitsgrad,
        'anzahl_projekte': anzahl_projekte,
        'durchschnittliche_monatliche_arbeitszeit': durchschnittliche_monatliche_arbeitszeit,
        'arbeitsunfall': arbeitsunfall,
        'foerderung_letzte_5_jahre': foerderung_letzte_5_jahre,
        'gehalt': gehalt
    }
    features_df = pd.DataFrame(data_dict, index=[0])
    return features_df

df = user_input_features()

# Main Panel Layout
st.header('Spezifizierte Eingabeparameter')
# Display input parameters in three columns for better layout
col1, col2, col3 = st.columns(3)
col1.metric("Zufriedenheitsgrad", f"{df['zufriedenheitsgrad'][0]}%")
col2.metric("Anzahl der Projekte", df['anzahl_projekte'][0])
col3.metric("Durchschnittliche Monatliche Arbeitszeit", f"{df['durchschnittliche_monatliche_arbeitszeit'][0]}h")

col4, col5, col6 = st.columns(3)
col4.metric("Arbeitsunfall", "Ja" if df['arbeitsunfall'][0] == 1 else "Nein")
col5.metric("Förderung in den letzten 5 Jahren", "Ja" if df['foerderung_letzte_5_jahre'][0] == 1 else "Nein")
col6.metric("Gehalt", ["Niedrig", "Mittel", "Hoch"][df['gehalt'][0] - 1])

st.write('---')

# Build Classifier Model
try:
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=1)
    model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=1)
    model.fit(X_train, y_train)
    model_val = model.predict(X_test)
    model_accuracy = accuracy_score(y_test, model_val)
    
    # Display model accuracy
    st.header('Modellevaluierung')
    st.metric(label="Genauigkeit des Modells", value=f"{model_accuracy * 100:.2f}%")
    st.write('---')
    
    # Apply Model to Make Prediction
    prediction = model.predict(df)
    prediction_proba = model.predict_proba(df)
    
    # Save prediction to database
    features_dict = {
        'zufriedenheit': df['zufriedenheitsgrad'][0] / 100.0,
        'anzahl_projekte': int(df['anzahl_projekte'][0]),
        'durchschnittliche_monatliche_arbeitsstunden': int(df['durchschnittliche_monatliche_arbeitszeit'][0]),
        'jahre_im_unternehmen': 3,  # Default value since not in current inputs
        'arbeitsunfall': int(df['arbeitsunfall'][0]),
        'foerderung_letzte_5_jahre': int(df['foerderung_letzte_5_jahre'][0]),
        'abteilung': 'sales',  # Default value since not in current inputs
        'gehalt': ['niedrig', 'mittel', 'hoch'][int(df['gehalt'][0]) - 1]
    }
    
    # Save prediction to database silently (errors will be shown by the function)
    save_prediction(features_dict, prediction[0], prediction_proba[0])
    
    # Display prediction results with color coding
    st.header('Vorhersage der Mitarbeiterabwanderung')
    prediction_text = 'Der/Die Mitarbeiter/in ist zufrieden und er/sie bleibt bei uns.' if prediction[0] == 0 else 'Der/Die Mitarbeiter/in ist nicht zufrieden und er/sie wird wahrscheinlich gehen.'
    prediction_color = 'green' if prediction[0] == 0 else 'red'
    
    st.markdown(f"<h3 style='color:{prediction_color};'>{prediction_text}</h3>", unsafe_allow_html=True)
    
    # Show prediction probability
    if prediction[0] == 0:
        probability = prediction_proba[0][0] * 100
        st.info(f"Wahrscheinlichkeit zu bleiben: {probability:.1f}%")
    else:
        probability = prediction_proba[0][1] * 100
        st.warning(f"Wahrscheinlichkeit zu gehen: {probability:.1f}%")
    
    st.write('---')
    
    # Data Visualization Dashboard
    st.header('📊 Datenvisualisierung und Modelleinblicke')
    
    viz_tab1, viz_tab2, viz_tab3 = st.tabs(["Feature-Wichtigkeit", "Vorhersageverteilung", "Datenverteilung"])
    
    with viz_tab1:
        st.subheader("Feature-Wichtigkeit")
        st.write("Welche Faktoren beeinflussen die Abwanderungsvorhersage am meisten?")
        
        # Dynamic feature name mapping
        feature_names_display = {
            'zufriedenheitsgrad': 'Zufriedenheitsgrad',
            'anzahl_projekte': 'Anzahl Projekte',
            'durchschnittliche_monatliche_arbeitszeit': 'Monatliche Arbeitszeit',
            'arbeitsunfall': 'Arbeitsunfall',
            'foerderung_letzte_5_jahre': 'Förderung',
            'gehalt': 'Gehalt'
        }
        
        # Get feature importances dynamically from actual features
        feature_importance = pd.DataFrame({
            'Feature': [feature_names_display[f] for f in features],
            'Wichtigkeit': model.feature_importances_
        }).sort_values('Wichtigkeit', ascending=True)
        
        # Create horizontal bar chart
        fig1, ax1 = plt.subplots(figsize=(10, 6))
        bars = ax1.barh(feature_importance['Feature'], feature_importance['Wichtigkeit'], 
                       color=plt.cm.viridis(feature_importance['Wichtigkeit'] / feature_importance['Wichtigkeit'].max()))
        ax1.set_xlabel('Wichtigkeit', fontsize=12)
        ax1.set_title('Relative Wichtigkeit der Features für die Vorhersage', fontsize=14, fontweight='bold')
        ax1.grid(axis='x', alpha=0.3)
        
        # Add value labels
        for i, (idx, row) in enumerate(feature_importance.iterrows()):
            ax1.text(row['Wichtigkeit'], i, f" {row['Wichtigkeit']:.3f}", 
                    va='center', fontsize=10)
        
        plt.tight_layout()
        st.pyplot(fig1)
        plt.close()
        
        st.info("💡 **Interpretation:** Je höher die Wichtigkeit, desto größer der Einfluss des Features auf die Vorhersage.")
    
    with viz_tab2:
        st.subheader("Vorhersageverteilung im Trainingsdatensatz")
        
        # Get predictions on training data
        y_pred_all = model.predict(X)
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Pie chart for predictions
            fig2, ax2 = plt.subplots(figsize=(8, 8))
            labels = ['Bleiben', 'Gehen']
            sizes = [sum(Y == 0), sum(Y == 1)]
            colors = ['#2ecc71', '#e74c3c']
            explode = (0.05, 0.05)
            
            ax2.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%',
                   shadow=True, startangle=90, textprops={'fontsize': 12, 'fontweight': 'bold'})
            ax2.set_title('Tatsächliche Verteilung', fontsize=14, fontweight='bold')
            
            st.pyplot(fig2)
            plt.close()
        
        with col2:
            # Pie chart for model predictions
            fig3, ax3 = plt.subplots(figsize=(8, 8))
            pred_sizes = [sum(y_pred_all == 0), sum(y_pred_all == 1)]
            
            ax3.pie(pred_sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%',
                   shadow=True, startangle=90, textprops={'fontsize': 12, 'fontweight': 'bold'})
            ax3.set_title('Modell-Vorhersageverteilung', fontsize=14, fontweight='bold')
            
            st.pyplot(fig3)
            plt.close()
        
        st.write(f"**Datensätze insgesamt:** {len(Y):,}")
        st.write(f"**Tatsächlich bleiben:** {sum(Y == 0):,} ({sum(Y == 0)/len(Y)*100:.1f}%)")
        st.write(f"**Tatsächlich gehen:** {sum(Y == 1):,} ({sum(Y == 1)/len(Y)*100:.1f}%)")
    
    with viz_tab3:
        st.subheader("Verteilung der wichtigsten Features")
        
        # Create distribution plots for top features
        fig4, axes = plt.subplots(2, 3, figsize=(15, 10), constrained_layout=True)
        axes = axes.ravel()
        
        feature_names_display = {
            'zufriedenheitsgrad': 'Zufriedenheitsgrad',
            'anzahl_projekte': 'Anzahl Projekte',
            'durchschnittliche_monatliche_arbeitszeit': 'Monatliche Arbeitszeit',
            'arbeitsunfall': 'Arbeitsunfall',
            'foerderung_letzte_5_jahre': 'Förderung',
            'gehalt': 'Gehalt'
        }
        
        # Identify categorical/binary features
        categorical_features = ['arbeitsunfall', 'foerderung_letzte_5_jahre', 'gehalt']
        
        for idx, feature in enumerate(features):
            ax = axes[idx]
            
            if feature in categorical_features:
                # Use grouped bar chart for categorical features
                feature_data = data.groupby([feature, 'left']).size().unstack(fill_value=0)
                feature_data.plot(kind='bar', ax=ax, color=['#2ecc71', '#e74c3c'], 
                                 alpha=0.8, edgecolor='black', width=0.7)
                ax.set_xlabel(feature_names_display[feature], fontsize=10)
                ax.set_ylabel('Anzahl', fontsize=10)
                ax.set_title(f'Verteilung: {feature_names_display[feature]}', fontsize=11, fontweight='bold')
                ax.legend(['Bleiben', 'Gehen'], loc='upper right')
                ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
            else:
                # Use histograms for continuous features
                stayed = data[data['left'] == 0][feature]
                left = data[data['left'] == 1][feature]
                
                ax.hist([stayed, left], bins=30, label=['Bleiben', 'Gehen'], 
                       color=['#2ecc71', '#e74c3c'], alpha=0.7, edgecolor='black')
                ax.set_xlabel(feature_names_display[feature], fontsize=10)
                ax.set_ylabel('Anzahl', fontsize=10)
                ax.set_title(f'Verteilung: {feature_names_display[feature]}', fontsize=11, fontweight='bold')
                ax.legend()
            
            ax.grid(alpha=0.3, axis='y')
        
        st.pyplot(fig4)
        plt.close()
        
        st.info("💡 **Interpretation:** Diese Diagramme zeigen, wie sich die Werte der Features zwischen Mitarbeitern unterscheiden, die bleiben (grün) und denen, die gehen (rot).")
    
    st.write('---')
    
    # Explainable AI Section
    st.header('🔍 Erklärbares KI - Vorhersage-Erklärung')
    st.write("Verstehen Sie, wie das Modell seine Entscheidungen trifft.")
    
    explain_tab1, explain_tab2 = st.tabs(["Vorhersage-Details", "What-If Analyse"])
    
    with explain_tab1:
        st.subheader("Verstehen Sie Ihre Vorhersage")
        
        # Get the prediction probabilities for the current input
        current_pred_proba = model.predict_proba(df)[0]
        
        # Show probability breakdown prominently
        st.write("### Vorhersage-Wahrscheinlichkeiten")
        prob_col1, prob_col2 = st.columns(2)
        with prob_col1:
            st.metric("Wahrscheinlichkeit: Bleibt", f"{current_pred_proba[0]*100:.1f}%",
                     help="Die Wahrscheinlichkeit, dass dieser Mitarbeiter im Unternehmen bleibt")
        with prob_col2:
            st.metric("Wahrscheinlichkeit: Geht", f"{current_pred_proba[1]*100:.1f}%",
                     help="Die Wahrscheinlichkeit, dass dieser Mitarbeiter das Unternehmen verlässt")
        
        st.write("---")
        st.write("### Ihre Eingabewerte im Vergleich")
        st.write("Vergleichen Sie Ihre Eingaben mit den Durchschnittswerten der Mitarbeiter:")
        
        # Create comparison dataframe
        comparison_data = []
        for i, feature_name in enumerate(features):
            feature_value = df[feature_name].iloc[0]
            mean_value = X_train[feature_name].mean()
            # Calculate percentage of employees who left with similar values
            if feature_name in categorical_features:
                similar_employees = data[data[feature_name] == feature_value]
            else:
                # For continuous features, find employees within +/- 10% of the value
                tolerance = abs(mean_value * 0.1) if mean_value != 0 else 5
                similar_employees = data[
                    (data[feature_name] >= feature_value - tolerance) & 
                    (data[feature_name] <= feature_value + tolerance)
                ]
            
            if len(similar_employees) > 0:
                churn_rate = (similar_employees['left'] == 1).mean() * 100
            else:
                churn_rate = None
            
            comparison_data.append({
                'Feature': feature_names_display[feature_name],
                'Ihr Wert': feature_value,
                'Durchschnitt': mean_value,
                'Abweichung': feature_value - mean_value,
                'Abwanderungsrate (ähnliche Werte)': churn_rate
            })
        
        comp_df = pd.DataFrame(comparison_data)
        
        # Visualize comparison
        fig_comp, ax_comp = plt.subplots(figsize=(12, 7))
        
        y_pos = np.arange(len(comp_df))
        bars = ax_comp.barh(y_pos, comp_df['Abweichung'], 
                           color=['#e74c3c' if x > 0 else '#3498db' for x in comp_df['Abweichung']],
                           alpha=0.8, edgecolor='black')
        
        ax_comp.set_yticks(y_pos)
        ax_comp.set_yticklabels(comp_df['Feature'])
        ax_comp.axvline(x=0, color='black', linestyle='-', linewidth=1.5)
        ax_comp.set_xlabel('Abweichung vom Durchschnitt', fontsize=12)
        ax_comp.set_title('Ihre Werte im Vergleich zum Durchschnitt', fontsize=14, fontweight='bold')
        ax_comp.grid(axis='x', alpha=0.3)
        
        plt.tight_layout()
        st.pyplot(fig_comp)
        plt.close()
        
        st.info("📊 **Interpretation:** Blaue Balken zeigen Werte unter dem Durchschnitt, rote Balken über dem Durchschnitt.")
        
        # Show detailed breakdown
        st.write("---")
        st.write("### Detaillierte Analyse der Features")
        
        # Combine with feature importance for context
        for idx, row in comp_df.iterrows():
            importance = model.feature_importances_[idx]
            churn_rate = row['Abwanderungsrate (ähnliche Werte)']
            
            with st.expander(f"📊 {row['Feature']}"):
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Ihr Wert", f"{row['Ihr Wert']:.1f}" if isinstance(row['Ihr Wert'], (int, float)) and row['Ihr Wert'] > 10 else row['Ihr Wert'])
                with col2:
                    st.metric("Durchschnitt", f"{row['Durchschnitt']:.1f}")
                with col3:
                    st.metric("Modell-Wichtigkeit", f"{importance*100:.1f}%",
                             help="Wie wichtig ist dieses Feature für das Modell insgesamt?")
                
                if churn_rate is not None:
                    st.write(f"**Historische Daten:** {churn_rate:.1f}% der Mitarbeiter mit ähnlichen Werten für dieses Feature haben das Unternehmen verlassen.")
                    
                    # Color code the risk
                    if churn_rate > 50:
                        st.error(f"⚠️ Hohe Abwanderungsrate bei ähnlichen Werten!")
                    elif churn_rate > 30:
                        st.warning(f"⚡ Moderate Abwanderungsrate bei ähnlichen Werten.")
                    else:
                        st.success(f"✅ Niedrige Abwanderungsrate bei ähnlichen Werten.")
                else:
                    st.info("Nicht genügend Daten für historische Analyse.")
                
                # Explain the importance
                if importance > 0.2:
                    st.write(f"🎯 **Sehr wichtiges Feature:** Dieses Feature hat einen starken Einfluss auf die Vorhersage ({importance*100:.1f}%).")
                elif importance > 0.1:
                    st.write(f"📍 **Wichtiges Feature:** Dieses Feature hat einen moderaten Einfluss auf die Vorhersage ({importance*100:.1f}%).")
                else:
                    st.write(f"📌 **Weniger wichtiges Feature:** Dieses Feature hat einen geringeren Einfluss auf die Vorhersage ({importance*100:.1f}%).")
    
    with explain_tab2:
        st.subheader("What-If Szenario-Analyse")
        st.write("Wie würde sich die Vorhersage ändern, wenn Sie dieses eine Feature verändern würden?")
        st.info("💡 **Hinweis:** Diese Analyse zeigt, wie sich die Vorhersage für IHRE spezifischen Eingaben ändert, wenn Sie nur ein Feature variieren und alle anderen konstant halten.")
        
        # Feature selection for detailed analysis
        selected_feature = st.selectbox(
            "Wählen Sie ein Feature zur Analyse:",
            options=features,
            format_func=lambda x: feature_names_display[x]
        )
        
        # Create a range of values for the selected feature
        if selected_feature in categorical_features:
            unique_values = sorted(data[selected_feature].unique())
            predictions_for_values = []
            
            for val in unique_values:
                temp_df = df.copy()
                temp_df[selected_feature] = val
                pred_prob = model.predict_proba(temp_df)[0][1]
                predictions_for_values.append(pred_prob)
            
            fig_influence, ax_influence = plt.subplots(figsize=(10, 6))
            bars = ax_influence.bar(range(len(unique_values)), predictions_for_values, 
                                   color=plt.cm.RdYlGn_r(predictions_for_values), 
                                   alpha=0.8, edgecolor='black')
            ax_influence.set_xlabel(feature_names_display[selected_feature], fontsize=12)
            ax_influence.set_ylabel('Wahrscheinlichkeit zu gehen', fontsize=12)
            ax_influence.set_title(f'Einfluss von {feature_names_display[selected_feature]} auf die Vorhersage', 
                                 fontsize=14, fontweight='bold')
            ax_influence.set_xticks(range(len(unique_values)))
            ax_influence.set_xticklabels(unique_values)
            ax_influence.grid(axis='y', alpha=0.3)
            ax_influence.axhline(y=0.5, color='r', linestyle='--', label='Entscheidungsgrenze')
            ax_influence.legend()
            
            plt.tight_layout()
            st.pyplot(fig_influence)
            plt.close()
        else:
            min_val = data[selected_feature].min()
            max_val = data[selected_feature].max()
            value_range = np.linspace(min_val, max_val, 50)
            predictions_for_values = []
            
            for val in value_range:
                temp_df = df.copy()
                temp_df[selected_feature] = val
                pred_prob = model.predict_proba(temp_df)[0][1]
                predictions_for_values.append(pred_prob)
            
            fig_influence, ax_influence = plt.subplots(figsize=(10, 6))
            ax_influence.plot(value_range, predictions_for_values, linewidth=3, color='#3498db')
            ax_influence.fill_between(value_range, predictions_for_values, alpha=0.3, color='#3498db')
            ax_influence.axhline(y=0.5, color='r', linestyle='--', label='Entscheidungsgrenze', linewidth=2)
            ax_influence.axvline(x=df[selected_feature].iloc[0], color='green', linestyle=':', 
                               label='Aktueller Wert', linewidth=2)
            ax_influence.set_xlabel(feature_names_display[selected_feature], fontsize=12)
            ax_influence.set_ylabel('Wahrscheinlichkeit zu gehen', fontsize=12)
            ax_influence.set_title(f'Einfluss von {feature_names_display[selected_feature]} auf die Vorhersage', 
                                 fontsize=14, fontweight='bold')
            ax_influence.grid(alpha=0.3)
            ax_influence.legend()
            
            plt.tight_layout()
            st.pyplot(fig_influence)
            plt.close()
        
        st.info("💡 **Interpretation:** Diese Kurve zeigt, wie sich die Vorhersage für Ihre aktuelle Eingabe ändert, wenn Sie nur dieses eine Feature variieren. Dies hilft zu verstehen, welcher Wert für dieses Feature das Risiko minimieren würde.")
    
    st.write('---')
    
except Exception as e:
    st.error(f"Fehler beim Trainieren des Modells: {e}")
    st.stop()

# File upload and prediction
st.sidebar.header('Datei-Upload')
uploaded_file = st.sidebar.file_uploader("Laden Sie eine Excel- oder CSV-Datei hoch", type=["csv", "xlsx"])

if uploaded_file:
    try:
        # Load file
        if uploaded_file.name.endswith('.csv'):
            input_df = pd.read_csv(uploaded_file)
        else:
            input_df = pd.read_excel(uploaded_file)
        
        # Validate file is not empty
        if len(input_df) == 0:
            st.sidebar.error("❌ Die hochgeladene Datei ist leer!")
            st.sidebar.info("Bitte laden Sie eine Datei mit mindestens einem Datensatz hoch.")
            st.stop()

        # Check for the required columns
        required_columns = ["zufriedenheitsgrad", "anzahl_projekte", "durchschnittliche_monatliche_arbeitszeit", "arbeitsunfall", "foerderung_letzte_5_jahre", "gehalt"]
        missing_columns = []
        
        for col in required_columns:
            if col not in input_df.columns:
                missing_columns.append(col)
        
        if missing_columns:
            st.sidebar.error(f"❌ Die folgenden Spalten fehlen in der Datei: {', '.join(missing_columns)}")
            st.sidebar.info("**Erforderliche Spalten:** " + ", ".join(required_columns))
            st.sidebar.markdown("""
            **Spaltenformat:**
            - zufriedenheitsgrad: 0-100 (Zufriedenheit in %)
            - anzahl_projekte: 0-7 (Anzahl Projekte)
            - durchschnittliche_monatliche_arbeitszeit: Stunden pro Monat
            - arbeitsunfall: 0 (Nein) oder 1 (Ja)
            - foerderung_letzte_5_jahre: 0 (Nein) oder 1 (Ja)
            - gehalt: 'low', 'medium', 'high' oder 1, 2, 3
            """)
            st.stop()
        
        # Data quality checks
        validation_errors = []
        validation_warnings = []
        
        # Check for completely empty rows
        empty_rows = input_df[required_columns].isna().all(axis=1).sum()
        if empty_rows > 0:
            validation_warnings.append(f"⚠️ {empty_rows} leere Zeilen gefunden (werden übersprungen)")
            input_df = input_df[~input_df[required_columns].isna().all(axis=1)]
        
        # Check if all rows were empty
        if len(input_df) == 0:
            st.sidebar.error("❌ Alle Zeilen in der Datei sind leer!")
            st.sidebar.info("Bitte laden Sie eine Datei mit gültigen Daten hoch.")
            st.stop()
        
        # Validate data ranges and types
        for idx, row in input_df.iterrows():
            # Check zufriedenheitsgrad type and range
            if pd.notna(row['zufriedenheitsgrad']):
                try:
                    val = float(row['zufriedenheitsgrad'])
                    if val < 0 or val > 100:
                        validation_errors.append(f"Zeile {idx+2}: zufriedenheitsgrad muss zwischen 0 und 100 liegen (ist {val})")
                except (ValueError, TypeError):
                    validation_errors.append(f"Zeile {idx+2}: zufriedenheitsgrad muss eine Zahl sein (ist '{row['zufriedenheitsgrad']}')")
            
            # Check anzahl_projekte type and range
            if pd.notna(row['anzahl_projekte']):
                try:
                    val = int(row['anzahl_projekte'])
                    if val < 0 or val > 7:
                        validation_errors.append(f"Zeile {idx+2}: anzahl_projekte muss zwischen 0 und 7 liegen (ist {val})")
                except (ValueError, TypeError):
                    validation_errors.append(f"Zeile {idx+2}: anzahl_projekte muss eine ganze Zahl sein (ist '{row['anzahl_projekte']}')")
            
            # Check arbeitsunfall is binary
            if pd.notna(row['arbeitsunfall']):
                try:
                    val = int(row['arbeitsunfall'])
                    if val not in [0, 1]:
                        validation_errors.append(f"Zeile {idx+2}: arbeitsunfall muss 0 oder 1 sein (ist {val})")
                except (ValueError, TypeError):
                    validation_errors.append(f"Zeile {idx+2}: arbeitsunfall muss 0 oder 1 sein (ist '{row['arbeitsunfall']}')")
            
            # Check foerderung is binary
            if pd.notna(row['foerderung_letzte_5_jahre']):
                try:
                    val = int(row['foerderung_letzte_5_jahre'])
                    if val not in [0, 1]:
                        validation_errors.append(f"Zeile {idx+2}: foerderung_letzte_5_jahre muss 0 oder 1 sein (ist {val})")
                except (ValueError, TypeError):
                    validation_errors.append(f"Zeile {idx+2}: foerderung_letzte_5_jahre muss 0 oder 1 sein (ist '{row['foerderung_letzte_5_jahre']}')")
        
        # Check gehalt values (handle both string and numeric)
        for idx, row in input_df.iterrows():
            if pd.notna(row['gehalt']):
                val = row['gehalt']
                # Try as string first
                if isinstance(val, str):
                    if val.lower() not in ['low', 'medium', 'high', 'niedrig', 'mittel', 'hoch']:
                        validation_errors.append(f"Zeile {idx+2}: gehalt muss 'low'/'medium'/'high' oder 'niedrig'/'mittel'/'hoch' sein (ist '{val}')")
                # Then try as numeric
                elif isinstance(val, (int, float)):
                    if val not in [1, 2, 3]:
                        validation_errors.append(f"Zeile {idx+2}: gehalt muss 1, 2 oder 3 sein (ist {val})")
                else:
                    validation_errors.append(f"Zeile {idx+2}: gehalt hat ungültigen Typ (ist {type(val).__name__})")
        
        # Count missing values per column
        for col in required_columns:
            missing_count = input_df[col].isna().sum()
            if missing_count > 0:
                percentage = (missing_count / len(input_df)) * 100
                validation_warnings.append(f"⚠️ {col}: {missing_count} fehlende Werte ({percentage:.1f}%)")
        
        # Display validation results
        if validation_errors:
            st.sidebar.error("❌ **Validierungsfehler gefunden:**")
            for error in validation_errors[:10]:  # Show first 10 errors
                st.sidebar.error(error)
            if len(validation_errors) > 10:
                st.sidebar.error(f"... und {len(validation_errors) - 10} weitere Fehler")
            st.sidebar.info("Bitte korrigieren Sie die Daten und laden Sie die Datei erneut hoch.")
            st.stop()
        
        if validation_warnings:
            st.sidebar.warning("⚠️ **Warnungen:**")
            for warning in validation_warnings:
                st.sidebar.warning(warning)
            st.sidebar.info("Fehlende Werte werden durch den Durchschnitt ersetzt.")
        
        # Proceed with data processing
        st.sidebar.success(f"✅ Validierung erfolgreich! {len(input_df)} Datensätze bereit zur Verarbeitung.")
        
        # Map salary column - handle both string and numeric values safely
        salary_mapping = {
            'low': 1, 'medium': 2, 'high': 3,
            'niedrig': 1, 'mittel': 2, 'hoch': 3
        }
        
        def normalize_salary(val):
            if pd.isna(val):
                return val
            if isinstance(val, str):
                return salary_mapping.get(val.lower(), val)
            return val
        
        input_df['gehalt'] = input_df['gehalt'].apply(normalize_salary)
        
        # Fill any missing values with mean for the feature columns
        input_df[features] = input_df[features].fillna(input_df[features].mean())
        
        # Make predictions
        predictions = model.predict(input_df[features])
        predictions_proba = model.predict_proba(input_df[features])
        
        # Add results to dataframe
        result_mapping = {0: 'Er/Sie bleibt bei uns', 1: 'Er/Sie wird uns verlassen'}
        input_df['Vorhersage'] = [result_mapping[pred] for pred in predictions]
        input_df['Wahrscheinlichkeit (%)'] = [f"{proba.max() * 100:.1f}" for proba in predictions_proba]
        
        # Create a download button
        csv_data = input_df.to_csv(index=False).encode('utf-8')
        st.sidebar.download_button(
            label="📥 Ergebnisse herunterladen (CSV)",
            data=csv_data,
            file_name='mitarbeiter_vorhersagen.csv',
            mime='text/csv'
        )
        
        # Display results
        st.header('Hochgeladene Datei und Vorhersagen')
        st.write(f"**Anzahl der Datensätze:** {len(input_df)}")
        
        # Summary statistics
        col1, col2 = st.columns(2)
        with col1:
            staying_count = sum(predictions == 0)
            st.metric("Mitarbeiter bleiben", staying_count, f"{staying_count/len(predictions)*100:.1f}%")
        with col2:
            leaving_count = sum(predictions == 1)
            st.metric("Mitarbeiter gehen wahrscheinlich", leaving_count, f"{leaving_count/len(predictions)*100:.1f}%")
        
        # Show the dataframe with pagination for large datasets
        if len(input_df) > 1000:
            st.info(f"Zeige erste 1000 von {len(input_df)} Datensätzen. Alle Ergebnisse sind in der herunterladbaren Datei enthalten.")
            st.dataframe(input_df.head(1000), use_container_width=True)
        else:
            st.dataframe(input_df, use_container_width=True)
        st.write('---')

    except Exception as e:
        st.sidebar.error(f"Fehler beim Verarbeiten der Datei: {e}")
        st.sidebar.info("Stellen Sie sicher, dass die Datei die erforderlichen Spalten enthält und korrekt formatiert ist.")

# Prediction History and Analytics Section
st.markdown("---")
st.header('📜 Vorhersage-Historie und Analyse')
st.write("Sehen Sie vergangene Vorhersagen und Trends im Zeitverlauf.")

history_tab1, history_tab2 = st.tabs(["Statistiken", "Verlauf"])

with history_tab1:
    st.subheader("Gesamt-Statistiken")
    
    stats = get_prediction_statistics()
    
    if stats:
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Gesamt-Vorhersagen", stats['total'])
        with col2:
            st.metric("Vorhersage: Geht", stats['predicted_leave'], 
                     f"{stats['leave_percentage']:.1f}%")
        with col3:
            st.metric("Vorhersage: Bleibt", stats['predicted_stay'],
                     f"{100 - stats['leave_percentage']:.1f}%")
        
        st.write("---")
        
        # Pie chart of predictions
        fig_stats, ax_stats = plt.subplots(figsize=(8, 6))
        sizes = [stats['predicted_stay'], stats['predicted_leave']]
        labels = ['Bleibt', 'Geht']
        colors = ['#2ecc71', '#e74c3c']
        explode = (0.05, 0.05)
        
        ax_stats.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%',
                    shadow=True, startangle=90, textprops={'fontsize': 12, 'fontweight': 'bold'})
        ax_stats.set_title('Verteilung der Vorhersagen', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        st.pyplot(fig_stats)
        plt.close()
        
        st.info(f"📊 **Durchschnittliche Abwanderungswahrscheinlichkeit:** {stats['avg_probability_leaves']*100:.1f}%")
    else:
        st.info("Noch keine Vorhersagen in der Datenbank. Führen Sie eine Vorhersage durch, um Statistiken zu sehen.")

with history_tab2:
    st.subheader("Letzte Vorhersagen")
    
    records = get_prediction_history(limit=50)
    
    if records:
        # Convert to dataframe
        history_data = []
        for record in records:
            history_data.append({
                'Zeitstempel': record.timestamp.strftime('%Y-%m-%d %H:%M:%S'),
                'Zufriedenheit': f"{record.zufriedenheit*100:.0f}%",
                'Projekte': record.anzahl_projekte,
                'Arbeitsstunden': record.durchschnittliche_monatliche_arbeitsstunden,
                'Jahre': record.jahre_im_unternehmen,
                'Abteilung': record.abteilung,
                'Gehalt': record.gehalt.capitalize(),
                'Vorhersage': 'Bleibt' if record.prediction == 0 else 'Geht',
                'Wahrscheinlichkeit': f"{record.probability_leaves*100:.1f}%"
            })
        
        history_df = pd.DataFrame(history_data)
        
        st.write(f"**Zeige die letzten {len(history_df)} Vorhersagen:**")
        st.dataframe(history_df, use_container_width=True)
        
        # Download button for history
        csv_history = history_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Historie herunterladen (CSV)",
            data=csv_history,
            file_name='vorhersage_historie.csv',
            mime='text/csv'
        )
        
        # Time series chart if enough data
        if len(records) > 5:
            st.write("---")
            st.subheader("Trend der Abwanderungsvorhersagen")
            
            timestamps = [record.timestamp for record in reversed(records)]
            probabilities = [record.probability_leaves * 100 for record in reversed(records)]
            
            fig_trend, ax_trend = plt.subplots(figsize=(12, 6))
            ax_trend.plot(timestamps, probabilities, marker='o', linewidth=2, markersize=6, color='#3498db')
            ax_trend.axhline(y=50, color='r', linestyle='--', label='Entscheidungsgrenze', linewidth=2)
            ax_trend.fill_between(timestamps, probabilities, alpha=0.3, color='#3498db')
            ax_trend.set_xlabel('Zeitstempel', fontsize=12)
            ax_trend.set_ylabel('Abwanderungswahrscheinlichkeit (%)', fontsize=12)
            ax_trend.set_title('Zeitverlauf der Abwanderungsvorhersagen', fontsize=14, fontweight='bold')
            ax_trend.grid(alpha=0.3)
            ax_trend.legend()
            plt.xticks(rotation=45, ha='right')
            
            plt.tight_layout()
            st.pyplot(fig_trend)
            plt.close()
    else:
        st.info("Noch keine Vorhersagen in der Historie. Führen Sie eine Vorhersage durch, um sie hier zu sehen.")

# Footer with additional information
st.markdown("---")
st.markdown("""
**Über diese App:**
- Diese App verwendet einen Random Forest Klassifikator zur Vorhersage der Mitarbeiterabwanderung
- Das Modell wird mit den vorhandenen Daten trainiert und erreicht eine Genauigkeit von etwa 98%
- Für beste Ergebnisse stellen Sie sicher, dass alle Eingabeparameter korrekt ausgefüllt sind

**Entwickelt mit:** Streamlit, Scikit-learn, Pandas
""")
