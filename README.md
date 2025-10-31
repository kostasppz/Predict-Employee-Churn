# Overview

This is a comprehensive German-language employee churn prediction application (Mitarbeiter-Fluktuation Vorhersage) built with Streamlit and machine learning. The application allows users to predict the likelihood of employee turnover based on various metrics including satisfaction level, number of projects, average monthly working hours, work accidents, promotions, department, and salary level. The app uses a Random Forest Classifier to make predictions and provides extensive data visualization, explainable AI features, and historical tracking.

# Recent Changes

**Date: October 18, 2025**
- Implemented comprehensive data visualization dashboard with feature importance charts and distribution plots
- Added explainable AI section with value comparisons, historical churn rates for similar profiles, and what-if sensitivity analysis
- Created PostgreSQL database integration for prediction history tracking and analytics
- Enhanced file upload validation with detailed error handling, type checking, and mixed-format support
- All features are in German to match user requirements

# User Preferences

- Preferred communication style: Simple, everyday language
- Application language: German (Deutsch)
- Focus on practical usability and clear explanations

# System Architecture

## Application Structure

**Main Application (app.py):**
- Single employee prediction interface with sidebar inputs
- Bulk CSV/Excel upload with comprehensive validation
- Data visualization dashboard
- Explainable AI prediction explanations
- Prediction history and analytics

**Database Module (database.py):**
- SQLAlchemy ORM models for prediction history
- PostgreSQL integration with graceful degradation
- Automatic prediction tracking

## Frontend Architecture

**Technology:** Streamlit web framework
- **Rationale:** Streamlit provides a simple, Python-native way to build interactive data applications
- **Layout:** Wide layout configuration for optimal data visualization
- **Server Configuration:** Configured via `.streamlit/config.toml` to bind to 0.0.0.0:5000
- **User Interface Components:**
  - Sidebar for input parameters and file upload
  - Main area divided into tabs for predictions, visualizations, and analytics
  - Interactive sliders, dropdowns, and file uploaders

## Machine Learning Pipeline

**Model:** Random Forest Classifier (scikit-learn)
- **Rationale:** Random Forest provides excellent accuracy (98%), handles mixed data types, and offers built-in feature importance
- **Training Approach:** 80/20 train-test split with fixed random state for reproducibility
- **Hyperparameters:** 100 estimators, max depth of 5 to prevent overfitting

**Features:**
- Numerical: zufriedenheitsgrad (satisfaction 0-100%), anzahl_projekte (projects 0-7), durchschnittliche_monatliche_arbeitszeit (monthly hours)
- Binary: arbeitsunfall (work accident 0/1), foerderung_letzte_5_jahre (promotion 0/1)
- Categorical: gehalt (salary: low/medium/high or 1/2/3)

**Data Preprocessing:**
- Missing value imputation using mean values
- Salary normalization supporting both English (low/medium/high) and German (niedrig/mittel/hoch) formats
- Flexible handling of both string and numeric salary encodings

## Data Visualization

**Components:**
1. **Feature Importance Chart:** Horizontal bar chart showing relative importance of each feature
2. **Prediction Distribution:** Bar chart showing model predictions across the dataset
3. **Data Distribution Plots:** Feature-specific charts (bar charts for categorical, histograms for continuous)

**Dynamic Feature Mapping:**
- Feature names automatically map to German display labels
- Charts adapt based on feature type (categorical vs continuous)

## Explainable AI Features

**Design Philosophy:** Provide transparent, scientifically honest explanations without making invalid claims about local contributions

**Components:**
1. **Value Comparison:** Shows user's input values vs dataset averages with deviation visualization
2. **Feature Importance:** Displays global model importance for each feature
3. **Historical Churn Rates:** Calculates actual churn rates for employees with similar feature values
4. **What-If Analysis:** Sensitivity analysis showing how predictions change when varying individual features

**Important Note:** After initial attempts to implement invalid local contribution methods, the final implementation uses truthful approaches that don't claim to show additive decompositions when the underlying math doesn't support it.

## Database Integration

**Technology:** PostgreSQL with SQLAlchemy ORM
- **Connection:** Uses DATABASE_URL environment variable
- **Graceful Degradation:** Application continues to function even if database is unavailable
- **Schema:** PredictionHistory table with all input features, predictions, probabilities, and timestamps

**Analytics:**
- Total prediction counts
- Predicted leave vs stay breakdown
- Average churn probability
- Time series visualization of predictions
- Downloadable prediction history (CSV)

## File Upload Validation

**Comprehensive Validation:**
1. Empty file detection
2. Missing column validation with clear format requirements
3. Empty row detection and removal
4. Type validation with try-except error handling
5. Range validation (satisfaction 0-100, projects 0-7)
6. Binary field validation (accident and promotion must be 0 or 1)
7. Salary value validation supporting mixed types and formats
8. Missing value tracking with percentage reporting

**Error Handling:**
- Detailed row-level error messages showing invalid values
- Limits error display to first 10 to avoid spam
- Clear formatting requirements shown to users
- Graceful halting via st.stop() when errors exist

# External Dependencies

## Python Libraries

- **streamlit:** Web application framework
- **pandas:** Data manipulation and CSV/Excel handling
- **matplotlib:** Plotting and visualization
- **seaborn:** Statistical data visualization
- **scikit-learn:** Machine learning (RandomForestClassifier, train_test_split, accuracy_score)
- **numpy:** Numerical computing
- **openpyxl:** Excel file handling
- **sqlalchemy:** Database ORM
- **psycopg2-binary:** PostgreSQL adapter

## Database

- **PostgreSQL:** Development database accessed via DATABASE_URL
- **Tables:** prediction_history (tracks all predictions with features and results)

## Data Sources

- **Training Data:** `modified_file.csv` (required in root directory)
  - Contains ~15K historical employee records
  - Columns: zufriedenheitsgrad, anzahl_projekte, durchschnittliche_monatliche_arbeitszeit, arbeitsunfall, foerderung_letzte_5_jahre, gehalt, left

## Deployment Configuration

- **Platform:** Replit
- **Entry Point:** app.py
- **Port:** 5000 (configured in .streamlit/config.toml)
- **Workflow:** `streamlit run app.py --server.port 5000`

# Known Limitations

1. SHAP library incompatible with Python 3.11, so alternative explainability methods were used
2. Custom model retraining feature not implemented (would require significant additional development)
3. Database operations are silent on failure to avoid error spam
4. Department field not included in sidebar inputs (uses default value)
5. Years at company field not included in sidebar inputs (uses default value)

# Future Enhancement Opportunities

1. Add department and tenure fields to single prediction interface
2. Implement custom model retraining with user-uploaded datasets
3. Add more sophisticated explainability if SHAP compatibility is resolved
4. Implement user authentication and per-user prediction tracking
5. Add export functionality for visualizations
6. Create admin dashboard for model performance monitoring

Application: https://churn-predictor-papazoglouk33.replit.app
