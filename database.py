import os
from datetime import datetime
from sqlalchemy import create_engine, Column, Integer, Float, String, DateTime, Boolean
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import streamlit as st

Base = declarative_base()

class PredictionHistory(Base):
    __tablename__ = 'prediction_history'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    timestamp = Column(DateTime, default=datetime.now, nullable=False)
    
    # Input features
    zufriedenheit = Column(Float, nullable=False)
    anzahl_projekte = Column(Integer, nullable=False)
    durchschnittliche_monatliche_arbeitsstunden = Column(Integer, nullable=False)
    jahre_im_unternehmen = Column(Integer, nullable=False)
    arbeitsunfall = Column(Boolean, nullable=False)
    foerderung_letzte_5_jahre = Column(Boolean, nullable=False)
    abteilung = Column(String(50), nullable=False)
    gehalt = Column(String(20), nullable=False)
    
    # Prediction results
    prediction = Column(Integer, nullable=False)  # 0 = stays, 1 = leaves
    probability_stays = Column(Float, nullable=False)
    probability_leaves = Column(Float, nullable=False)
    
    def __repr__(self):
        return f"<PredictionHistory(id={self.id}, timestamp={self.timestamp}, prediction={self.prediction})>"


@st.cache_resource
def get_database_engine():
    """Create and cache the database engine"""
    database_url = os.getenv('DATABASE_URL')
    if not database_url:
        st.warning("⚠️ DATABASE_URL nicht gefunden. Datenbank-Funktionen sind deaktiviert.")
        return None
    
    try:
        engine = create_engine(database_url, echo=False)
        return engine
    except Exception as e:
        st.error(f"Fehler beim Verbinden mit der Datenbank: {str(e)}")
        return None


def init_database():
    """Initialize the database tables"""
    try:
        engine = get_database_engine()
        if engine is None:
            return False
        Base.metadata.create_all(engine)
        return True
    except Exception as e:
        st.error(f"Fehler beim Initialisieren der Datenbank: {str(e)}")
        return False


def get_session():
    """Create a new database session"""
    engine = get_database_engine()
    if engine is None:
        return None
    Session = sessionmaker(bind=engine)
    return Session()


def save_prediction(features_dict, prediction, probabilities):
    """
    Save a prediction to the database
    
    Args:
        features_dict: Dictionary containing all input features
        prediction: The prediction result (0 or 1)
        probabilities: Array of [prob_stays, prob_leaves]
    
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        session = get_session()
        if session is None:
            return False
        
        # Create new prediction record
        prediction_record = PredictionHistory(
            zufriedenheit=features_dict['zufriedenheit'],
            anzahl_projekte=features_dict['anzahl_projekte'],
            durchschnittliche_monatliche_arbeitsstunden=features_dict['durchschnittliche_monatliche_arbeitsstunden'],
            jahre_im_unternehmen=features_dict['jahre_im_unternehmen'],
            arbeitsunfall=features_dict['arbeitsunfall'] == 1,
            foerderung_letzte_5_jahre=features_dict['foerderung_letzte_5_jahre'] == 1,
            abteilung=features_dict['abteilung'],
            gehalt=features_dict['gehalt'],
            prediction=int(prediction),
            probability_stays=float(probabilities[0]),
            probability_leaves=float(probabilities[1])
        )
        
        session.add(prediction_record)
        session.commit()
        session.close()
        return True
    except Exception as e:
        # Silent fail - just return False
        return False


def get_prediction_history(limit=100):
    """
    Retrieve prediction history from the database
    
    Args:
        limit: Maximum number of records to retrieve
    
    Returns:
        List of prediction records
    """
    try:
        session = get_session()
        if session is None:
            return []
        records = session.query(PredictionHistory).order_by(
            PredictionHistory.timestamp.desc()
        ).limit(limit).all()
        session.close()
        return records
    except Exception as e:
        return []


def get_prediction_statistics():
    """
    Get statistics about predictions
    
    Returns:
        Dictionary with statistics
    """
    try:
        session = get_session()
        if session is None:
            return None
        
        total_predictions = session.query(PredictionHistory).count()
        
        if total_predictions == 0:
            session.close()
            return None
        
        predictions_leave = session.query(PredictionHistory).filter(
            PredictionHistory.prediction == 1
        ).count()
        
        predictions_stay = total_predictions - predictions_leave
        
        avg_probability_leaves = session.query(
            PredictionHistory.probability_leaves
        ).all()
        avg_prob = sum([p[0] for p in avg_probability_leaves]) / len(avg_probability_leaves)
        
        session.close()
        
        return {
            'total': total_predictions,
            'predicted_leave': predictions_leave,
            'predicted_stay': predictions_stay,
            'avg_probability_leaves': avg_prob,
            'leave_percentage': (predictions_leave / total_predictions) * 100
        }
    except Exception as e:
        return None
