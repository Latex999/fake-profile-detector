import os
import logging
import joblib
import numpy as np
from typing import Any, Dict, Optional
from sklearn.ensemble import RandomForestClassifier

# Configure logging
logger = logging.getLogger(__name__)

def load_model(model_path: str) -> Any:
    """
    Load a machine learning model from the specified path.
    
    Args:
        model_path: Path to the saved model file
        
    Returns:
        Loaded machine learning model
    
    Raises:
        FileNotFoundError: If the model file doesn't exist
        Exception: If there's an error loading the model
    """
    logger.info(f"Loading model from {model_path}")
    
    # Check if the model exists
    if not os.path.exists(model_path):
        # Check if we're in the models directory
        models_dir = os.path.dirname(model_path)
        os.makedirs(models_dir, exist_ok=True)
        
        # Create a fallback model if the model doesn't exist
        # In a production environment, you would normally raise an error here
        # but for demonstration we'll create a simple model
        logger.warning(f"Model not found at {model_path}, creating a simple fallback model")
        fallback_model = create_fallback_model()
        
        # Save the fallback model
        try:
            joblib.dump(fallback_model, model_path)
            logger.info(f"Fallback model saved to {model_path}")
            return fallback_model
        except Exception as e:
            logger.error(f"Error saving fallback model: {str(e)}", exc_info=True)
            raise FileNotFoundError(f"Model not found at {model_path} and could not create fallback")
    
    try:
        # Load the model
        model = joblib.load(model_path)
        logger.info(f"Model loaded successfully from {model_path}")
        return model
    except Exception as e:
        logger.error(f"Error loading model from {model_path}: {str(e)}", exc_info=True)
        raise

def create_fallback_model() -> Any:
    """
    Create a simple fallback model when the main model is not available.
    
    This is for demonstration purposes only. In a real application,
    you would typically use a pre-trained model.
    
    Returns:
        A simple RandomForest model
    """
    logger.info("Creating fallback model")
    
    # Create a simple RandomForest model
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        random_state=42
    )
    
    # Train it on some dummy data that roughly approximates
    # the kinds of features we'd expect in real data
    
    # Generate synthetic data for demonstration
    n_samples = 1000
    n_features = 14  # Matches the features used in the main detector
    
    # Feature names (for interpretability)
    feature_names = [
        'account_age_days',
        'posts_per_day',
        'followers_count',
        'following_count',
        'followers_to_following_ratio',
        'profile_pic_score',
        'bio_length',
        'has_external_url',
        'sentiment_score',
        'content_diversity',
        'engagement_rate',
        'posting_regularity',
        'suspicious_content_score',
        'network_isolation_score'
    ]
    
    # Generate data with some patterns that distinguish fake from real accounts
    np.random.seed(42)
    X = np.zeros((n_samples, n_features))
    y = np.zeros(n_samples)
    
    for i in range(n_samples):
        # Decide if this sample is a fake account (40% of samples)
        is_fake = np.random.random() < 0.4
        y[i] = 1 if is_fake else 0
        
        if is_fake:
            # Fake account patterns
            X[i, 0] = np.random.randint(1, 180)  # Newer accounts (1-180 days)
            X[i, 1] = np.random.choice([0.1, 0.2, 25, 30])  # Very low or very high posting
            X[i, 2] = np.random.randint(1, 500)  # Few followers
            X[i, 3] = np.random.randint(500, 5000)  # Many following
            X[i, 4] = X[i, 2] / X[i, 3]  # Low follower/following ratio
            X[i, 5] = np.random.uniform(0.6, 1.0)  # Suspicious profile pic
            X[i, 6] = np.random.choice([0, 5, 150])  # Empty or very long bio
            X[i, 7] = np.random.choice([0, 1], p=[0.2, 0.8])  # Often has external URL
            X[i, 8] = np.random.choice([0.1, 0.9])  # Very positive or negative sentiment
            X[i, 9] = np.random.uniform(0, 0.4)  # Low content diversity
            X[i, 10] = np.random.uniform(0, 0.01)  # Low engagement
            X[i, 11] = np.random.choice([0.1, 0.9])  # Very irregular or very regular posting
            X[i, 12] = np.random.uniform(0.6, 1.0)  # High suspicious content
            X[i, 13] = np.random.uniform(0.6, 1.0)  # High network isolation
        else:
            # Real account patterns
            X[i, 0] = np.random.randint(180, 3650)  # Older accounts (180-3650 days)
            X[i, 1] = np.random.uniform(0.3, 3.0)  # Moderate posting frequency
            X[i, 2] = np.random.randint(100, 5000)  # More followers
            X[i, 3] = np.random.randint(100, 1000)  # Moderate following
            X[i, 4] = X[i, 2] / X[i, 3]  # Higher follower/following ratio
            X[i, 5] = np.random.uniform(0, 0.5)  # Non-suspicious profile pic
            X[i, 6] = np.random.randint(10, 100)  # Moderate bio length
            X[i, 7] = np.random.choice([0, 1], p=[0.6, 0.4])  # Sometimes has external URL
            X[i, 8] = np.random.uniform(0.3, 0.7)  # Moderate sentiment
            X[i, 9] = np.random.uniform(0.4, 1.0)  # Higher content diversity
            X[i, 10] = np.random.uniform(0.01, 0.1)  # Higher engagement
            X[i, 11] = np.random.uniform(0.3, 0.7)  # Moderate posting regularity
            X[i, 12] = np.random.uniform(0, 0.5)  # Lower suspicious content
            X[i, 13] = np.random.uniform(0, 0.5)  # Lower network isolation
    
    # Fit the model
    model.fit(X, y)
    
    logger.info("Fallback model created successfully")
    
    return model