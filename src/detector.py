import os
import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Tuple, Optional, Union
import joblib
import csv
from datetime import datetime

from src.features.content_analyzer import ContentAnalyzer
from src.features.activity_analyzer import ActivityAnalyzer
from src.features.image_analyzer import ImageAnalyzer
from src.features.network_analyzer import NetworkAnalyzer
from src.models.model_loader import load_model
from src.utils.feature_extractor import FeatureExtractor

# Configure logging
logger = logging.getLogger(__name__)

class FakeProfileDetector:
    """
    Main class for detecting fake profiles on social media platforms.
    
    This class orchestrates the entire detection process by:
    1. Extracting features from profile data
    2. Analyzing different aspects (content, activity, network, images)
    3. Running the ML model to predict if a profile is fake
    4. Generating detailed reports with confidence scores
    """
    
    def __init__(self, model_path: str = 'models/fake_profile_detector.pkl'):
        """
        Initialize the detector with pre-trained models and analyzers.
        
        Args:
            model_path: Path to the pre-trained ML model
        """
        logger.info("Initializing FakeProfileDetector")
        
        # Load the ML model if it exists
        self.model = None
        try:
            self.model = load_model(model_path)
            logger.info(f"Loaded model from {model_path}")
        except FileNotFoundError:
            logger.warning(f"Model not found at {model_path}, will use heuristic detection")
            
        # Initialize the analyzers
        self.content_analyzer = ContentAnalyzer()
        self.activity_analyzer = ActivityAnalyzer()
        self.image_analyzer = ImageAnalyzer()
        self.network_analyzer = NetworkAnalyzer()
        
        # Initialize the feature extractor
        self.feature_extractor = FeatureExtractor()
        
        logger.info("FakeProfileDetector initialized successfully")
    
    def analyze_profile(self, profile_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze a single profile and determine if it's fake.
        
        Args:
            profile_data: Dictionary containing profile information
            
        Returns:
            Dictionary with analysis results including:
            - is_fake: Boolean indicating if profile is likely fake
            - probability: Confidence score (0-1)
            - indicators: List of suspicious indicators found
            - feature_importance: Which features contributed most to the decision
            - recommendations: Suggested actions
        """
        logger.info(f"Analyzing profile: {profile_data.get('username', 'Unknown')}")
        
        try:
            # Extract features from the profile data
            features = self.feature_extractor.extract_features(profile_data)
            
            # Run specialized analyzers
            content_analysis = self.content_analyzer.analyze(profile_data)
            activity_analysis = self.activity_analyzer.analyze(profile_data)
            image_analysis = self.image_analyzer.analyze(profile_data)
            network_analysis = self.network_analyzer.analyze(profile_data)
            
            # Combine all analyses
            combined_features = {
                **features,
                **content_analysis,
                **activity_analysis,
                **image_analysis,
                **network_analysis
            }
            
            # Prepare features for the model
            feature_vector = self._prepare_feature_vector(combined_features)
            
            # Make prediction using the model or fallback to heuristic
            if self.model:
                probability = self.model.predict_proba([feature_vector])[0][1]
                is_fake = probability >= 0.7  # Threshold can be adjusted
                feature_importance = self._get_feature_importance(feature_vector)
            else:
                # Fallback to heuristic analysis if no model is available
                probability, is_fake, feature_importance = self._heuristic_detection(combined_features)
            
            # Generate list of suspicious indicators
            indicators = self._identify_suspicious_indicators(combined_features, feature_importance)
            
            # Generate recommendations based on the analysis
            recommendations = self._generate_recommendations(is_fake, probability, indicators)
            
            # Prepare the final result
            result = {
                'is_fake': is_fake,
                'probability': probability,
                'indicators': indicators,
                'feature_importance': feature_importance,
                'recommendations': recommendations,
                'profile_data': profile_data,
                'timestamp': datetime.now().isoformat()
            }
            
            logger.info(f"Analysis complete for {profile_data.get('username', 'Unknown')}. "
                      f"Result: {'Fake' if is_fake else 'Real'} (Probability: {probability:.2f})")
            
            return result
            
        except Exception as e:
            logger.error(f"Error analyzing profile: {str(e)}", exc_info=True)
            raise
    
    def batch_analyze(self, file_path: str, platform: str) -> List[Dict[str, Any]]:
        """
        Process a batch of profiles from a CSV or TXT file.
        
        Args:
            file_path: Path to file containing profile URLs or usernames
            platform: Social media platform name
            
        Returns:
            List of analysis results for each profile
        """
        logger.info(f"Starting batch analysis from {file_path} for platform {platform}")
        
        results = []
        profiles_processed = 0
        
        try:
            # Read the input file
            if file_path.endswith('.csv'):
                with open(file_path, 'r', encoding='utf-8') as f:
                    reader = csv.reader(f)
                    # Skip header if exists
                    header = next(reader, None)
                    profiles = [row[0] for row in reader if row]
            else:
                with open(file_path, 'r', encoding='utf-8') as f:
                    profiles = [line.strip() for line in f if line.strip()]
            
            # Process each profile
            from src.utils.data_processor import process_profile_url
            for profile in profiles:
                try:
                    profile_data = process_profile_url(profile, platform)
                    result = self.analyze_profile(profile_data)
                    results.append(result)
                    profiles_processed += 1
                    logger.info(f"Processed {profiles_processed}/{len(profiles)} profiles")
                except Exception as e:
                    logger.error(f"Error processing profile {profile}: {str(e)}")
                    results.append({
                        'error': str(e),
                        'profile': profile,
                        'platform': platform,
                        'timestamp': datetime.now().isoformat()
                    })
            
            # Save batch results to a CSV file
            output_file = os.path.join('results', f'batch_results_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv')
            os.makedirs('results', exist_ok=True)
            self._save_batch_results(results, output_file)
            
            logger.info(f"Batch analysis completed. Processed {profiles_processed}/{len(profiles)} profiles")
            
            return results
            
        except Exception as e:
            logger.error(f"Error in batch analysis: {str(e)}", exc_info=True)
            raise
    
    def _prepare_feature_vector(self, features: Dict[str, Any]) -> np.ndarray:
        """Convert the feature dictionary to a vector for the ML model."""
        # This would be implemented based on the model's expected features
        # For simplicity, we'll just use a basic implementation
        feature_list = [
            features.get('account_age_days', 0),
            features.get('posts_per_day', 0),
            features.get('followers_count', 0),
            features.get('following_count', 0),
            features.get('followers_to_following_ratio', 0),
            features.get('profile_pic_score', 0),
            features.get('bio_length', 0),
            features.get('has_external_url', 0),
            features.get('sentiment_score', 0),
            features.get('content_diversity', 0),
            features.get('engagement_rate', 0),
            features.get('posting_regularity', 0),
            features.get('suspicious_content_score', 0),
            features.get('network_isolation_score', 0)
        ]
        return np.array(feature_list)
    
    def _get_feature_importance(self, feature_vector: np.ndarray) -> Dict[str, float]:
        """Calculate the importance of each feature in the prediction."""
        if not self.model:
            return {}
            
        # Get feature importances from the model
        # This would be implemented based on the specific model type
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
        
        # Mock implementation - in a real system, this would use model-specific code
        # to extract feature importances
        try:
            importances = getattr(self.model, 'feature_importances_', None)
            if importances is None:
                # For models without feature_importances_ attribute
                importances = np.ones(len(feature_names)) / len(feature_names)
            
            return {name: float(imp) for name, imp in zip(feature_names, importances)}
        except:
            # Fallback to equal importance if can't extract from model
            return {name: 1.0/len(feature_names) for name in feature_names}
    
    def _heuristic_detection(self, features: Dict[str, Any]) -> Tuple[float, bool, Dict[str, float]]:
        """
        Fallback detection method when ML model isn't available.
        Uses rule-based heuristics to determine if a profile is fake.
        
        Returns:
            Tuple of (probability, is_fake, feature_importance)
        """
        # Define red flags (suspicious indicators)
        red_flags = {
            'new_account': features.get('account_age_days', 365) < 30,
            'high_following_ratio': features.get('followers_to_following_ratio', 1) < 0.1,
            'excessive_posting': features.get('posts_per_day', 2) > 20,
            'suspicious_profile_pic': features.get('profile_pic_score', 0) > 0.7,
            'low_engagement': features.get('engagement_rate', 0.05) < 0.01,
            'suspicious_content': features.get('suspicious_content_score', 0) > 0.7,
            'isolated_network': features.get('network_isolation_score', 0) > 0.7,
            'duplicate_content': features.get('content_diversity', 1) < 0.3,
            'no_bio': features.get('bio_length', 10) < 5
        }
        
        # Calculate probability based on number of red flags
        red_flag_count = sum(1 for flag in red_flags.values() if flag)
        probability = min(0.3 + (red_flag_count / len(red_flags)) * 0.7, 1.0)
        is_fake = probability >= 0.7
        
        # Calculate importance based on which flags were triggered
        importance = {}
        for flag_name, is_triggered in red_flags.items():
            importance[flag_name] = 0.1 if is_triggered else 0.0
            
        return probability, is_fake, importance
    
    def _identify_suspicious_indicators(self, features: Dict[str, Any], 
                                      importance: Dict[str, float]) -> List[Dict[str, Any]]:
        """
        Generate list of suspicious indicators based on features and their importance.
        
        Returns:
            List of dictionaries with indicator name, description, and severity
        """
        indicators = []
        
        # Check for suspicious indicators
        # Account age
        if features.get('account_age_days', 365) < 30:
            indicators.append({
                'name': 'New Account',
                'description': 'Account was created recently',
                'severity': 'medium'
            })
            
        # Following to follower ratio
        if features.get('followers_to_following_ratio', 1) < 0.1:
            indicators.append({
                'name': 'Following/Follower Imbalance',
                'description': 'Account follows many users but has few followers',
                'severity': 'high'
            })
            
        # Posting frequency
        if features.get('posts_per_day', 2) > 20:
            indicators.append({
                'name': 'Excessive Posting',
                'description': 'Account posts with unusually high frequency',
                'severity': 'medium'
            })
            
        # Profile picture
        if features.get('profile_pic_score', 0) > 0.7:
            indicators.append({
                'name': 'Suspicious Profile Picture',
                'description': 'Profile picture shows signs of being AI-generated or stock photo',
                'severity': 'high'
            })
            
        # Engagement rate
        if features.get('engagement_rate', 0.05) < 0.01:
            indicators.append({
                'name': 'Low Engagement',
                'description': 'Posts receive very little engagement relative to follower count',
                'severity': 'medium'
            })
            
        # Content diversity
        if features.get('content_diversity', 1) < 0.3:
            indicators.append({
                'name': 'Repetitive Content',
                'description': 'Account posts very similar content repeatedly',
                'severity': 'medium'
            })
            
        # Suspicious content
        if features.get('suspicious_content_score', 0) > 0.7:
            indicators.append({
                'name': 'Suspicious Content',
                'description': 'Content contains patterns associated with fake accounts',
                'severity': 'high'
            })
            
        # Network isolation
        if features.get('network_isolation_score', 0) > 0.7:
            indicators.append({
                'name': 'Isolated Network',
                'description': 'Account has minimal interaction with legitimate accounts',
                'severity': 'high'
            })
            
        # More indicators can be added based on other features
            
        return indicators
    
    def _generate_recommendations(self, is_fake: bool, probability: float, 
                               indicators: List[Dict[str, Any]]) -> List[str]:
        """Generate recommendations based on the analysis results."""
        recommendations = []
        
        if is_fake and probability > 0.9:
            recommendations.append("This profile is highly likely to be fake. Consider blocking and reporting.")
        elif is_fake:
            recommendations.append("This profile shows several suspicious patterns. Exercise caution when interacting.")
        elif probability > 0.4:
            recommendations.append("This profile shows some suspicious patterns but may be legitimate. Verify before engaging.")
        else:
            recommendations.append("This profile appears to be legitimate based on our analysis.")
            
        # Add specific recommendations based on indicators
        if any(i['name'] == 'New Account' for i in indicators):
            recommendations.append("This is a new account. Consider waiting for a longer activity history before engaging.")
            
        if any(i['name'] == 'Suspicious Profile Picture' for i in indicators):
            recommendations.append("Verify the profile picture by performing a reverse image search.")
            
        if any(i['name'] == 'Suspicious Content' for i in indicators):
            recommendations.append("Review the content posted by this account carefully before engaging.")
            
        return recommendations
    
    def _save_batch_results(self, results: List[Dict[str, Any]], output_file: str) -> None:
        """Save batch analysis results to a CSV file."""
        try:
            # Create a DataFrame from the results
            df = pd.DataFrame([{
                'username': r.get('profile_data', {}).get('username', 'Unknown'),
                'platform': r.get('profile_data', {}).get('platform', 'Unknown'),
                'is_fake': r.get('is_fake', None),
                'probability': r.get('probability', None),
                'indicators_count': len(r.get('indicators', [])),
                'top_indicator': r.get('indicators', [{}])[0].get('name', 'None') if r.get('indicators') else 'None',
                'url': r.get('profile_data', {}).get('url', ''),
                'error': r.get('error', '')
            } for r in results])
            
            # Save to CSV
            df.to_csv(output_file, index=False)
            logger.info(f"Batch results saved to {output_file}")
        except Exception as e:
            logger.error(f"Error saving batch results: {str(e)}", exc_info=True)