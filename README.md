# Fake Profile Detector for Social Media

An AI-powered tool to detect fake/bot accounts on social media platforms.

## Features

- Machine learning model trained to identify fake profiles based on various signals
- Support for analyzing Twitter/X, Instagram, and Facebook profiles
- User-friendly web interface for quick analysis
- Detailed reports with confidence scores and reasoning
- Batch processing for analyzing multiple accounts simultaneously

## Demo

![Demo Screenshot](screenshots/demo.png)

## Installation

### Prerequisites

- Python 3.8+
- pip (Python package installer)

### Setup Instructions

1. **Clone the repository**

```bash
git clone https://github.com/Latex999/fake-profile-detector.git
cd fake-profile-detector
```

2. **Create a virtual environment (recommended)**

```bash
# On Windows
python -m venv venv
venv\Scripts\activate

# On macOS/Linux
python -m venv venv
source venv/bin/activate
```

3. **Install dependencies**

```bash
pip install -r requirements.txt
```

4. **Set up environment variables**

Create a `.env` file in the root directory with the following variables (or set them in your environment):

```
# Required for the web interface
FLASK_SECRET_KEY=your_random_secret_key

# Optional: Social media API credentials
# These are optional and depend on which platforms you want to analyze
TWITTER_API_KEY=your_twitter_api_key
TWITTER_API_SECRET=your_twitter_api_secret
INSTAGRAM_ACCESS_TOKEN=your_instagram_access_token
FACEBOOK_APP_ID=your_facebook_app_id
FACEBOOK_APP_SECRET=your_facebook_app_secret
```

5. **Run the application**

```bash
python app.py
```

The application will be available at `http://127.0.0.1:5000/`

## Usage

1. Open the web interface at `http://127.0.0.1:5000/`
2. Enter the social media profile URL or username you want to analyze
3. Select the platform (Twitter/X, Instagram, or Facebook)
4. Click "Analyze Profile"
5. View the detailed results, including:
   - Fake account probability score
   - Key indicators that influenced the decision
   - Suggested actions

## How It Works

The fake profile detector uses a combination of techniques:

1. **Feature Extraction**: Analyzes profile information, activity patterns, and engagement metrics
2. **Natural Language Processing**: Evaluates post content, comment sentiment, and linguistic patterns
3. **Network Analysis**: Examines follower/following relationships and interaction networks
4. **Image Analysis**: Checks profile pictures for signs of AI generation or stock photos
5. **Machine Learning Model**: Combines all factors to calculate a probability score

## Limitations

- Requires public profiles or appropriate API access
- Analysis accuracy depends on profile completeness
- Some platforms may have rate limiting that affects analysis speed
- The model is regularly updated but may not detect the newest fake account techniques

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgements

- [scikit-learn](https://scikit-learn.org/) for machine learning components
- [Hugging Face Transformers](https://huggingface.co/) for NLP analysis
- [Flask](https://flask.palletsprojects.com/) for the web interface