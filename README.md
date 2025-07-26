# Anomaly Detection System (ADS)

This project implements a comprehensive Anomaly Detection System using machine learning models to detect outliers and anomalies in various types of data. The system provides both a command-line interface and a graphical user interface for data analysis, model training, and real-time anomaly detection.

## Features

- Multiple machine learning models (Logistic Regression, Decision Tree, Random Forest, Gradient Boosting)
- Real-time data streaming and anomaly detection
- Interactive GUI with live visualization
- Ensemble prediction with voting mechanism
- Comprehensive model analysis and reporting
- Data preprocessing and feature engineering
- Support for various data types (numerical, categorical, time-series)
- Configurable anomaly thresholds and sensitivity

## Prerequisites

- Python 3.7 or higher
- pip (Python package installer)

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd <repository-directory>
```

2. Create and activate a virtual environment (recommended):
```bash
python -m venv venv
# On Windows
venv\Scripts\activate
# On Unix or MacOS
source venv/bin/activate
```

3. Install required packages:
```bash
pip install -r requirements.txt
```

## Project Structure

```
├── app.py              # Core application logic and ML models
├── ads_gui.py          # GUI implementation for Anomaly Detection
├── requirements.txt    # Python dependencies
├── models/            # Directory for saved ML models
├── training/          # Training data directory
└── testing/           # Testing data directory
```

## Usage

### Starting the Application

Run the GUI application:
```bash
python ads_gui.py
```

### Using the GUI

1. **Data Loading**
   - Click "Select Training CSV" to choose your training dataset
   - Click "Select Testing CSV" to choose your testing dataset
   - Click "Load & Preprocess Data" to load and prepare the data

2. **Model Training**
   - After loading data, click "Train Models" to train all available models
   - The system will train: Logistic Regression, Decision Tree, Random Forest, and Gradient Boosting models

3. **Real-time Analysis**
   - Click "Start Streaming" to begin real-time anomaly detection
   - The system will process test data one sample at a time
   - Live visualizations will update showing:
     - Cumulative counts of normal vs anomalous data points
     - Anomaly rate over time
     - Normal/Anomaly classification with confidence scores

4. **Full Analysis**
   - Click "Full Analysis" to get comprehensive performance metrics for all models
   - Results include precision, recall, F1-score, and support for each model

### Visualizations

The GUI provides three real-time visualizations:
1. **Cumulative Count**: Bar chart showing total normal vs anomalous data points
2. **Anomaly Rate**: Line plot showing the rate of anomalies over time
3. **Classification Plot**: Shows individual classifications with confidence levels

## Data Types Supported

The system can detect anomalies in various types of data:
- **Network Traffic**: Unusual patterns in network communications
- **Financial Transactions**: Fraudulent or suspicious financial activities
- **System Logs**: Abnormal system behavior or security threats
- **Sensor Data**: Environmental or IoT sensor anomalies
- **User Behavior**: Unusual user activity patterns
- **Custom Datasets**: Any labeled dataset with binary classification (normal/anomaly)

## Development

### Adding New Models

To add a new model:
1. Import the model in `app.py`
2. Add it to the `initialize_models()` function
3. The GUI will automatically include the new model in training and analysis

### Data Format

The system expects CSV files with the following requirements:
- Must include a 'label' column (0 for normal, 1 for anomaly)
- Should not include irrelevant identifier columns
- Numeric features will be automatically scaled
- Categorical features will be automatically encoded
- Missing values will be handled during preprocessing

### Customizing Preprocessing

The preprocessing pipeline can be modified in the `preprocess_data()` function in `app.py`:
- Feature scaling and normalization
- Categorical encoding
- Outlier detection and handling
- Feature selection and engineering
- Data augmentation for imbalanced datasets

### Anomaly Detection Algorithms

The system supports multiple anomaly detection approaches:
- **Supervised Learning**: Uses labeled data to train classification models
- **Ensemble Methods**: Combines predictions from multiple models
- **Threshold-based Detection**: Configurable sensitivity levels
- **Probability-based Scoring**: Confidence levels for each prediction

## Troubleshooting

Common issues and solutions:

1. **Data Loading Errors**
   - Ensure CSV files are properly formatted
   - Check for required 'label' column
   - Verify file paths are correct

2. **Model Training Issues**
   - Check available system memory
   - Verify data preprocessing steps
   - Ensure all required packages are installed

3. **GUI Display Problems**
   - Update PyQt5 to the latest version
   - Check system display settings
   - Verify matplotlib backend configuration

4. **Performance Issues**
   - Reduce dataset size for initial testing
   - Consider feature selection for high-dimensional data
   - Adjust streaming interval for real-time analysis

## Best Practices

- **Data Quality**: Ensure training data is representative and well-labeled
- **Feature Engineering**: Select relevant features for your specific use case
- **Model Selection**: Test different algorithms for your data type
- **Threshold Tuning**: Adjust anomaly thresholds based on business requirements
- **Regular Retraining**: Update models with new data periodically

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

[Specify your license here]

## Contact

[Your contact information] 