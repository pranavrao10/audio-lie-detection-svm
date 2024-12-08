# Speech Deception with Machine Learning
An ML Model trained on a Support Vector Machine (SVM), takes an audio signal as an input, and outputs whether the narrated story is true or false.

### Overview
This project uses machine learning techniques to detect whether a narrated 30-second audio recording is truthful or deceptive. By extracting key audio features such as MFCCs, chroma features, pitch, and spectral properties, and leveraging a Support Vector Machine (SVM) classifier with stratified k-fold cross-validation, the model achieves robust performance metrics.

### Key Features
• Extracts prosodic and spectral audio features using Python's librosa library.

• Implements stratified k-fold cross-validation for robust evaluation.

• Uses hyperparameter-tuned SVM to classify truthful vs. deceptive audio recordings.

• Evaluates performance using metrics like accuracy, precision, recall, F1 score, AUC-ROC, and confusion matrices.

### Performance
Accuracy:	83%
Mean Accuracy: 48%
AUC-ROC:	87.96%
Precision:	83%
Recall:	83%
F1-Score:	83%

##### The confusion matrix shows the following results:
• 44 true stories correctly classified as true.

• 39 deceptive stories correctly classified as false.

• 11 false positives (deceptive stories classified as true).

• 6 false negatives (true stories classified as deceptive).


### Data
The dataset consists of:

• 100 audio files in .wav format, each containing a 30-second narrated story.

• Metadata specifying whether each story is truthful or deceptive.


### How It Works

Pipeline

1. Preprocessing:
   
	• Load audio files and resample to 16kHz.

	• Standardize audio lengths to 30 seconds.
 
2. Feature Extraction:
    
	• Extract features like MFCCs, deltas, chroma, pitch, spectral centroid, zero-crossing rate, RMS energy, and energy variance.
 
3. Modeling:
   
   	• Train an SVM classifier using stratified k-fold cross-validation.
   
   	• Optimize hyperparameters with GridSearchCV.
 
4. Evaluation:
   
   	• Use metrics such as accuracy, AUC-ROC, and confusion matrix to assess performance.


### Setup and Installation

To run this project locally, follow these steps:

	1. Clone the repository:
         	git clone https://github.com/yourusername/audio-lie-detection.git
         	cd audio-lie-detection
  	2. Install dependencies:
        	Create a virtual environment and install required Python packages.
        	python -m venv env
        	source env/bin/activate   # On Windows: env\Scripts\activate
        	pip install -r requirements.txt
	3. Prepare the dataset:
		• Place your .wav audio files in the data/audio directory.
		• Place the corresponding metadata file (metadata.csv) in the data directory.
	4. Run the project:
	        • Open the notebook file from the server interface.
		• Follow the instructions provided in the notebook to run the code cells sequentially.


### Requirements

• Python 3.8 or higher

• Libraries:

	• librosa: Audio feature extraction
	• scikit-learn: Machine learning models and metrics
	• numpy: Numerical computations
	• pandas: Data manipulation and analysis
	• matplotlib: Visualizations


### Results and Insights

• Key features influencing the model include:

	• MFCCs and their deltas
	• Spectral contrast
	• Pitch variations
	• Zero-crossing rate and RMS energy

• Limitations

	• The small dataset (100 samples) limits the model’s ability to generalize to diverse or real-world scenarios.
	• Performance may vary with different accents, recording environments, or speech styles.

• Future Work

	• Data Augmentation: Add noise, pitch shifts, and time stretching to expand the dataset.
	• Feature Enhancement: Incorporate linguistic and prosodic features.
	• Model Exploration: Test ensemble methods like Random Forests or Gradient Boosting.
	• Real-World Deployment: Adapt the system for live audio classification.


### Contributors

• Pranav Rao (@pranavrao10)
        
