# Speech Emotion Recognition System

## Group Members
- Hrisheek: Ravdess data analysis and CNN model training
- Likhitha: Crema dataset analysis and LSTM model training
- Himaswetha: Savee data analysis and frontend UI development

## Abstract (Focus Area)
This work presents a deep learning approach for recognizing emotions from speech using multiple state-of-the-art neural network architectures. Given the widespread applications of emotion recognition in areas like human-computer interaction, customer service analysis, and mental health monitoring, accurate emotion classification from audio data is an important problem. Our research focuses on developing robust models that can generalize well across diverse speakers, languages, and recording conditions.

Our proposed model leverages convolutional neural networks (CNN) that take audio features as input and learns discriminative representations for emotion classification. We explore advanced feature extraction techniques including MFCC, zero-crossing rate, and RMS energy, combined with data augmentation methods to improve model robustness. The system achieves 97.25% accuracy on emotion classification tasks across multiple datasets.

## Dataset
We utilize multiple standard speech emotion datasets for comprehensive training and evaluation:

1. **RAVDESS Dataset**
   - Professional actors performing emotions
   - 7 distinct emotion categories
   - Controlled recording conditions

2. **CREMA-D Dataset**
   - 7,442 original clips
   - 91 actors (48 male, 43 female)
   - Age range: 20-74 years
   - Diverse racial/ethnic backgrounds

3. **TESS Dataset**
   - 2,800 audio files
   - Two female actresses (26 and 64 years)
   - Seven emotion categories

4. **SAVEE Dataset**
   - Four male speakers
   - Seven emotion categories
   - Multiple sentence types

## Requirements

### Hardware Requirements
- GPU with >=8GB VRAM (Recommended)
- High-end CPU (Intel i7/AMD Ryzen 7 or better)
- 16GB+ RAM
- Sufficient Storage (SSD preferred)

### Software Requirements
- Ubuntu 18.04+ or Windows 10
- Python 3.8+
- PyTorch 1.8+
- Libraries:
  - librosa
  - numpy
  - pandas
  - streamlit
  - tensorflow
  - matplotlib

## Methodology

### Model Architecture
- Multiple convolutional layers with batch normalization
- MaxPooling layers for feature reduction
- Dropout layers for preventing overfitting
- Dense layers for final classification

### Feature Extraction
- MFCC (Mel-frequency cepstral coefficients)
- Zero-crossing rate
- Root Mean Square Energy
- Data augmentation techniques:
  - Noise injection
  - Time stretching
  - Pitch shifting

### Training Process
- Batch size: 64
- Epochs: 50
- Optimizer: Adam
- Loss function: Categorical Cross-entropy
- Early stopping and learning rate reduction

## Results

### Result Interpretations
The model achieved state-of-the-art performance with:
- 97.25% accuracy on emotion classification
- Robust performance across different speakers and recording conditions
- Real-time inference capability through web interface

### Performance Metrics
- Classification accuracy: 97.25%
- Low false positive rate across emotion categories
- Consistent performance across datasets

## Future Work
- Integration with more languages and accents
- Real-time emotion tracking over time
- Deployment optimizations for mobile devices
- Enhancement of the web interface
- Integration with video emotion recognition

## Roles and Responsibilities

### Hrisheek
- Ravdess dataset analysis
- CNN model architecture design
- Model training and optimization

### Likhitha
- Crema dataset processing
- LSTM model implementation
- Performance evaluation

### Himaswetha
- Savee dataset analysis
- Frontend UI development
- Integration testing

## References
1. RAVDESS Dataset: https://zenodo.org/records/1188976
2. SAVEE Dataset: http://kahlan.eps.surrey.ac.uk/savee/
3. CREMA-D Dataset: https://github.com/CheyneyComputerScience/CREMA-D
4. TESS Dataset: https://tspace.library.utoronto.ca/handle/1807/24487

## Project Milestone Progress
1. Data preprocessing and feature extraction pipeline ✅
2. Model architecture implementation ✅
3. Training pipeline setup ✅
4. Web interface development ✅
5. Performance optimization ✅
6. Integration testing ✅

The project has successfully met its initial goals and demonstrates state-of-the-art performance in speech emotion recognition. Future work will focus on expanding the model's capabilities and optimizing for real-world applications.
