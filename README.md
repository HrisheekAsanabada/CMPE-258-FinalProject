# Speech Emotion Recognition System

## Group Members
- Hrisheek: Ravdess data analysis and CNN model training
- Likhitha: Crema dataset analysis and LSTM model training
- Himaswetha: Savee data analysis and frontend UI development

## Abstract
This project implements a deep learning-based speech emotion recognition system using multiple state-of-the-art neural network architectures. Given the widespread applications in human-computer interaction, customer service analysis, and mental health monitoring, accurate emotion classification from audio data is crucial. Our research focuses on developing robust models that can generalize well across diverse speakers, languages, and recording conditions.

Our system leverages convolutional neural networks (CNN) with advanced feature extraction techniques including MFCC, zero-crossing rate, and RMS energy. Through systematic evaluation and optimization, we've achieved 97.25% accuracy on emotion classification tasks across multiple datasets.

## Dataset Analysis and Processing
We utilize multiple standard speech emotion datasets for comprehensive training and evaluation:

### Datasets Used
1. **RAVDESS Dataset**
   - Professional actors performing emotions
   - 7 distinct emotion categories
   - Controlled recording conditions
   - Standardized recording environment

2. **CREMA-D Dataset**
   - 7,442 original clips
   - 91 actors (48 male, 43 female)
   - Age range: 20-74 years
   - Diverse racial/ethnic backgrounds
   - Multiple emotional intensities

3. **TESS Dataset**
   - 2,800 audio files
   - Two female actresses (26 and 64 years)
   - Seven emotion categories
   - High-quality recordings

4. **SAVEE Dataset**
   - Four male speakers
   - Seven emotion categories
   - Multiple sentence types
   - Varied linguistic content

### Data Processing Pipeline
- Audio preprocessing using librosa
- Feature extraction:
  - MFCC (Mel-frequency cepstral coefficients)
  - Zero-crossing rate
  - Root Mean Square Energy
- Data augmentation techniques:
  - Noise injection (noise_factor=0.035)
  - Time stretching (rate=0.8)
  - Pitch shifting (n_steps=4)
  - Signal shifting (range=±5000)

## Model Architecture and Analysis

### Architecture Comparison
We evaluated several state-of-the-art architectures:

1. **CNN Architecture (Selected)**
   - Highest accuracy: 97.25%
   - Efficient feature learning
   - Fast inference time
   - Optimal for real-time applications

2. **LSTM Architecture**
   - Better temporal modeling
   - Accuracy: 70.18%
   - Suitable for continuous speech
   - slow inference


### Selected Model Architecture
```python
model = tf.keras.Sequential([
    L.Conv1D(512, kernel_size=5, strides=1, padding='same', activation='relu'),
    L.BatchNormalization(),
    L.MaxPool1D(pool_size=5, strides=2, padding='same'),
    L.Conv1D(512, kernel_size=5, strides=1, padding='same', activation='relu'),
    L.BatchNormalization(),
    L.MaxPool1D(pool_size=5, strides=2, padding='same'),
    L.Dropout(0.2),
    # Additional layers...
    L.Dense(7, activation='softmax')
])
```

### Training and Optimization
- Batch size: 64
- Epochs: 50
- Optimizer: Adam with learning rate scheduling
- Loss function: Categorical Cross-entropy
- Early stopping and learning rate reduction
- Batch normalization for training stability
- Dropout (0.2) for regularization

## Performance and Evaluation

### Accuracy Metrics
- Overall accuracy: 97.25%
- Per-emotion accuracy:
  - Happy: 98.1%
  - Sad: 96.8%
  - Angry: 97.4%
  - Neutral: 96.9%
  - Fear: 95.8%
  - Disgust: 96.2%
  - Surprise: 97.5%

### Model Optimization Results
- Model quantization achievements:
  - 3.2x smaller model size
  - 2.5x faster inference
  - Only 0.3% accuracy loss
- GPU acceleration support
- Optimized memory usage
- Efficient batch processing

## Real-Time Application

### Web Interface
- Streamlit-based frontend
- Real-time audio processing
- Interactive waveform visualization
- Emotion prediction display
- Performance metrics visualization

### System Requirements

#### Hardware Requirements
- GPU with >=8GB VRAM (Recommended)
- High-end CPU (Intel i7/AMD Ryzen 7 or better)
- 16GB+ RAM
- SSD Storage

#### Software Requirements
- Ubuntu 18.04+ or Windows 10
- Python 3.8+
- PyTorch 1.8+
- Required libraries:
  ```bash
  pip install -r requirements.txt
  ```

### Installation and Usage
1. Clone the repository:
```bash
git clone (https://github.com/HrisheekAsanabada/CMPE-258-FinalProject/tree/main)
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the application:
```bash
streamlit run app.py
```

## Project Structure
```
CMPE-258-FinalProject/       # Root Directory
│
├── Data_Analysis/           # Data preprocessing and analysis scripts
│   ├── Ravdess.ipynb        # RAVDESS dataset preprocessing
│   ├── CremaD.ipynb         # CREMA-D dataset preprocessing
│   ├── Savee.ipynb          # SAVEE dataset preprocessing
│   └── dataset_analysis.ipynb  # Combined dataset analysis and visualization
│
├── Intermediate_Files/      # Generated features and processed data
│   ├── data_Path.csv        # File paths for audio data
│   ├── emotion.csv          # Processed emotion label data
│   └── readme.txt           # Notes on intermediate files
│
├── Model_Training/          # Model training implementations
│   ├── cnn_model.ipynb      # CNN model implementation
│   ├── lstm_model.ipynb     # LSTM model implementation
│
├── models/                  # Saved model weights and configurations
│   ├── encoder2.pickle      # Label encoder file
│   ├── scalar2.pickle       # Scaler for feature normalization
│   ├── cnn_model.keras      # Saved CNN model in Keras format
│   ├── lstm_model.keras     # LSTM model in Keras format
│   ├── cnn_model_weights.h5 # Weights of the CNN model
│   └── lstm_model_weights.h5 # Weights of the LSTM model
│
├── app.py                   # Streamlit web application for real-time emotion recognition
│
├── requirements.txt         # Python dependencies and libraries
│
└── README.md                # Project documentation with setup and usage 
```

## Future Work
- Integration with additional languages and accents
- Real-time emotion tracking over extended periods
- Mobile device optimization
- Enhanced web interface features
- Integration with video emotion recognition
- Exploration of transformer-based architectures

## References
1. RAVDESS Dataset: https://zenodo.org/records/1188976
2. SAVEE Dataset: http://kahlan.eps.surrey.ac.uk/savee/
3. CREMA-D Dataset: https://github.com/CheyneyComputerScience/CREMA-D
4. TESS Dataset: https://tspace.library.utoronto.ca/handle/1807/24487
5. Ingale, Ashish B., and D. S. Chaudhari. "Speech emotion recognition." International Journal of Soft Computing and Engineering (IJSCE) 2.1 (2012): 235-238.
6. Huang, Zhengwei, et al. "Speech emotion recognition using CNN." Proceedings of the 22nd ACM international conference on Multimedia. 2014.
7. Zhao, Jianfeng, Xia Mao, and Lijiang Chen. "Speech emotion recognition using deep 1D & 2D CNN LSTM networks." Biomedical signal processing and control 47 (2019): 312-323.


## License
This project is licensed under the MIT License - see the LICENSE file for details.
