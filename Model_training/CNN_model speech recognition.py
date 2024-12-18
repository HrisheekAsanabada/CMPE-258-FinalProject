# -*- coding: utf-8 -*-
"""
Original file is located at
    https://colab.research.google.com/drive/10K4AWtzpcmmZ10wsXB6-az_zRCr-vK_M
"""

#IMPORT THE LIBRARIES
import pandas as pd
import numpy as np

import os
import sys

# librosa is a Python library for analyzing audio and music. It can be used to extract the data from the audio files we will see it later.
import librosa
import librosa.display
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split

# to play the audio files
import IPython.display as ipd
from IPython.display import Audio
import keras
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Embedding
from keras.layers import LSTM,BatchNormalization , GRU
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from keras.layers import Input, Flatten, Dropout, Activation
from keras.layers import Conv1D, MaxPooling1D, AveragePooling1D
from keras.models import Model
from keras.callbacks import ModelCheckpoint
from tensorflow.keras.optimizers import SGD



import warnings
if not sys.warnoptions:
    warnings.simplefilter("ignore")
warnings.filterwarnings("ignore", category=DeprecationWarning)
import tensorflow as tf
print ("Done")

!apt-get update
!apt-get install -y libsndfile1

"""# Importing Data

Ravdess Dataframe
Here is the filename identifiers as per the official RAVDESS website:

* Modality (01 = full-AV, 02 = video-only, 03 = audio-only).
* Vocal channel (01 = speech, 02 = song).
* Emotion (01 = neutral, 02 = calm, 03 = happy, 04 = sad, 05 = angry, 06 = fearful, 07 = disgust, 08 = surprised).
* Emotional intensity (01 = normal, 02 = strong). NOTE: There is no strong intensity for the 'neutral' emotion.
* Statement (01 = "Kids are talking by the door", 02 = "Dogs are sitting by the door").
* Repetition (01 = 1st repetition, 02 = 2nd repetition).
* Actor (01 to 24. Odd numbered actors are male, even numbered actors are female).

So, here's an example of an audio filename. 02-01-06-01-02-01-12.mp4 This means the meta data for the audio file is:

* Video-only (02)
* Speech (01)
* Fearful (06)
* Normal intensity (01)
* Statement "dogs" (02)
* 1st Repetition (01)
* 12th Actor (12) - Female (as the actor ID number is even)
"""



"""references for dataset:
1) Ravdess - https://zenodo.org/records/1188976
2) SAVEE - http://kahlan.eps.surrey.ac.uk/savee/
3) chrema-D - https://github.com/CheyneyComputerScience/CREMA-D
4) TESS- https://utoronto.scholaris.ca/collections/036db644-9790-4ed0-90cc-be1dfb8a4b66
"""

!unzip ravdess-emotional-speech-audio.zip -d /content/ravdess_data
/content/drive/MyDrive/ravdess_data/audio_speech_actors_01-24

#preparing data set

ravdess = "/content/drive/MyDrive/ravdess_data/audio_speech_actors_01-24"
ravdess_directory_list = os.listdir(ravdess)
print(ravdess_directory_list)

Crema = "/content/drive/MyDrive/cremad/AudioWAV/"
Tess = "/content/drive/MyDrive/toronto-emotional-speech-set-tess/tess toronto emotional speech set data/TESS Toronto emotional speech set data/"
Savee = "/content/drive/MyDrive/surrey-audiovisual-expressed-emotion-savee/ALL/"

"""# preprocessing

**Ravdees**
"""

file_emotion = []
file_path = []
def extract_ravdess_data(ravdess, ravdess_directory_list):
    for actor in ravdess_directory_list:
        actor_directory = os.listdir(ravdess + actor)
        for dir in actor_directory:
            part = dir.split('.')[0].split('-')
            file_emotion.append(int(part[2]))
            file_path.append(ravdess + actor + '/' + dir)

extract_ravdess_data(ravdess, ravdess_directory_list)

print(actor[0])
print(part[0])
print(file_path[0])
print(int(part[2]))
print(f)

emotion_df = pd.DataFrame(file_emotion, columns=['Emotions'])

path_df = pd.DataFrame(file_path, columns=['Path'])
ravdess_df = pd.concat([emotion_df, path_df], axis=1)

ravdess_df.Emotions.replace({1:'neutral', 2:'neutral', 3:'happy', 4:'sad', 5:'angry', 6:'fear', 7:'disgust',
                             8:'surprise'},
                            inplace=True)
print(ravdess_df.head())
print("______________________________________________")
print(ravdess_df.tail())
print("_______________________________________________")
print(ravdess_df.Emotions.value_counts())

"""**Crema DataFrame**

- The dataset consists of 7,442 original audio clips recorded by a diverse group of 91 actors (48 male and 43 female actors).
- The age range of the actors spans from 20 to 74 years old, representing multiple racial and ethnic backgrounds including African American, Asian, Caucasian, Hispanic, and Unspecified groups.
- Each recording is based on one of 12 different sentences, which were performed with varying emotional expressions.
- The emotional expressions in the recordings cover six distinct emotions: Anger, Disgust, Fear, Happy, Neutral, and Sad.
- Each emotional expression was recorded at different intensity levels: Low, Medium, High, and Unspecified, allowing for varying degrees of emotional manifestation.
"""

def process_audio_dataset(root_directory):
   audio_files = os.listdir(root_directory)

   sentiment_labels = []
   audio_paths = []

   emotion_mapping = {
       'SAD': 'melancholy',
       'ANG': 'irritated',
       'DIS': 'repulsed',
       'FEA': 'anxious',
       'HAP': 'joyful',
       'NEU': 'balanced'
   }

   for audio_file in audio_files:
       full_path = os.path.join(root_directory, audio_file)
       audio_paths.append(full_path)

       components = audio_file.split('_')
       emotion_code = components[2]

       sentiment = emotion_mapping.get(emotion_code, 'undefined')
       sentiment_labels.append(sentiment)

   sentiment_data = pd.DataFrame(sentiment_labels, columns=['Sentiment'])
   filepath_data = pd.DataFrame(audio_paths, columns=['FilePath'])
   combined_dataset = pd.concat([sentiment_data, filepath_data], axis=1)

   print("\nSentiment Distribution:")
   print(combined_dataset.Sentiment.value_counts())

   return combined_dataset

process_audio_dataset(Crema)
audio_df = process_audio_dataset(dataset_path)
print("\nFirst few entries:")
print(audio_df.head())

"""**TESS dataset**

* The dataset contains recordings of 200 target words embedded in the phrase "Say the word _"

* Two female actresses participated in the recordings:
  * First actress: 26 years old
  * Second actress: 64 years old

* Each word was recorded expressing seven different emotions:
  * Anger
  * Disgust
  * Fear
  * Happiness
  * Pleasant surprise
  * Sadness
  * Neutral

* The dataset contains a total of 2,800 audio files (WAV format)

* The dataset structure is organized hierarchically:
  * Main folders are divided by actress
  * Sub-folders are organized by emotion
  * Each emotion folder contains all 200 target word recordings
"""

tess_directory_list = os.listdir(Tess)

file_emotion = []
file_path = []

for dir in tess_directory_list:
    directories = os.listdir(Tess + dir)
    for file in directories:
        part = file.split('.')[0]
        part = part.split('_')[2]
        if part=='ps':
            file_emotion.append('surprise')
        else:
            file_emotion.append(part)
        file_path.append(Tess + dir + '/' + file)

emotion_df = pd.DataFrame(file_emotion, columns=['Emotions'])

path_df = pd.DataFrame(file_path, columns=['Path'])
Tess_df = pd.concat([emotion_df, path_df], axis=1)
Tess_df.head()
print(Tess_df.Emotions.value_counts())

"""**SAVEE Dataset**

* The SAVEE database features recordings from four native English male speakers:
  * Identified as DC, JE, JK, and KL
  * All were postgraduate students and researchers at the University of Surrey
  * Age range: 27 to 31 years

* The database covers seven emotion categories:
  * Anger
  * Disgust
  * Fear
  * Happiness
  * Sadness
  * Surprise
  * Neutral (added as a baseline category)

* The recording material structure:
  * 15 TIMIT sentences per emotion
    * 3 common sentences
    * 2 emotion-specific sentences
    * 10 generic phonetically-balanced sentences
  * 30 neutral sentences total (including common and emotion-specific sentences)

* Dataset composition:
  * 120 utterances per speaker
  * Example sentences for each emotion:
    * Common: "She had your dark suit in greasy wash water all year"
    * Anger: "Who authorized the unlimited expense account?"
    * Disgust: "Please take this dirty table cloth to the cleaners for me"
    * Fear: "Call an ambulance for medical assistance"
    * Happiness: "Those musicians harmonize marvelously"
    * Sadness: "The prospect of cutting back spending is an unpleasant one for any governor"
    * Surprise: "The carpet cleaners shampooed our oriental rug"
    * Neutral: "The best way to learn is to solve extra problems"
"""

savee_directory_list = os.listdir(Savee)

file_emotion = []
file_path = []

for file in savee_directory_list:
    file_path.append(Savee + file)
    part = file.split('_')[1]
    ele = part[:-6]
    if ele=='a':
        file_emotion.append('angry')
    elif ele=='d':
        file_emotion.append('disgust')
    elif ele=='f':
        file_emotion.append('fear')
    elif ele=='h':
        file_emotion.append('happy')
    elif ele=='n':
        file_emotion.append('neutral')
    elif ele=='sa':
        file_emotion.append('sad')
    else:
        file_emotion.append('s
emotion_df = pd.DataFrame(file_emotion, columns=['Emotions'])

path_df = pd.DataFrame(file_path, columns=['Path'])
Savee_df = pd.concat([emotion_df, path_df], axis=1)
Savee_df.head()
print(Savee_df.Emotions.value_counts())

"""**Integration**"""

data_path = pd.concat([ravdess_df, Crema_df, Tess_df, Savee_df], axis = 0)
data_path.to_csv("data_path.csv",index=False)
data_path.head()

print(data_path.Emotions.value_counts())

""">*                           Data Visualisation and Exploration"""

import matplotlib.pyplot as plt
import seaborn as sns

plt.title('Count of Emotions', size=16)
sns.countplot(data_path.Emotions)
plt.ylabel('Count', size=12)
plt.xlabel('Emotions', size=12)
sns.despine(top=True, right=True, left=False, bottom=False)
plt.show()

data,sr = librosa.load(file_path[0])
sr

ipd.Audio(data,rate=sr)

plt.figure(figsize=(10, 5))
spectrogram = librosa.feature.melspectrogram(y=data, sr=sr, n_mels=128,fmax=8000)
log_spectrogram = librosa.power_to_db(spectrogram)
librosa.display.specshow(log_spectrogram, y_axis='mel', sr=sr, x_axis='time');
plt.title('Mel Spectrogram ')
plt.colorbar(format='%+2.0f dB')

mfcc = librosa.feature.mfcc(y=data, sr=sr, n_mfcc=30)


plt.figure(figsize=(16, 10))
plt.subplot(3,1,1)
librosa.display.specshow(mfcc, x_axis='time')
plt.ylabel('MFCC')
plt.colorbar()

ipd.Audio(data,rate=sr)

"""# Data augmentation"""

def noise(data, noise_factor=0.035):
    noise_amplitude = noise_factor * np.random.uniform() * np.max(data)
    noise = noise_amplitude * np.random.normal(size=data.shape[0])
    return data + noise

def stretch(data, rate=0.8):
    return librosa.effects.time_stretch(data, rate=rate)

def shift(data, shift_range=(-5000, 5000)):
    shift_amount = np.random.randint(shift_range[0], shift_range[1])
    return np.roll(data, shift_amount)

def pitch(data, sampling_rate, n_steps=4):
    return librosa.effects.pitch_shift(data, sr=sampling_rate, n_steps=n_steps)

    return librosa.effects.pitch_shift(data, sampling_rate, pitch_factor)

# NORMAL AUDIO


import librosa.display
plt.figure(figsize=(12, 5))
librosa.display.waveshow(y=data, sr=sr)
ipd.Audio(data,rate=sr)

# AUDIO WITH NOISE
x = noise(data)
plt.figure(figsize=(12,5))
librosa.display.waveshow(y=x, sr=sr)
ipd.Audio(x, rate=sr)

# STRETCHED AUDIO
x = stretch(data)
plt.figure(figsize=(12, 5))
librosa.display.waveshow(y=x, sr=sr)
ipd.Audio(x, rate=sr)

# SHIFTED AUDIO
x = shift(data)
plt.figure(figsize=(12,5))
librosa.display.waveshow(y=x, sr=sr)
ipd.Audio(x, rate=sr)

# AUDIO WITH PITCH
x = pitch(data, sr)
plt.figure(figsize=(12, 5))
librosa.display.waveshow(y=x, sr=sr)
ipd.Audio(x, rate=sr)

"""# Feature extraction"""

def zcr(data, frame_length, hop_length):
    zero_crossings = librosa.feature.zero_crossing_rate(y=data, frame_length=frame_length, hop_length=hop_length)
    return np.squeeze(zero_crossings)

def rmse(data, frame_length=2048, hop_length=512):
    root_mean_square = librosa.feature.rms(y=data, frame_length=frame_length, hop_length=hop_length)
    return np.squeeze(root_mean_square)

def mfcc(data, sr, frame_length=2048, hop_length=512, flatten: bool = True):
    mel_coefficients = librosa.feature.mfcc(y=data, sr=sr, n_mfcc=20, n_fft=frame_length, hop_length=hop_length)
    transposed = mel_coefficients.T
    return np.ravel(transposed) if flatten else np.squeeze(transposed)

def extract_features(data, sr=22050, frame_length=2048, hop_length=512):
    feature_list = [
        zcr(data, frame_length, hop_length),
        rmse(data, frame_length, hop_length),
        mfcc(data, sr, frame_length, hop_length)
    ]
    return np.hstack(feature_list)

def get_features(path, duration=2.5, offset=0.6):
    data, sr = librosa.load(path, duration=duration, offset=offset)

    original_features = extract_features(data)
    audio_features = np.array(original_features)

    noised_data = noise(data)
    noised_features = extract_features(noised_data)
    audio_features = np.vstack((audio_features, noised_features))

    pitched_data = pitch(data, sr)
    pitched_features = extract_features(pitched_data)
    audio_features = np.vstack((audio_features, pitched_features))

    pitched_noised_data = noise(pitch(data, sr))
    pitched_noised_features = extract_features(pitched_noised_data)
    audio_features = np.vstack((audio_features, pitched_noised_features))

    return audio_features

def noise(data, noise_factor=0.005):
    noise = np.random.randn(len(data))
    augmented_data = data + noise_factor * noise
    return augmented_data

def pitch(data, sampling_rate, pitch_factor=0.7):
    return librosa.effects.pitch_shift(data, sr=sampling_rate, n_steps=pitch_factor)

import multiprocessing as mp
print("Number of processors: ", mp.cpu_count())

"""# Noraml way to get features"""

import timeit
from tqdm import tqdm
start = timeit.default_timer()
X,Y=[],[]
for path,emotion,index in tqdm (zip(data_path.Path,data_path.Emotions,range(data_path.Path.shape[0]))):
    features=get_features(path)
    if index%500==0:
        print(f'{index} audio has been processed')
    for i in features:
        X.append(i)
        Y.append(emotion)
print('Done')
stop = timeit.default_timer()

print('Time: ', stop - start)

from joblib import Parallel, delayed
import timeit
def process_feature(path, emotion):
    features = get_features(path)
    local_X = [feature for feature in features]
    local_Y = [emotion] * len(features)
    return local_X, local_Y

def parallel_processing(paths, emotions):
    results = Parallel(n_jobs=-1)(
        delayed(process_feature)(path, emotion)
        for path, emotion in zip(paths, emotions)
    )
    return results

def combine_results(results):
    X_combined = []
    Y_combined = []
    for x, y in results:
        X_combined.extend(x)
        Y_combined.extend(y)
    return X_combined, Y_combined

start = timeit.default_timer()

paths = data_path.Path
emotions = data_path.Emotions

processed_results = parallel_processing(paths, emotions)
X, Y = combine_results(processed_results)

stop = timeit.default_timer()
execution_time = stop - start
print('Time: ', execution_time)

len(X), len(Y), data_path.Path.shape

"""# Saving features"""

Emotions = pd.DataFrame(X)
Emotions['Emotions'] = Y
Emotions.to_csv('emotion.csv', index=False)
Emotions.head()

Emotions = pd.read_csv('./emotion.csv')
Emotions.head()

print(Emotions.isna().any())

Emotions=Emotions.fillna(0)
print(Emotions.isna().any())
Emotions.shape

np.sum(Emotions.isna())

"""# Data preparation"""

X = Emotions.iloc[: ,:-1].values
Y = Emotions['Emotions'].values

from sklearn.preprocessing import StandardScaler, OneHotEncoder
encoder = OneHotEncoder()
Y = encoder.fit_transform(np.array(Y).reshape(-1,1)).toarray()

print(Y.shape)
X.shape

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(X, Y, random_state=42,test_size=0.2, shuffle=True)
x_train.shape, y_train.shape, x_test.shape, y_test.shape

# scaling our data with sklearn's Standard scaler
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)
x_train.shape, y_train.shape, x_test.shape, y_test.shape

"""# CNN model"""

cnn_x_train =np.expand_dims(x_train, axis=2)
cnn_x_test= np.expand_dims(x_test, axis=2)
cnn_x_train.shape, y_train.shape, cnn_x_test.shape, y_test.shape
#x_testcnn[0]

import tensorflow.keras.layers as L

model = tf.keras.Sequential([
    L.Conv1D(512,kernel_size=5, strides=1,padding='same', activation='relu',input_shape=(X_train.shape[1],1)),
    L.BatchNormalization(),
    L.MaxPool1D(pool_size=5,strides=2,padding='same'),

    L.Conv1D(512,kernel_size=5,strides=1,padding='same',activation='relu'),
    L.BatchNormalization(),
    L.MaxPool1D(pool_size=5,strides=2,padding='same'),
    Dropout(0.2),

    L.Conv1D(256,kernel_size=5,strides=1,padding='same',activation='relu'),
    L.BatchNormalization(),
    L.MaxPool1D(pool_size=5,strides=2,padding='same'),

    L.Conv1D(256,kernel_size=3,strides=1,padding='same',activation='relu'),
    L.BatchNormalization(),
    L.MaxPool1D(pool_size=5,strides=2,padding='same'),
    Dropout(0.2),

    L.Conv1D(128,kernel_size=3,strides=1,padding='same',activation='relu'),
    L.BatchNormalization(),
    L.MaxPool1D(pool_size=3,strides=2,padding='same'),
    Dropout(0.2),

    L.Flatten(),
    L.Dense(512,activation='relu'),
    L.BatchNormalization(),
    L.Dense(7,activation='softmax')
])
model.compile(optimizer='adam',loss='categorical_crossentropy',metrics='accuracy')
model.summary()

history=model.fit(cnn_x_train, y_train, epochs=50, validation_data=(cnn_x_test, y_test), batch_size=64,callbacks=[early_stop,lr_reduction,model_checkpoint])

print("Accuracy of our model on test data : " , model.evaluate(cnn_x_train,y_test)[1]*100 , "%")

epochs = [i for i in range(50)]
fig , ax = plt.subplots(1,2)
train_acc = history.history['accuracy']
train_loss = history.history['loss']
test_acc = history.history['val_accuracy']
test_loss = history.history['val_loss']

fig.set_size_inches(20,6)
ax[0].plot(epochs , train_loss , label = 'Training Loss')
ax[0].plot(epochs , test_loss , label = 'Testing Loss')
ax[0].set_title('Training & Testing Loss')
ax[0].legend()
ax[0].set_xlabel("Epochs")

ax[1].plot(epochs , train_acc , label = 'Training Accuracy')
ax[1].plot(epochs , test_acc , label = 'Testing Accuracy')
ax[1].set_title('Training & Testing Accuracy')
ax[1].legend()
ax[1].set_xlabel("Epochs")
plt.show()

target_test = model.predict(cnn_x_test)
target_pred = encoder.inverse_transform(target_test)
y_test_ = encoder.inverse_transform(y_test)

data = pd.DataFrame(columns=['Predicted Labels', 'Actual Labels'])
data['Predicted Labels'] = target_pred.flatten()
data['Actual Labels'] = y_test_.flatten()

data.head(10)

data

"""# Evalutation

Results of best model
"""

from sklearn.metrics import confusion_matrix,classification_report
confusion_matrix_var = confusion_matrix(y_test0, y_pred0)
plt.figure(figsize = (12, 10))
cm = pd.DataFrame(cm , index = [i for i in encoder.categories_] , columns = [i for i in encoder.categories_])

sns.heatmap(cm, linecolor='white', cmap='Blues', linewidth=1, annot=True, fmt='.2f')
plt.title('Confusion Matrix', size=20)
plt.xlabel('Predicted Labels', size=14)
plt.ylabel('Actual Labels', size=14)
plt.show()
print(classification_report(y_test_, target_pred))

"""# Saving Best Model"""

from tensorflow.keras.models import Sequential, model_from_json
model_json = model.to_json()
with open("Best_classification_model.json", "w") as json_file:
    json_file.write(model_json)
model.save_weights("Best_classification_model_weights.h5")
print("Saved model to disk")

from tensorflow.keras.models import Sequential, model_from_json
json_file = open('/content/Best_classification_model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
print("Loaded model from disk")

loaded_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
score = loaded_model.evaluate(x_testcnn,y_test)
print("%s: %.2f%%" % (loaded_model.metrics_names[1], score[1]*100))

"""# Saving Standard Scaler

pickle file
"""

import pickle

with open('scaler2.pickle', 'wb') as f:
    pickle.dump(scaler, f)

with open('scaler2.pickle', 'rb') as f:
    scaler2 = pickle.load(f)

with open('encoder2.pickle', 'wb') as f:
    pickle.dump(encoder, f)

with open('encoder2.pickle', 'rb') as f:
    encoder2 = pickle.load(f)


print("Done")

"""# Test script

"""

from tensorflow.keras.models import Sequential, model_from_json
json_file = open('/content/Best_classification_model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("/content/Best_classification_model_weights.h5")
print("Loaded model from disk")

import pickle

with open('/content/scaler2.pickle', 'rb') as f:
    scaler2 = pickle.load(f)

with open('/content/encoder2.pickle', 'rb') as f:
    encoder2 = pickle.load(f)


print("Done")

import librosa

res=get_predict_feat("/content/ravdess-emotional-speech-audio/Actor_01/03-01-07-01-01-01-01.wav")
print(res.shape)

emotions1={1:'Neutral', 2:'Calm', 3:'Happy', 4:'Sad', 5:'Angry', 6:'Fear', 7:'Disgust',8:'Surprise'}
def predict_voice(path1):
    res=get_predict_feat(path1)
    result=loaded_model.predict(res)
    y_pred = encoder2.inverse_transform(result)
    print(y_pred[0][0])

predict_voice("/content/ravdess-emotional-speech-audio/Actor_02/03-01-01-01-01-01-02.wav")

predict_voice("/content/ravdess-emotional-speech-audio/Actor_01/03-01-01-01-01-01-01.wav")

predict_voice("/content/ravdess-emotional-speech-audio/Actor_01/03-01-05-01-02-02-01.wav")

predict_voice("/content/ravdess-emotional-speech-audio/Actor_21/03-01-04-02-02-02-21.wav")

predict_voice("/content/ravdess-emotional-speech-audio/Actor_02/03-01-06-01-02-02-02.wav")

predict_voice("/content/ravdess-emotional-speech-audio/Actor_01/03-01-08-01-01-01-01.wav")

predict_voice("/content/ravdess-emotional-speech-audio/Actor_01/03-01-07-01-01-01-01.wav")
