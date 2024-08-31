# Speech Emotion Recognition

This project implements a Speech Emotion Recognition system using machine learning techniques. The system can detect emotions from audio recordings and classify them into various emotional states such as calm, happy, fearful, and disgusted, etc.

## Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Installation](#installation)
- [Data Preprocessing](#data-preprocessing)
- [Feature Extraction](#feature-extraction)
- [Model Training](#model-training)
- [Model Testing](#model-testing)
- [Prediction](#prediction)
- [Visualization](#visualization)
- [Results](#results)
- [Next Steps](#next-steps)

## Overview

Speech Emotion Recognition (SER) is the process of identifying human emotions from speech. This project uses the RAVDESS dataset, which consists of emotional speech recordings. The implemented system uses various Python libraries such as `librosa`, `pyaudio`, and `scikit-learn`, etc to preprocess the audio data, extract features, train a model, and predict emotions.

## Dataset

The RAVDESS dataset is used for training and testing the emotion detection model. It contains 1,439 audio files from 24 different actors, each portraying various emotions. The dataset is stored in the `audio_speech_actors_01-24` directory.

### Emotion Labels in RAVDESS Dataset:

- **01**: Neutral
- **02**: Calm
- **03**: Happy
- **04**: Sad
- **05**: Angry
- **06**: Fearful
- **07**: Disgust
- **08**: Surprised

## Installation

To run this project, you need to install the following libraries:

- `numpy`
- `pandas`
- `librosa`
- `soundfile`
- `scikit-learn`
- `tensorflow`
- `keras`
- `python_speech_features`
- `noisereduce`
- `matplotlib`
- `tqdm`
- `pyaudio`
- `scipy`
- `speech_recognition`

To install these dependencies, run:

```
pip install numpy pandas librosa soundfile scikit-learn tensorflow keras python_speech_features noisereduce matplotlib tqdm pyaudio scipy speech_recognition
```

Or, Run the following script:  
```
pip install -r requirements.txt
```

## Data Preprocessing

1. **Load Audio Files**: The audio files are loaded from the `audio_speech_actors_01-24` directory.

2. **Speech Recognition**: The `speech_recognition` library is used to convert speech to text. This step is optional and primarily for validation purposes.

3. **Noise Reduction**: The `envelope` function is used to apply a mask to audio signals to reduce noise and silence. The cleaned audio files are saved to the `clean_speech` directory.

4. **Downsampling**: Audio files are downsampled to 16,000 Hz to ensure consistency across the dataset.

## Feature Extraction

Features are extracted from the audio files to train the model. The following features are computed:

- **MFCC (Mel-Frequency Cepstral Coefficients)**: Captures the power spectrum of the audio signal.
- **Chroma Features**: Represents 12 different pitch classes.
- **Mel Spectrogram**: Provides a perceptual representation of the audio signal.

## Model Training

The project uses a **Multi-Layer Perceptron (MLP)** classifier from `scikit-learn` to classify emotions based on the extracted features. The model is trained with the following parameters:

- `alpha=0.01`
- `batch_size=256`
- `epsilon=1e-08`
- `hidden_layer_sizes=(300,)`
- `learning_rate='adaptive'`
- `max_iter=500`

![image](https://github.com/user-attachments/assets/9dfad79e-7846-4b46-a319-8e003a06e279)


After training, the model is saved as `Emotion_Voice_Detection_Model.pkl` using the `pickle` library.

## Model Testing

The model is tested on a subset of the dataset that was not used for training. The predictions are compared with the actual labels to evaluate the model's performance.

## Prediction

The trained model can predict emotions for new audio recordings. The following steps are used to make predictions:

1. **Record Audio**: The script uses `pyaudio` to record audio from a microphone and save it as `output10.wav`.

2. **Extract Features**: Features are extracted from the recorded audio file using the `extract_feature` function.

3. **Predict Emotion**: The extracted features are fed into the trained model to predict the emotion.


## Visualization
The sample illustrations of the audio features are as follows:  

![image](https://github.com/user-attachments/assets/d1bb8a74-4889-4308-832c-c711276297e3)  
  
![image](https://github.com/user-attachments/assets/488a1bec-64c1-498f-84e1-0cf49cbaf558)  
  
![image](https://github.com/user-attachments/assets/b7301b0d-2298-44e8-9cf4-06f1c0e1942f)  
  
![image](https://github.com/user-attachments/assets/f1797b81-f691-47ca-a7c7-dffdba27fa12)  



## Results  

The final result of the new audio sample will be given in a Numpy array format as follows:  
`array(['calm', 'fearful'], dtype='<U8')`  

## Next Steps

To enhance the Emotion Voice Detection model and expand its capabilities, the following steps are suggested:

1. **Expand the Dataset**

2. **Improve Model Architecture**

3. **Optimize Hyperparameters**

4. **Real-time Emotion Detection**

5. **Add Support for Multiple Languages**

6. **Enhance Emotion Categories**

7. **Develop a User Interface**

8. **Conduct Extensive Testing**

9. **Collaborate and Open Source**

These next steps aim to improve the performance, usability, and versatility of the Emotion Voice Detection model, making it more suitable for real-world applications and diverse environments.
