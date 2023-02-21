# Voice Fingerprint Identification || DSP-Task-3

***(Voice Door-Lock)***
> Voice Fingerprint Door-Lock is a Digital-Signal-Processing **WebApp** that is used for **Speaker-Identification** and **Sentence-Verification** using **Machine-Learning** and extracted **Audio-Features** from voice biometrics. 

## Table of contents:

- [Voice Fingerprint Principles](#Voice-Fingerprint-Principles)
- [Project full Demo](#project-full-demo)
- [Dynamic E-Poster Graphs](#Dynamic-E-Poster-Graphs)
- [Project Structure](#project-structure)
- [Run The Project](#run-the-project)
- [Team Members](#team-members)

## Voice Fingerprint Principles
Voice Fingerprint is one of the DSP Applications that depends on **Audio Feature Extraction** and **Machine-Learning Model Trainig** 
### Feature Extraction
The **Audio Features** are extracted from the **Audio Signal** using **Fourier Transform** and **Mel-Frequency Cepstral Coefficients** (MFCC) and their Delta

***What is MFCC?***
- A set of features used in speech recognition and audio information retrieval.
- Represent the spectral envelope of a sound by measuring the magnitude of the spectral components

- Represent the short-term power spectrum of a sound by combining a number of adjacent frequency bands

- Represent the spectral shape of a sound in the frequency domain
- Calculated by taking the Discrete Fourier Transform of a signal and then taking the logarithm of the power spectrum

1. Frame the signal, and compute fourier.
2. Apply mel filterbank to power spectra, sum energy bands.
3. Take the log of all filterbank energies, then take Discrete Fourier Transform (DCT).
4. Keep DCT coefficients 2-13, discard the rest.
5. Take the logarithm of the power spectrum
• Delta and Delta-Delta features are usually also appended, then applying liftering.

You can read more about MFCC [here](https://haythamfayek.com/2016/04/21/speech-processing-for-machine-learning.html)
img

    
### Model Training
***Gaussian Mixture Model (GMM)***
- GMM is an unsupervised Clustering model
- GMM is a probabilistic model that assumes all the data points are generated from a mixture of a finite number of Gaussian distributions with unknown parameters.
- GMM is used in voice identification to identify the speaker by analyzing the spectral characteristics of the voice.
- GMM uses a set of Gaussian distributions to model the spectral characteristics of the voice.
- Each Gaussian distribution is characterized by its mean and variance.
- GMM uses an Expectation Maximization (EM) algorithm to estimate the parameters of the Gaussian distributions.
- The EM algorithm iteratively estimates the parameters of the Gaussian distributions by maximizing the likelihood of the observed data.
- The GMM model is then used to classify the speaker by comparing the spectral characteristics of the voice with the estimated parameters of the Gaussian distributions.

img

You can read more about GMM [here](https://www.researchgate.net/publication/274963749_Speaker_Identification_Using_GMM_with_MFCC)

## Project full Demo
video
## Dynamic E-Poster Graphs



## Project Structure
- Frontend :
  - HTML
  - CSS
  - JavaScript
- Backend :
  - Flask (Python)
- Machine Learning Model Training
    - Python

* Used Libraries
    * python_speech_features
    * librosa
    * sklearn
    * Numpy
    * Scipy

 ## Run The Project
- Clone the project
- Open Terminal and write
```bash
cd src
pip install -r requirements.txt
flask run --reload 
```
- Open server link in browser http://127.0.0.1:5000/
## Team Members
- [Omar Saad](https://github.com/Omar-Saad-ELGharbawy)
- [Neven Mohamed](https://github.com/NeveenMohamed)
- [Nourhan Sayed](https://github.com/nourhansayed102)
- [Salman](https://github.com/Salmoon8)

