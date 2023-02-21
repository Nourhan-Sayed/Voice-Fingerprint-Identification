# Voice Fingerprint Identification || DSP-Task-3

***(Voice Door-Lock)***
> Voice Fingerprint Door-Lock is a Digital-Signal-Processing **WebApp** that is used for **Speaker-Identification** and **Sentence-Verification** using **Machine-Learning** and extracted **Audio-Features** from voice biometrics. 

## Table of contents:

- [Voice Fingerprint Principles](#voice-fingerprint-principles)
- [Project full Demo](#project-full-demo)
- [Dynamic E-Poster Graphs](#dynamic-e-poster-graphs)
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
- Calculation Steps
    1. Frame the signal, and compute fourier.
    2. Apply mel filterbank to power spectra, sum energy bands.
    3. Take the log of all filterbank energies, then take Discrete Fourier Transform (DCT).
    4. Keep DCT coefficients 2-13, discard the rest.
    5. Take the logarithm of the power spectrum
    • Delta and Delta-Delta features are usually also appended, then applying liftering.
    
 ![mfcc](https://user-images.githubusercontent.com/84602951/220436300-47e48fef-e70a-4e96-a8b8-32bad3940a59.gif)
  
You can read more about MFCC [here](https://haythamfayek.com/2016/04/21/speech-processing-for-machine-learning.html)
    
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

![gmm2](https://user-images.githubusercontent.com/84602951/220436625-d07f20a7-94a5-4519-94ea-a55b28d9f108.PNG)

You can read more about GMM [here](https://www.researchgate.net/publication/274963749_Speaker_Identification_Using_GMM_with_MFCC)

## Project full Demo
video
## Dynamic E-Poster Graphs
### MFCC Spectogram
* Spectogram represents the Mel-Frequency Cepstral Coefficients of the user audio.

![MFCC](https://user-images.githubusercontent.com/84602951/220437312-d5f64abe-370e-40b8-bf55-93b2bb2c2a60.png)

### Gaussian Normal Distribution
* Represents the normal distribution of mfcc feauture of each user of the team and the input user voice to represent which team fingerprint is closer to the input audio based on principles of GMM Model.

![Normal](https://user-images.githubusercontent.com/84602951/220438245-ca8697bf-4ca0-4385-8448-bc484b5f8b6d.png)

### Scores Bar Chart
* Bar chart represents scores of gmm models to represent which score is closer to the team scores and compares them with the threshold of dissimilarity.

![scores](https://user-images.githubusercontent.com/84602951/220438689-e68fd2b6-6fbf-4ab7-a4e2-a59357929736.png)


## Project Structure
* Frontend takes the user audio and sends it to the backend.
* Backend extracts the audio features and sends them to the machine learning model.
* Machine learning model compares the input audio features with the team audio features in team verification step
* If the Voice Fingerprint is verified(From Registered team Users), the machine learning model compares the input audio features with the user audio features in sentence verification step.
* Door is opened only if the Voice Fingerprint(User in team) is verified and the sentence(Open The Door) is verified.
* Then Machine learning model returns the result to the backend and the backend returns the result to the frontend
* Frontend displays the result to the user and the door is opened if the result is verified.


![process](https://user-images.githubusercontent.com/84602951/220436969-f1eb2bb3-c78a-413d-8d30-944ed46cfa9e.jpeg)

- Frontend :
  - HTML
  - CSS
  - JavaScript
- Backend :
  - Flask (Python)
- Machine Learning Model Training
    - GMM Model (Python)

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

