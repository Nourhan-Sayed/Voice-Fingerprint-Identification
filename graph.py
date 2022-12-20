import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import librosa

def features_extractor(file):
    audio, sample_rate = librosa.load(file, res_type='kaiser_fast') 
    mfccs_features = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=46)
#     mfccs_scaled_features = np.mean(mfccs_features.T,axis=0)
    return mfccs_features[22]


omar_f = features_extractor("normal_records\omar_open_door-sample0.wav")
neven_f = features_extractor("normal_records\open_door_neveen-sample1.wav")
salman_f = features_extractor("normal_records\open_the_door_nourhan-sample1.wav")
nur_f = features_extractor("normal_records\open_the_door_salman-sample1.wav")
# other_f = features_extractor("other.wav")
# other_f = features_extractor("omar_open_door-sample3.wav")
# other_f = features_extractor("open_door_neveen-sample3.wav")

#x-axis ranges from -5 and 5 with .001 steps
# x = np.arange(-5, 5, 0.001)

#define multiple normal distributions
plt.plot(omar_f, norm.pdf(omar_f, np.mean(omar_f), np.std(omar_f)), label='Omar', color='gold')
plt.plot(neven_f, norm.pdf(neven_f, np.mean(neven_f), np.std(neven_f)), label='Neven', color='red')
plt.plot(salman_f, norm.pdf(salman_f, np.mean(salman_f), np.std(salman_f)), label='Salman', color='pink')
plt.plot(nur_f, norm.pdf(nur_f, np.mean(nur_f), np.std(nur_f)), label='Nur', color='blue')
# plt.plot(other_f, norm.pdf(other_f, np.mean(other_f), np.std(other_f)), label='Other', color='black')
# plt.plot(other_f, norm.pdf(other_f, np.mean(other_f), np.std(other_f)), label='Other', color='black')

#add legend to plot
plt.legend(title='Parameters')

#add axes labels and a title
plt.ylabel('Density')
plt.xlabel('x')
plt.title('Normal Distributions', fontsize=14)
plt.savefig('static/Css/img/Normal.png', bbox_inches='tight',pad_inches = 0)

