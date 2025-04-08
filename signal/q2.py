import numpy as np
from scipy.io import wavfile
from scipy.fft import fft, fftfreq
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt
from scipy.io.wavfile import write

sample_rate, modulated_signal = wavfile.read("modulated_noisy_audio.wav")
modulated_signal = modulated_signal / np.max(np.abs(modulated_signal)) # normalizing taaki maths kaam kare
n = len(modulated_signal)
time_axis = np.linspace(0, n / sample_rate, n)
plt.plot(time_axis[:20000], modulated_signal[: 20000])
plt.title("Modulated Signal")
plt.grid()
plt.show()

# FFT analysis
yf = fft(modulated_signal)
xf = fftfreq(n , 1 / sample_rate)
plt.plot(xf[: n // 2], np.abs(yf[:n // 2])) #slicing till n/2 tak ki frequency repeat na ho 
plt.title("AFTER FFT")
plt.xlabel("Frequnecy in Hz")
plt.ylabel("Magnitude")
plt.xlim(0, sample_rate/2)
plt.grid()
plt.show()


# Demodulation
fc = 10582
carrier = np.cos(2* np.pi * fc * time_axis)
demodulated_signal = modulated_signal * carrier


cutoff = 2500 #Hz

def butter_lowpass(cutoff, sample_rate, order=8):
    nf = sample_rate / 2
    normal_cutoff = cutoff / nf
    b, a = butter(order, normal_cutoff, btype = "low", analog = False)
    return b, a


def lowpass_filter(data, cutoff, sample_rate, order = 8):
    b, a = butter_lowpass(cutoff, sample_rate, order = order)
    y = filtfilt(b, a, data)
    return y


filtered_signal = lowpass_filter(demodulated_signal, cutoff, sample_rate, order = 8)

filtered_signal = filtered_signal / np.max(np.abs(filtered_signal))

plt.plot(time_axis[: 20000], filtered_signal[: 20000])
plt.title("Filtered Signal after demodulation")
plt.grid()
plt.show()


write("Filtered_signal.wav", sample_rate, (filtered_signal * 32767).astype(np.int16))