import numpy as np
from scipy.io import wavfile
from scipy.fft import fft, fftfreq
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt
from scipy.io.wavfile import write
import os

sample_rate, modulated_signal = wavfile.read("modulated_noisy_audio.wav")
modulated_signal = modulated_signal / np.max(np.abs(modulated_signal)) # normalizing taaki maths kaam kare
n = len(modulated_signal)
time_axis = np.linspace(0, n / sample_rate, n)
plt.plot(time_axis[:20000], modulated_signal[: 20000])
plt.title("Modulated Signal")
plt.grid()
plt.show()

# FFT analysis
yf = np.abs(fft(modulated_signal))
xf = fftfreq(n , 1 / sample_rate)
plt.plot(xf[: n // 2], np.abs(yf[:n // 2])) #slicing till n/2 tak ki frequency repeat na ho 
plt.title("FFT of Modulated Signal")
plt.xlabel("Frequnecy in Hz")
plt.ylabel("Magnitude")
plt.grid()
plt.show()


# Demodulation

def butter_bandpass(lowcut, highcut, fs, order = 6):
    nyq = fs / 2
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype = "band")
    return b, a


def apply_bandpass(data, lowcut, highcut, fs, order = 6):
    b, a = butter_bandpass(lowcut, highcut, fs, order)
    y = filtfilt(b, a, data)
    return y


def sweep_carrier_frequency(modulated, time, fs, fc_start, fc_end, step = 500):
    for fc in range(fc_start, fc_end + 1, step):
        os.makedirs("sweep_outputs", exist_ok=True)
        carrier = np.cos(2 * np.pi * fc * time)
        demod = modulated * carrier

        filtered = apply_bandpass(demod, 300, 4000, fs)
        filtered = filtered / np.max(np.abs(filtered))

        file = f"sweep_outputs/Recovered_{fc}Hz.wav"
        write(file, fs, (filtered * 32767).astype(np.int16))

        plt.plot(time[:20000], filtered[:20000])
        plt.title(f"Demodulated Signal @ {fc} Hz")
        plt.grid()
        plt.savefig(f"sweep_outputs/plot_{fc}Hz.png")
        plt.close()


sweep_carrier_frequency(modulated_signal, time_axis, sample_rate, fc_start = 9500, fc_end= 11500, step = 500)



