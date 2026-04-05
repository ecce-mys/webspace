"""Synthetic EEG signal generation and filtering.

**Overall data flow**
----------------------
1. **Signal generation** – ``generate_signal`` creates a time vector ``t``
   and a composite waveform that mixes a 10 Hz sinusoid (typical alpha
   rhythm) with a 50 Hz sinusoid (simulated power‑line interference).
2. **Filter design** – ``butter_lowpass`` computes the coefficients of a
   Butterworth low‑pass filter for a given cutoff frequency, sampling
   rate and filter order.  The Butterworth design is chosen because it
   provides a maximally flat magnitude response in the pass‑band, which
   preserves the shape of the desired brain‑wave component.
3. **Filtering** – ``apply_filter`` applies the filter to the raw signal
   using ``scipy.signal.filtfilt`` which performs forward‑ and reverse‑
   filtering.  This zero‑phase filtering removes phase distortion, a
   crucial property when analysing EEG where the timing of events matters.
4. **Visualization** – ``plot_signals`` draws the original and filtered
   waveforms and writes the figure to ``eeg_filter.png``.

The script ties these steps together in ``main`` and runs them when the
module is executed as a script.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt
from datetime import datetime


def generate_signal(duration: float = 1.0, fs: int = 500) -> tuple[np.ndarray, np.ndarray]:
    """Generate a synthetic EEG signal.

    Parameters
    ----------
    duration: float
        Length of the signal in seconds.  A longer duration yields more
        samples, improving frequency resolution when analysing the
        spectrum.
    fs: int
        Sampling frequency in Hz.  EEG is typically sampled between 250 Hz
        and 1 kHz; 500 Hz provides a Nyquist frequency of 250 Hz, comfortably
        above the 50 Hz interference we wish to model.

    Returns
    -------
    t: np.ndarray
        Time vector.
    signal: np.ndarray
        Composite signal containing a 10 Hz sinusoid (brain rhythm) and a
        50 Hz sinusoid (power‑line noise).
    """
    # 핵심 요약 (Korean): 지정된 기간과 샘플링 주파수에 따라 시간 벡터와 두 개의 사인파(10 Hz, 50 Hz)를 합성합니다.
    t = np.arange(0, duration, 1 / fs)
    # 10 Hz brain rhythm (amplitude 1.0) and 50 Hz power‑line noise (amplitude 0.5)
    signal = np.sin(2 * np.pi * 10 * t) + 0.5 * np.sin(2 * np.pi * 50 * t)
    return t, signal


def butter_lowpass(cutoff: float, fs: int, order: int = 4) -> tuple[np.ndarray, np.ndarray]:
    """Design a Butterworth low‑pass filter.

    Parameters
    ----------
    cutoff: float
        Cut‑off frequency in Hz.  Frequencies below this value are passed
        with minimal attenuation; frequencies above are increasingly
        attenuated.  A 30 Hz cutoff preserves the 10 Hz brain rhythm while
        suppressing the 50 Hz interference.
    fs: int
        Sampling frequency in Hz – required to normalise the cutoff to the
        Nyquist frequency (half the sampling rate).
    order: int, default 4
        The filter order determines the steepness of the roll‑off.  A 4th‑
        order Butterworth provides a good trade‑off between attenuation
        of unwanted components and stability of the filter.

    Returns
    -------
    b, a: tuple[np.ndarray, np.ndarray]
        Numerator (b) and denominator (a) coefficients of the IIR filter.
    """
    # 핵심 요약 (Korean): 지정된 차단 주파수와 샘플링 주파수에 따라 정규화된 차단 주파수를 계산하고, Butterworth 필터 계수를 반환합니다.
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype="low", analog=False)
    return b, a


def apply_filter(data: np.ndarray, cutoff: float, fs: int, order: int = 4) -> np.ndarray:
    """Apply a zero‑phase Butterworth low‑pass filter to ``data``.

    Parameters
    ----------
    data: np.ndarray
        The raw EEG signal to be filtered.
    cutoff, fs, order: see :func:`butter_lowpass`.

    Returns
    -------
    np.ndarray
        The filtered signal.  ``scipy.signal.filtfilt`` is used to achieve
        zero‑phase distortion by filtering forward and then backward.
    """
    # 핵심 요약 (Korean): 입력 신호에 설계된 Butterworth 필터를 zero‑phase 방식으로 적용하여 위상 왜곡 없이 잡음을 제거합니다.
    b, a = butter_lowpass(cutoff, fs, order)
    return filtfilt(b, a, data)


def plot_signals(t: np.ndarray, original: np.ndarray, filtered: np.ndarray) -> None:
    """Plot original and filtered signals and save the figure.

    Parameters
    ----------
    t: np.ndarray
        Time axis for both signals.
    original: np.ndarray
        Unfiltered synthetic EEG.
    filtered: np.ndarray
        Result after low‑pass filtering.
    """
    # 핵심 요약 (Korean): 원본 및 필터링된 신호를 시각화하여 두 파형을 비교하고, 결과 이미지를 파일로 저장합니다.
    plt.figure(figsize=(10, 4))
    plt.plot(t, original, label="Original (10 Hz + 50 Hz)", alpha=0.7)
    plt.plot(t, filtered, label="Filtered (≤30 Hz)", linewidth=2)
    plt.xlabel("Time [s]")
    plt.ylabel("Amplitude")
    plt.title("EEG Signal: Original vs. Butterworth Low‑pass Filtered")
    plt.legend()
    plt.tight_layout()
    plt.savefig("eeg_filter.png")
    plt.close()


def main() -> None:
    fs = 500  # Sampling frequency in Hz
    t, raw_signal = generate_signal(duration=1.0, fs=fs)
    filtered_signal = apply_filter(raw_signal, cutoff=30.0, fs=fs, order=4)
    plot_signals(t, raw_signal, filtered_signal)
    print("EEG plot saved to eeg_filter.png")
    # 전체 활동 요약을 로그 파일에 기록 (Korean)
    try:
        with open("log.txt", "a", encoding="utf-8-sig") as log_file:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            log_entry = (
                f"[{timestamp}] 스크립트 실행 완료: "
                f"신호 길이={len(t)} 샘플, "
                f"차단 주파수={30.0}Hz, "
                f"필터 차수={4}\n"
            )
            log_file.write(log_entry)
    except Exception as e:
        # 로그 기록 실패 시 콘솔에 알림
        print(f"로그 기록 중 오류 발생: {e}")


if __name__ == "__main__":
    main()
