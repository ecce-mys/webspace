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
   waveforms and writes the figure to disk.
5. **Automation** – every stage is appended to ``logs/progress.md`` and the
   run's results are sorted into ``outputs/<timestamp>/`` (``figures/`` for
   graphs, ``data/`` for raw/filtered samples), following ``.clinerules``.

The script ties these steps together in ``main`` and runs them when the
module is executed as a script.
"""

import matplotlib

matplotlib.use("Agg")  # 헤드리스 환경(서버/CI)에서도 그래프 저장이 가능하도록 설정

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt
from datetime import datetime
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
LOG_DIR = BASE_DIR / "logs"
PROGRESS_FILE = LOG_DIR / "progress.md"
RUN_LOG_FILE = LOG_DIR / "log.txt"
OUTPUT_DIR = BASE_DIR / "outputs"


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


def plot_signals(t: np.ndarray, original: np.ndarray, filtered: np.ndarray, out_path: Path) -> None:
    """Plot original and filtered signals and save the figure.

    Parameters
    ----------
    t: np.ndarray
        Time axis for both signals.
    original: np.ndarray
        Unfiltered synthetic EEG.
    filtered: np.ndarray
        Result after low‑pass filtering.
    out_path: Path
        Destination of the PNG figure.
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
    plt.savefig(out_path)
    plt.close()


def save_signals(t: np.ndarray, original: np.ndarray, filtered: np.ndarray, out_path: Path) -> None:
    """Save the time axis, raw and filtered samples as CSV."""
    # 핵심 요약 (Korean): 그래프와 별개로 원본/필터링 데이터를 CSV로 남겨 재분석이 가능하도록 합니다.
    np.savetxt(
        out_path,
        np.column_stack([t, original, filtered]),
        delimiter=",",
        header="time_s,original,filtered",
        comments="",
        fmt="%.6f",
    )


def log_progress(step: str, message: str) -> None:
    """Append one stage of the current run to ``logs/progress.md``."""
    # 핵심 요약 (Korean): .clinerules 규칙에 따라 매 단계 진행 상황을 progress.md에 누적 기록합니다.
    try:
        LOG_DIR.mkdir(exist_ok=True)
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open(PROGRESS_FILE, "a", encoding="utf-8") as f:
            f.write(f"- [{timestamp}] **{step}** – {message}\n")
    except OSError as e:
        print(f"진행 로그 기록 중 오류 발생: {e}")


def prepare_run_dirs(run_id: str) -> tuple[Path, Path]:
    """Create ``outputs/<run_id>/figures`` and ``outputs/<run_id>/data``."""
    # 핵심 요약 (Korean): 실행마다 타임스탬프 폴더를 만들어 그래프와 데이터를 종류별로 자동 정렬합니다.
    run_dir = OUTPUT_DIR / run_id
    fig_dir = run_dir / "figures"
    data_dir = run_dir / "data"
    fig_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)
    return fig_dir, data_dir


def main() -> None:
    fs = 500  # Sampling frequency in Hz
    duration = 1.0
    cutoff = 30.0
    order = 4
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")

    LOG_DIR.mkdir(exist_ok=True)
    with open(PROGRESS_FILE, "a", encoding="utf-8") as f:
        f.write(f"\n## Run {run_id}\n\n")

    t, raw_signal = generate_signal(duration=duration, fs=fs)
    log_progress("1. 신호 생성", f"{len(t)} 샘플 (fs={fs}Hz, {duration}s, 10Hz + 50Hz)")

    filtered_signal = apply_filter(raw_signal, cutoff=cutoff, fs=fs, order=order)
    log_progress("2. 필터링", f"Butterworth {order}차 저역통과, 차단 주파수 {cutoff}Hz, filtfilt(zero-phase)")

    fig_dir, data_dir = prepare_run_dirs(run_id)
    fig_path = fig_dir / "eeg_filter.png"
    data_path = data_dir / "eeg_signals.csv"
    plot_signals(t, raw_signal, filtered_signal, fig_path)
    save_signals(t, raw_signal, filtered_signal, data_path)
    log_progress(
        "3. 결과 정렬",
        f"그래프 → `{fig_path.relative_to(BASE_DIR)}`, 데이터 → `{data_path.relative_to(BASE_DIR)}`",
    )
    print(f"EEG plot saved to {fig_path.relative_to(BASE_DIR)}")
    print(f"EEG data saved to {data_path.relative_to(BASE_DIR)}")

    # 전체 활동 요약을 로그 파일에 기록 (Korean)
    try:
        with open(RUN_LOG_FILE, "a", encoding="utf-8-sig") as log_file:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            log_entry = (
                f"[{timestamp}] 스크립트 실행 완료: "
                f"신호 길이={len(t)} 샘플, "
                f"차단 주파수={cutoff}Hz, "
                f"필터 차수={order}, "
                f"결과 폴더=outputs/{run_id}\n"
            )
            log_file.write(log_entry)
    except OSError as e:
        # 로그 기록 실패 시 콘솔에 알림
        print(f"로그 기록 중 오류 발생: {e}")
    log_progress("4. 완료", "실행 요약을 `logs/log.txt`에 기록")


if __name__ == "__main__":
    main()
