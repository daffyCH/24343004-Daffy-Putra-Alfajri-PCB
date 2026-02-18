# ============================================
# PRAKTIKUM 2: SAMPLING DAN KUANTISASI SINYAL
# ============================================

import numpy as np
import matplotlib.pyplot as plt

print("=== PRAKTIKUM 2: DIGITALISASI SINYAL ===")


# =====================================================
# FUNGSI DISPLAY GRID (Adaptasi Template Sebelumnya)
# =====================================================
def display_signal_grid(signals, titles, rows, cols):

    fig, axes = plt.subplots(rows, cols, figsize=(12,8))
    axes = axes.ravel()

    for i,(sig,title) in enumerate(zip(signals,titles)):
        axes[i].plot(sig[0], sig[1])
        axes[i].set_title(title)
        axes[i].grid(True)

    plt.tight_layout()
    plt.show()


# =====================================================
# FUNGSI ANALISIS SINYAL (Adaptasi analyze_image_properties)
# =====================================================
def analyze_signal_properties(signal, name="Signal"):

    x, y = signal

    print(f"\n===== ANALISIS {name} =====")
    print(f"Jumlah Sampel : {len(y)}")
    print(f"Nilai Min     : {np.min(y):.3f}")
    print(f"Nilai Max     : {np.max(y):.3f}")
    print(f"Mean          : {np.mean(y):.3f}")
    print(f"Std Dev       : {np.std(y):.3f}")


# =====================================================
# FUNGSI UTAMA DIGITALISASI
# =====================================================
def simulate_digitization(analog_func, sampling_rate, quantization_levels):

    """
    analog_func: fungsi analog (misal sin)
    sampling_rate: jumlah sampel
    quantization_levels: jumlah level kuantisasi
    """

    # ============================================
    # 1. SINYAL ANALOG (Kontinu)
    # ============================================
    x_cont = np.linspace(0, 2*np.pi, 1000)
    y_cont = analog_func(x_cont)

    analog_signal = (x_cont, y_cont)

    # ============================================
    # 2. SAMPLING
    # ============================================
    x_sample = np.linspace(0, 2*np.pi, sampling_rate)
    y_sample = analog_func(x_sample)

    sampled_signal = (x_sample, y_sample)

    # ============================================
    # 3. QUANTIZATION
    # ============================================
    y_min = np.min(y_sample)
    y_max = np.max(y_sample)

    step = (y_max - y_min) / quantization_levels
    y_quant = np.round((y_sample - y_min) / step) * step + y_min

    quantized_signal = (x_sample, y_quant)

    # ============================================
    # ANALISIS
    # ============================================
    analyze_signal_properties(analog_signal, "Analog")
    analyze_signal_properties(sampled_signal, "Sampled")
    analyze_signal_properties(quantized_signal, "Quantized")

    # ============================================
    # VISUALISASI
    # ============================================
    display_signal_grid(
        [analog_signal, sampled_signal, quantized_signal],
        ["Analog Signal", "Sampled Signal", "Quantized Signal"],
        3,1
    )

    # Overlay perbandingan
    plt.figure(figsize=(10,4))

    plt.plot(x_cont, y_cont, label="Analog", alpha=0.6)
    plt.scatter(x_sample, y_quant, label="Digital", color="red")

    plt.title("Perbandingan Analog vs Digital")
    plt.legend()
    plt.grid(True)
    plt.show()

    return {
        "sampling_rate": sampling_rate,
        "quantization_levels": quantization_levels
    }

# =====================================================
# CONTOH PEMAKAIAN
# =====================================================
result = simulate_digitization(
    analog_func=np.sin,
    sampling_rate=30,
    quantization_levels=8
)

print(result)
