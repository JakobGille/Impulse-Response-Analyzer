# -*- coding: utf-8 -*-

# =============================================================================
# Impulse Response Analyzer
#
# Description: Analyzes room impulse responses (Mono, Stereo, AmbiX B-Format).
# Implements ISO 3382 truncation, onset detection, spatial metrics,
# Spatio-Temporal Heatmap, and a 3D Mollweide Projection for Ambisonics.
# Generates an HTML report with localized English target values.

#Created by: Jakob Gille & Gemini AI
#Date: 03.06.2026
#Version: 1.0.0 - Initial release
# =============================================================================

import os
import warnings
import tkinter as tk
from tkinter import filedialog
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from scipy.io import wavfile
from scipy.io.wavfile import WavFileWarning
from scipy.fft import rfft, rfftfreq
from scipy.signal import spectrogram, butter, sosfiltfilt
import base64
from io import BytesIO

# Suppress metadata chunk warnings
warnings.filterwarnings("ignore", category=WavFileWarning)

# --- Core Analysis Functions ---

def find_ir_start(signal, threshold_db=-20):
    peak_idx = np.argmax(np.abs(signal))
    peak_val = np.abs(signal[peak_idx])
    threshold = peak_val * (10 ** (threshold_db / 20))
    
    for i in range(peak_idx, -1, -1):
        if np.abs(signal[i]) < threshold:
            return i
    return 0

def calculate_center_time(ir, fs):
    start_index = find_ir_start(ir)
    ir_trimmed = ir[start_index:]
    ir_sq = ir_trimmed**2
    time_axis = np.arange(len(ir_sq)) / fs
    
    energy = np.sum(ir_sq)
    if energy == 0: return None
    ts = np.sum(time_axis * ir_sq) / energy
    return ts * 1000

def get_octave_filter(center_freq, fs, order=4):
    nyq = 0.5 * fs
    lower = center_freq / np.sqrt(2)
    upper = center_freq * np.sqrt(2)
    if upper >= nyq: upper = nyq - 0.1
    sos = butter(order, [lower, upper], btype='bandpass', output='sos', fs=fs)
    return sos

def calculate_schroeder_decay(ir, fs):
    start_index = find_ir_start(ir)
    ir_trimmed = ir[start_index:]
    ir_sq = ir_trimmed**2

    tail_length = max(1, int(len(ir_sq) * 0.1))
    noise_floor = np.mean(ir_sq[-tail_length:])

    window_len = max(1, int(0.01 * fs))
    window = np.ones(window_len) / window_len
    ir_sq_smoothed = np.convolve(ir_sq, window, mode='same')

    threshold = noise_floor * 2
    intersection_idx = len(ir_sq) - 1
    
    for i in range(len(ir_sq_smoothed) - 1, -1, -1):
        if ir_sq_smoothed[i] > threshold:
            intersection_idx = min(len(ir_sq) - 1, i + int(0.05 * fs))
            break

    ir_truncated = ir_sq[:intersection_idx]
    schroeder_integral_trunc = np.cumsum(ir_truncated[::-1])[::-1]

    pad_length = len(ir_sq) - intersection_idx
    last_val = schroeder_integral_trunc[-1] if len(schroeder_integral_trunc) > 0 else 1e-12
    pad_values = np.full(pad_length, last_val)
    schroeder_integral = np.concatenate((schroeder_integral_trunc, pad_values))

    with np.errstate(divide='ignore', invalid='ignore'):
        schroeder_db = 10 * np.log10(schroeder_integral / np.max(schroeder_integral))
    
    schroeder_db[np.isneginf(schroeder_db) | np.isnan(schroeder_db)] = -120
    return schroeder_db

def calculate_t30(schroeder_db, fs):
    time_axis = np.arange(len(schroeder_db)) / fs
    try:
        start_idx = np.nanargmax(schroeder_db <= -5)
        end_idx = np.nanargmax(schroeder_db <= -35)
        if start_idx == 0 and schroeder_db[0] > -5: return None
        if end_idx == 0 and schroeder_db[0] > -35: return None
        if end_idx > start_idx:
            poly = np.polyfit(time_axis[start_idx:end_idx], schroeder_db[start_idx:end_idx], 1)
            return -60 / poly[0]
    except (ValueError, np.linalg.LinAlgError):
        return None
    return None

def calculate_octave_rt60(ir, fs):
    bands = [125, 250, 500, 1000, 2000, 4000]
    rt60_bands = {}
    for f_c in bands:
        sos = get_octave_filter(f_c, fs)
        filtered_ir = sosfiltfilt(sos, ir)
        schroeder_db = calculate_schroeder_decay(filtered_ir, fs)
        t30 = calculate_t30(schroeder_db, fs)
        rt60_bands[f_c] = t30
    return rt60_bands

def calculate_ratios(rt60_bands):
    try:
        t_low = rt60_bands[125] + rt60_bands[250]
        t_mid = rt60_bands[500] + rt60_bands[1000]
        t_high = rt60_bands[2000] + rt60_bands[4000]
        br = t_low / t_mid if t_mid and t_low else None
        tr = t_high / t_mid if t_mid and t_high else None
        return br, tr
    except (KeyError, TypeError):
        return None, None

def calculate_decay_times(schroeder_db, fs):
    metrics = {}
    time_axis = np.arange(len(schroeder_db)) / fs
    def fit_and_calc(start_db, end_db):
        try:
            start_idx = np.nanargmax(schroeder_db <= start_db)
            end_idx = np.nanargmax(schroeder_db <= end_db)
            if start_idx == 0 and schroeder_db[0] > start_db: return None
            if end_idx == 0 and schroeder_db[0] > end_db: return None
            if end_idx > start_idx:
                poly = np.polyfit(time_axis[start_idx:end_idx], schroeder_db[start_idx:end_idx], 1)
                return -60 / poly[0]
        except (ValueError, np.linalg.LinAlgError):
            return None
        return None
    metrics['EDT'] = fit_and_calc(0, -10)
    metrics['T20'] = fit_and_calc(-5, -25)
    metrics['T30'] = fit_and_calc(-5, -35)
    return metrics

def calculate_clarity_and_definition(ir, fs):
    metrics = {}
    start_index = find_ir_start(ir)
    ir_trimmed = ir[start_index:]
    squared_ir = ir_trimmed**2
    ms_50 = int(0.05 * fs)
    ms_80 = int(0.08 * fs)
    
    energy_early_50 = np.sum(squared_ir[:ms_50])
    energy_late_50 = np.sum(squared_ir[ms_50:])
    energy_early_80 = np.sum(squared_ir[:ms_80])
    energy_late_80 = np.sum(squared_ir[ms_80:])
    total_energy = np.sum(squared_ir)

    metrics['C50'] = 10 * np.log10(energy_early_50 / energy_late_50) if energy_late_50 > 0 else None
    metrics['C80'] = 10 * np.log10(energy_early_80 / energy_late_80) if energy_late_80 > 0 else None
    metrics['D50'] = 100 * (energy_early_50 / total_energy) if total_energy > 0 else None
    return metrics

def find_first_reflection(ir, fs):
    start_index = find_ir_start(ir)
    search_start = start_index + int(0.002 * fs)
    search_window = ir[search_start : start_index + int(0.05 * fs)]
    if len(search_window) > 0:
        reflection_index = np.argmax(np.abs(search_window))
        return (reflection_index + int(0.002 * fs)) / fs * 1000
    return None

# --- Spatial Functions ---

def calculate_iacc(left, right, fs):
    start_l = find_ir_start(left)
    start_r = find_ir_start(right)
    start = min(start_l, start_r)
    end = start + int(0.080 * fs)
    
    if end > len(left): return None
    
    l_early = left[start:end]
    r_early = right[start:end]
    
    cc = np.correlate(l_early, r_early, mode='full')
    energy_l = np.sum(l_early**2)
    energy_r = np.sum(r_early**2)
    
    if energy_l == 0 or energy_r == 0: return None
    return np.max(np.abs(cc)) / np.sqrt(energy_l * energy_r)

def calculate_lf(w, y, fs):
    start = find_ir_start(w)
    idx_5ms = start + int(0.005 * fs)
    idx_80ms = start + int(0.080 * fs)
    
    if idx_80ms > len(w): return None
    
    y_early = y[idx_5ms:idx_80ms]
    w_early = w[start:idx_80ms]
    
    energy_y = np.sum(y_early**2)
    energy_w = np.sum(w_early**2)
    
    if energy_w == 0: return None
    return energy_y / energy_w

def octave_smooth(freqs, mags, frac=3):
    octave_factor = 2**(1/(2*frac))
    smooth_mags = np.zeros_like(mags)
    for i, f in enumerate(freqs):
        if f == 0: continue
        lower_bound = f / octave_factor
        upper_bound = f * octave_factor
        band_indices = np.where((freqs >= lower_bound) & (freqs <= upper_bound))
        if len(band_indices[0]) > 0:
            smooth_mags[i] = np.mean(mags[band_indices])
    return smooth_mags

# --- Plotting Functions ---

def plot_to_base64(fig):
    buf = BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight', dpi=120)
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode('utf-8')

def plot_rt60_bands(rt60_bands):
    fig, ax = plt.subplots(figsize=(10, 4))
    freqs = list(rt60_bands.keys())
    values = [v if v is not None else 0 for v in rt60_bands.values()]
    
    ax.bar([str(f) for f in freqs], values, color='#007ACC', width=0.6)
    for i, v in enumerate(values):
        if v > 0: ax.text(i, v + 0.05, f"{v:.2f}s", ha='center', va='bottom', fontsize=9)
        
    ax.set_title('Reverberation Time (T30) per Octave Band')
    ax.set_xlabel('Octave Band Center Frequency (Hz)')
    ax.set_ylabel('RT60 (s)')
    ax.set_ylim(0, max(values) * 1.2 if max(values) > 0 else 1)
    ax.grid(axis='y', linestyle='--', alpha=0.6)
    return plot_to_base64(fig)

def plot_waveform(ir, fs):
    fig, ax = plt.subplots(figsize=(10, 4))
    time = np.arange(len(ir)) / fs
    ax.plot(time, ir, color='#007ACC')
    
    start_idx = find_ir_start(ir)
    ax.axvline(start_idx / fs, color='r', linestyle=':', label='Detected Onset')
    
    ax.set_title('Impulse Response Waveform (Mono/W-Channel)')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Amplitude')
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.legend()
    return plot_to_base64(fig)

def plot_decay_curve(schroeder_db, fs):
    fig, ax = plt.subplots(figsize=(10, 4))
    time = np.arange(len(schroeder_db)) / fs
    ax.plot(time, schroeder_db, color='#007ACC')
    ax.set_title('Broadband Schroeder Energy Decay Curve (Truncated)')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Energy (dB)')
    ax.set_ylim(-100, 5)
    ax.grid(True, linestyle='--', alpha=0.6)
    return plot_to_base64(fig)
    
def plot_frequency_response(ir, fs):
    N = len(ir)
    yf = rfft(ir)
    xf = rfftfreq(N, 1 / fs)
    mags_db = 20 * np.log10(np.abs(yf) + 1e-10)
    smoothed_mags_db = octave_smooth(xf, mags_db, frac=3)
    
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(xf, mags_db, color='#89CFF0', alpha=0.5, label='Raw')
    ax.plot(xf, smoothed_mags_db, color='#005A9C', linewidth=2, label='1/3 Octave Smoothed')
    ax.set_title('Frequency Response')
    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel('Magnitude (dB)')
    ax.set_xscale('log')
    ax.set_xlim(20, 20000)
    ax.set_xticks([20, 50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000])
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, pos: f'{x/1000:.0f}k' if x >= 1000 else f'{x:.0f}'))
    ax.grid(True, which='both', linestyle='--', alpha=0.6)
    ax.legend()
    return plot_to_base64(fig)

def plot_waterfall(ir, fs):
    fig, ax = plt.subplots(figsize=(10, 5))
    start_index = find_ir_start(ir)
    ir_trimmed = ir[start_index:]
    nperseg = int(fs * 0.02)
    noverlap = int(nperseg * 0.8)
    
    f, t, Sxx = spectrogram(ir_trimmed, fs, nperseg=nperseg, noverlap=noverlap)
    Sxx_db = 10 * np.log10(Sxx + 1e-12)
    
    cax = ax.pcolormesh(t, f, Sxx_db, shading='gouraud', cmap='inferno', vmin=np.max(Sxx_db)-60, vmax=np.max(Sxx_db))
    ax.set_title('Spectrogram (Top-Down Waterfall, 60dB Range)')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Frequency (Hz)')
    ax.set_ylim(20, 20000)
    ax.set_yscale('symlog', linthresh=1000)
    ax.yaxis.set_major_formatter(mticker.ScalarFormatter())
    ax.set_yticks([20, 100, 500, 1000, 5000, 10000, 20000])
    fig.colorbar(cax, ax=ax, label='Magnitude (dB)')
    return plot_to_base64(fig)

def plot_spatio_temporal_heatmap(w, x, y, fs):
    start_index = find_ir_start(w)
    end_index = start_index + int(0.100 * fs)
    if end_index > len(w): end_index = len(w)

    w_trim = w[start_index:end_index]
    x_trim = x[start_index:end_index]
    y_trim = y[start_index:end_index]

    i_x = w_trim * x_trim
    i_y = w_trim * y_trim

    azimuth = np.degrees(np.arctan2(i_y, i_x))
    intensity_mag = np.sqrt(i_x**2 + i_y**2)

    time_axis = np.arange(len(w_trim)) / fs * 1000

    time_bins = 100
    angle_bins = 72

    H, xedges, yedges = np.histogram2d(time_axis, azimuth, bins=[time_bins, angle_bins],
                                       range=[[0, 100], [-180, 180]], weights=intensity_mag)

    H_db = 10 * np.log10(H.T + 1e-12)
    H_db = H_db - np.max(H_db)

    fig, ax = plt.subplots(figsize=(10, 5))
    cax = ax.pcolormesh(xedges, yedges, H_db, shading='auto', cmap='magma', vmin=-40, vmax=0)

    ax.set_title('Spatio-Temporal Heatmap (Horizontal Reflections 0-100ms)')
    ax.set_xlabel('Time (ms)')
    ax.set_ylabel('Azimuth (Degrees)')
    ax.set_yticks([-180, -90, 0, 90, 180])
    ax.set_yticklabels(['-180° (Rear)', '-90° (Right)', '0° (Front)', '+90° (Left)', '+180° (Rear)'])
    ax.grid(True, linestyle='--', alpha=0.3)

    fig.colorbar(cax, ax=ax, label='Relative Intensity (dB)')
    return plot_to_base64(fig)

def plot_mollweide_heatmap(w, x, y, z, fs):
    """Generates a 3D Mollweide projection of early reflection intensity."""
    start_index = find_ir_start(w)
    end_index = start_index + int(0.100 * fs)
    if end_index > len(w): end_index = len(w)

    w_trim = w[start_index:end_index]
    x_trim = x[start_index:end_index]
    y_trim = y[start_index:end_index]
    z_trim = z[start_index:end_index]

    i_x = w_trim * x_trim
    i_y = w_trim * y_trim
    i_z = w_trim * z_trim

    intensity_mag = np.sqrt(i_x**2 + i_y**2 + i_z**2)
    intensity_mag_safe = np.where(intensity_mag == 0, 1e-12, intensity_mag)

    azimuth = np.arctan2(i_y, i_x)
    elevation = np.arcsin(np.clip(i_z / intensity_mag_safe, -1.0, 1.0))

    azimuth_bins = 72
    elevation_bins = 36

    H, lon_edges, lat_edges = np.histogram2d(azimuth, elevation, bins=[azimuth_bins, elevation_bins],
                                             range=[[-np.pi, np.pi], [-np.pi/2, np.pi/2]], weights=intensity_mag)

    H_db = 10 * np.log10(H.T + 1e-12)
    H_db = H_db - np.max(H_db)

    fig = plt.figure(figsize=(10, 5))
    ax = fig.add_subplot(111, projection='mollweide')

    lon, lat = np.meshgrid(lon_edges, lat_edges)
    cax = ax.pcolormesh(lon, lat, H_db, cmap='magma', vmin=-40, vmax=0, shading='auto')

    ax.set_title('3D Directional Energy Mapping (Mollweide Projection, 0-100ms)', pad=20)
    ax.grid(True, linestyle='--', alpha=0.5)

    fig.colorbar(cax, ax=ax, label='Relative Intensity (dB)', orientation='vertical', fraction=0.046, pad=0.04)
    return plot_to_base64(fig)

# --- File Handling and Report Generation ---

def select_wav_file():
    root = tk.Tk()
    root.withdraw()
    filepath = filedialog.askopenfilename(title="Select Impulse Response File", filetypes=[("WAV files", "*.wav")])
    return filepath

def generate_html_report(filename, base_filename, params, plots, channel_config):
    edt = f"{params['EDT']:.2f} s" if params.get('EDT') else "N/A"
    reverb_time = f"{params['T30']:.2f} s" if params.get('T30') else (f"{params['T20']:.2f} s" if params.get('T20') else "N/A")
    c50 = f"{params['C50']:.2f} dB" if params.get('C50') else "N/A"
    c80 = f"{params['C80']:.2f} dB" if params.get('C80') else "N/A"
    d50 = f"{params['D50']:.2f} %" if params.get('D50') else "N/A"
    first_reflection = f"{params['first_reflection']:.2f} ms" if params.get('first_reflection') else "N/A"
    ts = f"{params['Ts']:.1f} ms" if params.get('Ts') else "N/A"
    br = f"{params['BR']:.2f}" if params.get('BR') else "N/A"
    tr = f"{params['TR']:.2f}" if params.get('TR') else "N/A"
    iacc = f"{params.get('IACC', 0):.2f}" if params.get('IACC') else "N/A"
    lf = f"{params.get('LF', 0):.2f}" if params.get('LF') else "N/A"
    
    channel_note = ""
    spatial_section = ""
    spatial_plots_html = ""
    
    if channel_config == "Stereo":
        channel_note = "<p style='color: #c7254e; font-weight: bold;'>Note: Stereo file detected. Mono-Mixdown used for broadband metrics. L/R used for IACC.</p>"
        spatial_section = f"""
        <h2>Spatial Parameters (Stereo)</h2>
        <table class="results-table">
            <tr><th>Parameter</th><th>Value</th></tr>
            <tr>
                <td><b>IACC (Early, 0-80ms)</b><span class="param-desc">Interaural Cross-Correlation. Target: &lt; 0.3 for excellent spatial envelopment, &gt; 0.6 indicates a narrow, localized sound.</span></td>
                <td>{iacc}</td>
            </tr>
        </table>
        """
    elif channel_config == "AmbiX":
        channel_note = "<p style='color: #c7254e; font-weight: bold;'>Note: AmbiX B-Format detected (ACN: W, Y, Z, X). Full 3D spatial analysis applied.</p>"
        spatial_section = f"""
        <h2>Spatial Parameters (Ambisonics)</h2>
        <table class="results-table">
            <tr><th>Parameter</th><th>Value</th></tr>
            <tr>
                <td><b>Early Lateral Fraction (LF80)</b><span class="param-desc">Ratio of early lateral energy. Target: 0.10 to 0.35 for good spatial envelopment.</span></td>
                <td>{lf}</td>
            </tr>
        </table>
        """
        
    if "spatial_heatmap" in plots:
        spatial_plots_html += f'<div class="plot"><h3>Horizontal Energy Mapping</h3><img src="data:image/png;base64,{plots["spatial_heatmap"]}" alt="Spatial Heatmap"></div>'
    if "mollweide_heatmap" in plots:
        spatial_plots_html += f'<div class="plot"><h3>3D Spherical Energy Mapping</h3><img src="data:image/png;base64,{plots["mollweide_heatmap"]}" alt="Mollweide Heatmap"></div>'

    html_content = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Analysis Report: {base_filename}</title>
        <style>
            body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Arial, sans-serif; line-height: 1.6; color: #333; max-width: 960px; margin: 20px auto; background-color: #f9f9f9; }}
            h1, h2 {{ color: #005A9C; border-bottom: 2px solid #007ACC; padding-bottom: 10px; }}
            .container {{ background: #fff; padding: 20px; border-radius: 8px; box-shadow: 0 4px 8px rgba(0,0,0,0.1); }}
            .filename {{ font-family: monospace; background: #eee; padding: 2px 6px; border-radius: 4px; color: #c7254e; }}
            .results-table {{ width: 100%; border-collapse: collapse; margin-top: 20px; }}
            .results-table th, .results-table td {{ border: 1px solid #ddd; padding: 12px; text-align: left; vertical-align: top; width: 50%; }}
            .results-table th {{ background-color: #f2f2f2; }}
            .param-desc {{ color: #777; font-size: 0.85em; display: block; margin-top: 4px; font-weight: normal; line-height: 1.4; }}
            .plot {{ margin-top: 30px; text-align: center; }}
            .plot img {{ max-width: 100%; height: auto; border: 1px solid #ddd; border-radius: 4px; }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>Impulse Response Analysis Report - {base_filename}</h1>
            <p>Analysis for file: <span class="filename">{os.path.basename(filename)}</span></p>
            {channel_note}

            <h2>Acoustic Parameters (Broadband / Mono)</h2>
            <table class="results-table">
                <tr><th>Parameter</th><th>Value</th></tr>
                <tr>
                    <td><b>Center Time (Ts)</b><span class="param-desc">Center of gravity of the IR. Target: &lt; 80 ms for speech, 80-150 ms for music. Lower values correlate with acoustic intimacy.</span></td>
                    <td>{ts}</td>
                </tr>
                <tr>
                    <td><b>Bass Ratio (BR)</b><span class="param-desc">Low to mid-frequency RT60 ratio. Target: 1.1 - 1.25 for concert halls (warmth). Values &gt; 1.5 indicate boominess.</span></td>
                    <td>{br}</td>
                </tr>
                <tr>
                    <td><b>Treble Ratio (TR)</b><span class="param-desc">High to mid-frequency RT60 ratio. Target: 0.8 - 1.0. Typically falls below 1.0 due to air absorption.</span></td>
                    <td>{tr}</td>
                </tr>
                <tr>
                    <td><b>EDT (Early Decay Time)</b><span class="param-desc">Perceived reverberance. Target: 0.2s - 0.4s for control rooms, 1.0s - 2.0s for concert halls.</span></td>
                    <td>{edt}</td>
                </tr>
                <tr>
                    <td><b>RT60 (from T30)</b><span class="param-desc">Standard broad-band reverberation time. Targets depend strictly on room volume and intended purpose.</span></td>
                    <td>{reverb_time}</td>
                </tr>
                <tr>
                    <td><b>C50 (Speech Clarity)</b><span class="param-desc">Early-to-late energy ratio. Target: &gt; 2 dB for excellent speech intelligibility.</span></td>
                    <td>{c50}</td>
                </tr>
                <tr>
                    <td><b>C80 (Music Clarity)</b><span class="param-desc">Early-to-late energy ratio. Target: -2 dB to +4 dB depending on musical genre (higher for rhythmic, lower for choral).</span></td>
                    <td>{c80}</td>
                </tr>
                <tr>
                    <td><b>D50 (Definition)</b><span class="param-desc">Percentage of early energy. Target: &gt; 50% for good speech definition.</span></td>
                    <td>{d50}</td>
                </tr>
                <tr>
                    <td><b>First Reflection</b><span class="param-desc">Time delay between direct sound and first boundary reflection. Target: &gt; 15 - 20 ms to avoid comb filtering.</span></td>
                    <td>{first_reflection}</td>
                </tr>
            </table>

            {spatial_section}

            <h2>Plots</h2>
            {spatial_plots_html}
            <div class="plot"><h3>Octave Band Reverberation Time</h3><img src="data:image/png;base64,{plots['rt60_bands']}" alt="RT60 Bands"></div>
            <div class="plot"><h3>Spectrogram / Waterfall</h3><img src="data:image/png;base64,{plots['waterfall']}" alt="Waterfall"></div>
            <div class="plot"><h3>Waveform</h3><img src="data:image/png;base64,{plots['waveform']}" alt="Waveform"></div>
            <div class="plot"><h3>Broadband Energy Decay Curve</h3><img src="data:image/png;base64,{plots['decay_curve']}" alt="Decay Curve"></div>
            <div class="plot"><h3>Frequency Response</h3><img src="data:image/png;base64,{plots['frequency_response']}" alt="Frequency Response"></div>
        </div>
    </body>
    </html>
    """
    
    report_path = f"analysis_report_{base_filename}.html"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(html_content)
    return report_path

# --- Main Execution ---

def main():
    filepath = select_wav_file()
    if not filepath: return

    print(f"Analyzing: {filepath}...")
    try:
        base_filename = os.path.splitext(os.path.basename(filepath))[0]
        fs, signal = wavfile.read(filepath)
        
        signal = signal.astype(np.float64)
        signal = signal / np.max(np.abs(signal))
        
        channel_config = "Mono"
        if signal.ndim > 1:
            if signal.shape[1] >= 4:
                channel_config = "AmbiX"
                mono_signal = signal[:, 0] # W
                y_channel = signal[:, 1]   # Y
                z_channel = signal[:, 2]   # Z
                x_channel = signal[:, 3]   # X
            elif signal.shape[1] == 2:
                channel_config = "Stereo"
                mono_signal = signal.mean(axis=1)
                left = signal[:, 0]
                right = signal[:, 1]
            else:
                mono_signal = signal[:, 0] # Fallback for 3 channels
        else:
            mono_signal = signal
        
    except Exception as e:
        print(f"Error: {e}")
        return

    schroeder_db = calculate_schroeder_decay(mono_signal, fs)
    decay_times = calculate_decay_times(schroeder_db, fs)
    clarity_def = calculate_clarity_and_definition(mono_signal, fs)
    first_reflection = find_first_reflection(mono_signal, fs)
    ts = calculate_center_time(mono_signal, fs)
    
    rt60_bands = calculate_octave_rt60(mono_signal, fs)
    br, tr = calculate_ratios(rt60_bands)
    
    all_params = {
        **decay_times, **clarity_def, "first_reflection": first_reflection, 
        "Ts": ts, "BR": br, "TR": tr
    }

    plots = {
        'rt60_bands': plot_rt60_bands(rt60_bands),
        'waveform': plot_waveform(mono_signal, fs),
        'decay_curve': plot_decay_curve(schroeder_db, fs),
        'frequency_response': plot_frequency_response(mono_signal, fs),
        'waterfall': plot_waterfall(mono_signal, fs)
    }

    if channel_config == "Stereo":
        all_params['IACC'] = calculate_iacc(left, right, fs)
    elif channel_config == "AmbiX":
        all_params['LF'] = calculate_lf(mono_signal, y_channel, fs)
        plots['spatial_heatmap'] = plot_spatio_temporal_heatmap(mono_signal, x_channel, y_channel, fs)
        plots['mollweide_heatmap'] = plot_mollweide_heatmap(mono_signal, x_channel, y_channel, z_channel, fs)

    report_file = generate_html_report(filepath, base_filename, all_params, plots, channel_config)
    print(f"Done. Report saved to: {os.path.abspath(report_file)}")
    os.startfile(os.path.abspath(report_file))

if __name__ == "__main__":
    main()