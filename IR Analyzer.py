# -*- coding: utf-8 -*-

# =============================================================================
# Impulse Response Analyzer
#
# Author: Gemini (Coding-Assistent)
# Description: This script analyzes a room impulse response from a .wav file.
# It calculates key acoustic parameters and generates an HTML report with plots.
# Version: 3.2 (Improved report titles and metric clarification)
# =============================================================================

import os
import tkinter as tk
from tkinter import filedialog
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from scipy.io import wavfile
from scipy.fft import rfft, rfftfreq
import base64
from io import BytesIO

# --- Core Analysis Functions (No changes in this section) ---

def find_ir_start(signal, threshold_db=-20):
    """Find the start of the impulse response based on a threshold."""
    normalized_signal = signal / np.max(np.abs(signal))
    threshold_linear = 10**(threshold_db / 20)
    start_index = np.argmax(np.abs(normalized_signal) > threshold_linear)
    return start_index if start_index > 0 else 0

def calculate_schroeder_decay(ir, fs):
    """Calculates the Schroeder integral (energy decay curve) of the impulse response."""
    start_index = find_ir_start(ir)
    ir_trimmed = ir[start_index:]
    schroeder_integral = np.cumsum(ir_trimmed**2, axis=0)
    schroeder_integral = schroeder_integral[-1] - schroeder_integral
    with np.errstate(divide='ignore'):
        schroeder_db = 10 * np.log10(schroeder_integral / np.max(schroeder_integral))
    schroeder_db[np.isneginf(schroeder_db)] = -120
    return schroeder_db

def calculate_decay_times(schroeder_db, fs):
    """Calculates T20, T30, and EDT from the Schroeder decay curve."""
    metrics = {}
    time_axis = np.arange(len(schroeder_db)) / fs
    
    def fit_and_calc(start_db, end_db):
        try:
            start_idx = np.nanargmax(schroeder_db <= start_db)
            end_idx = np.nanargmax(schroeder_db <= end_db)
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
    """Calculates Clarity (C50, C80) and Definition (D50)."""
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
    """Find the time of the first reflection relative to the direct sound."""
    start_index = find_ir_start(ir)
    search_start = start_index + int(0.002 * fs)
    search_window = ir[search_start : start_index + int(0.05 * fs)]
    if len(search_window) > 0:
        reflection_index = np.argmax(np.abs(search_window))
        return (reflection_index + int(0.002 * fs)) / fs * 1000
    return None

def octave_smooth(freqs, mags, frac=3):
    """Performs 1/N-octave smoothing on a frequency response."""
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

# --- Plotting Functions (No changes in this section) ---

def plot_to_base64(fig):
    """Converts a Matplotlib figure to a base64 encoded string for HTML embedding."""
    buf = BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight')
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode('utf-8')

def plot_waveform(ir, fs):
    """Plots the impulse response waveform."""
    fig, ax = plt.subplots(figsize=(10, 4))
    time = np.arange(len(ir)) / fs
    ax.plot(time, ir, color='#007ACC')
    ax.set_title('Impulse Response Waveform')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Amplitude')
    ax.grid(True, linestyle='--', alpha=0.6)
    return plot_to_base64(fig)

def plot_decay_curve(schroeder_db, fs):
    """Plots the Schroeder decay curve."""
    fig, ax = plt.subplots(figsize=(10, 4))
    time = np.arange(len(schroeder_db)) / fs
    ax.plot(time, schroeder_db, color='#007ACC')
    ax.set_title('Schroeder Energy Decay Curve')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Energy (dB)')
    ax.set_ylim(-100, 5)
    ax.grid(True, linestyle='--', alpha=0.6)
    return plot_to_base64(fig)
    
def plot_frequency_response(ir, fs):
    """Plots the frequency response, raw and with 1/3-octave smoothing."""
    N = len(ir)
    yf = rfft(ir)
    xf = rfftfreq(N, 1 / fs)
    mags_db = 20 * np.log10(np.abs(yf))
    smoothed_mags_db = octave_smooth(xf, mags_db, frac=3)
    
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(xf, mags_db, color='#89CFF0', alpha=0.5, label='Raw')
    ax.plot(xf, smoothed_mags_db, color='#005A9C', linewidth=2, label='1/3 Octave Smoothed')
    ax.set_title('Frequency Response')
    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel('Magnitude (dB)')
    ax.set_xscale('log')
    ax.set_xlim(20, 20000)
    
    ax.set_xticks([20, 30, 40, 50, 60, 80, 100, 200, 300, 400, 500, 1000, 2000, 5000, 10000, 20000])
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, pos: f'{x/1000:.0f}k' if x >= 1000 else f'{x:.0f}'))
    
    ax.grid(True, which='both', linestyle='--', alpha=0.6)
    ax.legend()
    return plot_to_base64(fig)

# --- File Handling and Report Generation ---

def select_wav_file():
    """Opens a file dialog to select a .wav file."""
    root = tk.Tk()
    root.withdraw()
    filepath = filedialog.askopenfilename(title="Select Impulse Response File", filetypes=[("WAV files", "*.wav")])
    return filepath

def generate_html_report(filename, base_filename, params, plots):
    """Generates an HTML report with the analysis results and reference values."""
    edt = f"{params['EDT']:.2f} s" if params.get('EDT') is not None else "N/A"
    reverb_time = f"{params['T30']:.2f} s" if params.get('T30') is not None else (f"{params['T20']:.2f} s" if params.get('T20') is not None else "N/A")
    c50 = f"{params['C50']:.2f} dB" if params.get('C50') is not None else "N/A"
    c80 = f"{params['C80']:.2f} dB" if params.get('C80') is not None else "N/A"
    d50 = f"{params['D50']:.2f} %" if params.get('D50') is not None else "N/A"
    first_reflection = f"{params['first_reflection']:.2f} ms" if params.get('first_reflection') is not None else "N/A"
    
    html_content = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Analysis Report: {base_filename}</title>
        <style>
            body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif; line-height: 1.6; color: #333; max-width: 960px; margin: 20px auto; background-color: #f9f9f9; }}
            h1, h2 {{ color: #005A9C; border-bottom: 2px solid #007ACC; padding-bottom: 10px; }}
            .container {{ background: #fff; padding: 20px; border-radius: 8px; box-shadow: 0 4px 8px rgba(0,0,0,0.1); }}
            .filename {{ font-family: monospace; background: #eee; padding: 2px 6px; border-radius: 4px; color: #c7254e; }}
            .results-table {{ width: 100%; border-collapse: collapse; margin-top: 20px; }}
            .results-table th, .results-table td {{ border: 1px solid #ddd; padding: 12px; text-align: left; }}
            .results-table th {{ background-color: #f2f2f2; }}
            .ref-value {{ color: #555; font-size: 0.9em; }}
            .plot {{ margin-top: 30px; text-align: center; }}
            .plot img {{ max-width: 100%; height: auto; border: 1px solid #ddd; border-radius: 4px; }}
            footer {{ margin-top: 30px; text-align: center; font-size: 0.9em; color: #777; }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>Impulse Response Analysis Report - {base_filename}</h1>
            <p>Analysis for file: <span class="filename">{os.path.basename(filename)}</span></p>

            <h2>Acoustic Parameters</h2>
            <table class="results-table">
                <tr><th>Parameter</th><th>Value</th><th>Description & Reference Values</th></tr>
                <tr>
                    <td><b>EDT (Early Decay Time)</b></td>
                    <td>{edt}</td>
                    <td>Reverberation time based on the first 10 dB of decay. Relates to perceived reverberance.</td>
                </tr>
                <tr>
                    <td><b>RT60 (from T30)</b></td>
                    <td>{reverb_time}</td>
                    <td>The standard reverberation time (RT60) extrapolated from the T30 measurement (-5 dB to -35 dB), which is robust against noise.</td>
                </tr>
                <tr>
                    <td><b>C50 (Clarity for Speech)</b></td>
                    <td>{c50}</td>
                    <td>Ratio of early (0-50ms) to late energy. <span class="ref-value">(Reference: > 2 dB for good speech clarity)</span></td>
                </tr>
                 <tr>
                    <td><b>C80 (Clarity for Music)</b></td>
                    <td>{c80}</td>
                    <td>Ratio of early (0-80ms) to late energy. <span class="ref-value">(Reference: -2 dB to +4 dB, depending on music style)</span></td>
                </tr>
                <tr>
                    <td><b>D50 (Definition)</b></td>
                    <td>{d50}</td>
                    <td>Percentage of early energy (0-50ms) relative to total energy. <span class="ref-value">(Reference: > 50% for good speech intelligibility)</span></td>
                </tr>
                 <tr>
                    <td><b>First Reflection</b></td>
                    <td>{first_reflection}</td>
                    <td>Time gap between the direct sound and the first significant reflection.</td>
                </tr>
            </table>

            <h2>Plots</h2>
            <div class="plot"><h3>Waveform</h3><img src="data:image/png;base64,{plots['waveform']}" alt="Waveform Plot"></div>
            <div class="plot"><h3>Energy Decay Curve</h3><img src="data:image/png;base64,{plots['decay_curve']}" alt="Decay Curve Plot"></div>
            <div class="plot"><h3>Frequency Response</h3><img src="data:image/png;base64,{plots['frequency_response']}" alt="Frequency Response Plot"></div>
        </div>
        <footer>Report generated by the Python Impulse Response Analyzer.</footer>
    </body>
    </html>
    """
    
    report_path = f"analysis_report_{base_filename}.html"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(html_content)
    return report_path

# --- Main Execution Logic (No changes in this section) ---

def main():
    """Main function to run the analyzer."""
    filepath = select_wav_file()
    if not filepath:
        print("No file selected. Exiting.")
        return

    print(f"Loading and analyzing file: {filepath}...")
    try:
        base_filename = os.path.splitext(os.path.basename(filepath))[0]
        fs, signal = wavfile.read(filepath)
        if signal.ndim > 1: signal = signal.mean(axis=1)
        signal = signal / np.max(np.abs(signal))
    except Exception as e:
        print(f"Error reading WAV file: {e}")
        return

    schroeder_db = calculate_schroeder_decay(signal, fs)
    decay_times = calculate_decay_times(schroeder_db, fs)
    clarity_def = calculate_clarity_and_definition(signal, fs)
    first_reflection = find_first_reflection(signal, fs)
    all_params = {**decay_times, **clarity_def, "first_reflection": first_reflection}

    plots = {
        'waveform': plot_waveform(signal, fs),
        'decay_curve': plot_decay_curve(schroeder_db, fs),
        'frequency_response': plot_frequency_response(signal, fs)
    }

    report_file = generate_html_report(filepath, base_filename, all_params, plots)
    
    print("-" * 50)
    print("Analysis complete!")
    print(f"Report saved to: {os.path.abspath(report_file)}")
    print("-" * 50)
    
    try:
        os.startfile(os.path.abspath(report_file))
    except AttributeError: # For macOS/Linux
        import subprocess, sys
        opener = "open" if sys.platform == "darwin" else "xdg-open"
        subprocess.call([opener, os.path.abspath(report_file)])

if __name__ == "__main__":
    main()