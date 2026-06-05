# Impulse Response Analyzer

A Python-based analysis tool for Room Impulse Responses (RIR). This script processes Mono, Stereo, and Ambisonics B-Format (AmbiX) WAV files, calculates key acoustic parameters, and generates a standalone HTML report with comprehensive visualizations. Check out the Example Analysis Report to take a look at the output of the script. 

Check out an example analysis here: [Link] (Example_Analysis_Report.html)

> **Note** > This script was developed experimentally with the assistance of AI (Google Gemini). While it implements standard acoustic formulas, the calculated acoustic and spatial parameters **require further validation and rigorous testing**. Do not use this tool for critical architectural acoustics, certified measurements, or scientific publications without prior verification of its accuracy.

## Features

The tool automatically detects the channel format of the loaded WAV file (Mono, Stereo, or 4-channel AmbiX) and applies the appropriate analysis methods.

**Acoustic Parameters (Broadband / Mono)**
* **Reverberation Times:** EDT, T20, T30 (includes ISO 3382 noise floor truncation).
* **Clarity & Definition:** C50 (Speech), C80 (Music), D50.
* **Tonal Balance:** Bass Ratio (BR) and Treble Ratio (TR) using octave-band filtering.
* **Time Structure:** Center Time (Ts) and First Reflection Gap.
* **Accurate Onset Detection:** Searches backwards from the absolute peak to identify the true arrival of the direct sound.

**Spatial Analysis**
* **Stereo:** Interaural Cross-Correlation (IACC) to evaluate spatial envelopment.
* **Ambisonics (AmbiX ACN: W, Y, Z, X):** * Early Lateral Fraction (LF80).
  * Spatio-Temporal Directional Mapping (Horizontal 2D & Spherical 3D Mollweide Projections).

**Visualizations**
* Waveform with Onset Marker
* Broadband Energy Decay Curve (Schroeder Integration)
* Frequency Response (FFT with 1/3 Octave Smoothing)
* Spectrogram (Top-Down Waterfall)
* Octave Band RT60 Bar Chart (125 Hz – 4 kHz)
* Directional Energy Heatmaps (Ambisonics only)

Find here a more detailed description of all acoustic parameters that are getting analyzed: [Link] (IR Analyzer_Acoustic Parameters Guide.html)

## System Requirements & Installation

This script is optimized for Windows. It requires **Python 3** and a few standard scientific libraries.

1. Install Python (if not already installed).
2. Open your Command Prompt (CMD) or PowerShell.
3. Install the required dependencies:

```bash
pip install numpy scipy matplotlib
```
(Note: The tkinter library used for the file dialog is included in the standard Windows Python installation).

## Usage
1. Run the script from your terminal:
```bash
python "IR Analyzer.py"
```
2. A standard Windows file dialog will open.
3. Select the .wav file you want to analyze.
4. The script will process the file and create a file named analysis_report_[filename].html in the same directory.
5. Once finished, the HTML report will automatically open in your default web browser.

## Output
The generated HTML report is completely standalone—it does not require an internet connection. All plots are embedded directly into the file as Base64 strings, making it easy to share. Alongside the calculated data, the report includes brief explanations and typical acoustic target values for every parameter to help contextualize the results.
