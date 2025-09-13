# Full 3D Unconventional & Black-Oil Reservoir Simulator

This repository contains an interactive web application for performing 3D reservoir simulations for both unconventional (shale) and conventional black-oil assets. The application is built with Python using the Streamlit framework for the user interface and Plotly for visualizations.

## 🚀 Key Features

- **Dual Simulation Modes**: Supports both unconventional multi-stage fractured well (MSFW) models and traditional black-oil models.
- **Geological Presets**: Includes pre-loaded presets for major North American shale plays like the Permian, Eagle Ford, and Marcellus.
- **3D Property Modeling**: Interactively generate and visualize 3D permeability (kx, ky) and porosity (ϕ) volumes.
- **Interactive Visualization**:
  - 3D Isosurface Viewer for properties and results (e.g., pressure).
  - 2D Slice Viewer for detailed cross-section analysis.
  - DFN (Discrete Fracture Network) viewer.
- **Advanced Analytics**:
  - Rate Transient Analysis (RTA) for flow regime diagnostics.
  - Sensitivity analysis and Monte Carlo simulation for uncertainty quantification.
  - Well placement optimization to find the best drilling locations.
- **Field Data Matching**: Upload historical production data (CSV) to compare simulation results against actual field performance.

## ⚙️ How to Run Locally

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/srpeteng73/Full-3D-Unconventional-Black-Oil-Reservoir-Simulator.git
    cd Full-3D-Unconventional-Black-Oil-Reservoir-Simulator
    ```

2.  **Install the required libraries:**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Run the Streamlit application:**
    ```bash
    streamlit run app.py
    ```

## Disclaimer

This simulator is an educational and demonstrational tool. While it is built on fundamental reservoir engineering principles, it should not be used for making real-world financial or operational decisions without validation against commercial-grade simulators.
