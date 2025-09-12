# 3D Unconventional Reservoir Simulator

This is an interactive Streamlit application for simulating a full 3D unconventional black-oil reservoir.

## Features

-   3D grid and property modeling (kx, ky, ϕ)
-   Black-oil PVT property calculations
-   Discrete Fracture Network (DFN) support via CSV upload or auto-generation
-   Rate Transient Analysis (RTA) for flow regime diagnostics
-   3D visualization of pressure and saturation
-   Sensitivity analysis and Monte Carlo simulation

## How to Deploy on Streamlit Cloud

1.  **Create a GitHub Repository:** Create a new, public repository on your GitHub account.
2.  **Upload Files:** Upload the four files provided (`app.py`, `requirements.txt`, `README.md`, and the `.streamlit/config.toml` file inside its folder) to this repository.
3.  **Deploy on Streamlit Cloud:**
    -   Go to [share.streamlit.io](https://share.streamlit.io).
    -   Click "New app" and connect it to your new GitHub repository.
    -   Make sure the "Main file path" is set to `app.py`.
    -   Click "Deploy!".
