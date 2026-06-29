import numpy as np
import plotly.graph_objs as go
import streamlit as st
from scipy.special import erfc

# Physikalische Konstanten
R_GAS = 8.31446261815324  # J/(mol*K)

# Geochemische Datenbank
DATABASE = {
    "Olivin": {
        "Fe-Mg": {"D0": 1e-12, "EA": 250.0},
        "Ca": {"D0": 5e-13, "EA": 220.0},
    },
    "Plagioklas": {
        "Sr": {"D0": 8e-13, "EA": 230.0},
        "Ba": {"D0": 7e-13, "EA": 225.0},
    },
    "Quarz": {
        "Ti": {"D0": 3e-13, "EA": 240.0},
    },
}

# App-Header
st.set_page_config(
    page_title="Diffusionsprofil-Modeling",
    page_icon="🧪",
    layout="wide",
)
st.title("High-Level Diffusionsprofil-Modellierung")
st.markdown(
    "Eine interaktive Streamlit-App zur Modellierung von Diffusionsprofilen in Mineralen "
    "mit praxisnahen geochemischen Parametern und analytischen Lösungen."
)

# Sidebar: Geometrie-Auswahl
st.sidebar.header("Modellparameter")
geometry = st.sidebar.selectbox(
    "Geometrie wählen",
    ["Halbunendlicher Raum", "Sphärisches Mineral/Kugel"],
)

# Sidebar: Mineralspezifische Daten und Elementauswahl
mineral = st.sidebar.selectbox("Mineral", list(DATABASE.keys()))
selected_element = st.sidebar.selectbox("Element", list(DATABASE[mineral].keys()))
parameter = DATABASE[mineral][selected_element]

# Sidebar: Temperatur und andere Parameter
st.sidebar.subheader("Umgebungsbedingungen")
T_celsius = st.sidebar.slider("Temperatur (°C)", min_value=600, max_value=1200, value=900, step=25)
T_kelvin = T_celsius + 273.15

st.sidebar.subheader("Diffusionsparameter")
t = st.sidebar.slider("Zeit (Jahre)", min_value=1.0, max_value=100.0, value=25.0, step=1.0)
Cs = st.sidebar.slider("Randkonzentration Cs", min_value=0.5, max_value=1.5, value=1.0, step=0.05)
C0 = st.sidebar.slider("Anfangskonzentration C0", min_value=0.0, max_value=0.5, value=0.1, step=0.01)

# Physikalische Parameter
radius_um = 100.0
x_max_um = 100.0
x = np.linspace(0, x_max_um, 301)
x_m = x * 1e-6

# Arrhenius-Berechnung
D0 = parameter["D0"]
EA = parameter["EA"] * 1000.0
D = D0 * np.exp(-EA / (R_GAS * T_kelvin))

# Charakteristische Diffusionszeit
if D > 0:
    tau_seconds = (x_max_um * 1e-6) ** 2 / D
else:
    tau_seconds = np.inf
tau_years = tau_seconds / (3600.0 * 24.0 * 365.25)

# Konzentrationsprofile
def profile_half_space(x_m, t_s, D, Cs, C0):
    argument = x_m / (2.0 * np.sqrt(D * t_s))
    return C0 + (Cs - C0) * erfc(argument)


def profile_sphere(x_um, t_s, D, Cs, C0, radius_um, n_terms=30):
    r = x_um / radius_um
    r = np.clip(r, 0, 1)
    profile = np.zeros_like(r)
    for n in range(1, n_terms + 1):
        lambda_n = np.pi * (2 * n - 1) / 2.0
        term = ((-1) ** (n + 1) / lambda_n) * np.sin(lambda_n * r)
        decay = np.exp(-lambda_n ** 2 * D * t_s / (radius_um * 1e-6) ** 2)
        profile += term * decay
    profile = C0 + 2.0 * (Cs - C0) * profile
    return profile

# Berechnung
t_seconds = t * 365.25 * 24.0 * 3600.0

if geometry == "Halbunendlicher Raum":
    concentrations = profile_half_space(x_m, t_seconds, D, Cs, C0)
else:
    concentrations = profile_sphere(x, t_seconds, D, Cs, C0, radius_um)

# KPI-Leiste
st.subheader("Kinetische KPI")
col1, col2, col3 = st.columns([1, 1, 1])
col1.metric("Diffusionskoeffizient D", f"{D:.3e} m²/s")
col2.metric("Charakteristische Zeit τ", f"{tau_years:.2f} Jahre")
col3.metric("Geometrie", geometry)

# Plot
fig = go.Figure()
fig.add_trace(
    go.Scatter(
        x=x,
        y=concentrations,
        mode="lines",
        line=dict(color="#1f77b4", width=3),
        name="Konzentration",
    )
)
fig.update_layout(
    title=f"Diffusionsprofil für {mineral} - {selected_element}",
    xaxis_title="Tiefe (µm)",
    yaxis_title="Konzentration",
    template="plotly_white",
    margin=dict(l=60, r=20, t=70, b=50),
)
fig.update_yaxes(range=[0.0, max(1.6, Cs * 1.1)])

st.plotly_chart(fig, use_container_width=True)

st.markdown(
    "---\n"
    "**Erläuterung:** Die Berechnung basiert auf analytischen Lösungen für den halbunendlichen Raum und eine kugelförmige Geometrie. "
    "Für die Kugel wird eine Fourier-Reihe mit 30 Termen verwendet, um ein qualitativ stabiles Profil zu erzeugen."
)
