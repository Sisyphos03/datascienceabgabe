import streamlit as st

try:
    import numpy as np
except ModuleNotFoundError:
    st.error("NumPy ist nicht verfügbar. Bitte installiere die Abhängigkeiten zuerst.")
    st.stop()

try:
    import plotly.graph_objects as go
    PLOTLY_AVAILABLE = True
except ModuleNotFoundError:
    go = None
    PLOTLY_AVAILABLE = False

try:
    from scipy.special import erfc
except ModuleNotFoundError:
    import math

    def erfc(x):
        values = np.asarray(x, dtype=float)
        if np.isscalar(x):
            return math.erfc(float(values))
        return np.vectorize(math.erfc, otypes=[float])(values)

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

preset_options = {
    "0.001 Mio. Jahre": 0.001,
    "0.01 Mio. Jahre": 0.01,
    "0.1 Mio. Jahre": 0.1,
    "1 Mio. Jahre": 1.0,
    "10 Mio. Jahre": 10.0,
    "100 Mio. Jahre": 100.0,
}

selected_preset_label = st.sidebar.selectbox(
    "Zeitskala wählen",
    list(preset_options.keys()),
    index=3,
)

preset_value = preset_options[selected_preset_label]

t_myr = st.sidebar.slider(
    "Zeit fein anpassen (Mio. Jahre)",
    min_value=0.001,
    max_value=100.0,
    value=preset_value,
    step=0.001,
    format="%.3f",
)

t = t_myr * 1_000_000

st.sidebar.caption(f"Aktuell gewählt: {t_myr:.3g} Mio. Jahre")

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
col2.metric("Charakteristische Zeit τ", f"{tau_years / 1_000_000:.2f} Mio. Jahre")
col3.metric("Geometrie", geometry)

# Plot
if PLOTLY_AVAILABLE:
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
        xaxis_title="Tiefe von der Oberfläche (µm)",
        yaxis_title="Konzentration (relativ)",
        template="plotly_white",
        margin=dict(l=60, r=20, t=70, b=50),
    )
    fig.update_xaxes(
        title=dict(text="Tiefe von der Oberfläche (µm)", standoff=10),
        automargin=True,
    )
    fig.update_yaxes(
        title=dict(text="Konzentration (relativ)", standoff=10),
        range=[0.0, max(1.6, Cs * 1.1)],
        automargin=True,
    )
    st.plotly_chart(fig, use_container_width=True)
else:
    import pandas as pd

    chart_data = pd.DataFrame(
        {"Tiefe von der Oberfläche (µm)": x, "Konzentration (relativ)": concentrations}
    )
    st.line_chart(
        chart_data,
        x="Tiefe von der Oberfläche (µm)",
        y="Konzentration (relativ)",
    )

st.markdown(
    "---\n"
    "**Erläuterung:** Die Berechnung basiert auf analytischen Lösungen für den halbunendlichen Raum und eine kugelförmige Geometrie. "
    "Für die Kugel wird eine Fourier-Reihe mit 30 Termen verwendet, um ein qualitativ stabiles Profil zu erzeugen."
)
