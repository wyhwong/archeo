import os

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from archeo.constants import Columns as C
from archeo.frontend import viz


st.set_page_config(page_title="Exploring Distribution of Remnant Properties", layout="wide")

###############################################
#                Sidebar Content              #
###############################################

st.sidebar.markdown("## Configure Your Ancestor Priors")

# Store slider values in session state
prior_name = st.sidebar.text_input("Name of the prior", "My prior")
spin_setting = st.sidebar.selectbox("Spin setting", ["Precessing spin", "Aligned spin"])
q_range = st.sidebar.slider("Mass ratio $q$", min_value=1.0, max_value=6.0, value=(1.0, 6.0))

st.sidebar.markdown("#### Primary Black Hole Settings")
m1_range = st.sidebar.slider("Mass $m_1$", min_value=5.0, max_value=200.0, value=(5.0, 200.0))
a1_range = st.sidebar.slider("Spin $\\chi_1$", min_value=0.0, max_value=0.99, value=(0.0, 0.99))

st.sidebar.markdown("#### Secondary Black Hole Settings")
m2_range = st.sidebar.slider("Mass $m_2$", min_value=5.0, max_value=200.0, value=(5.0, 200.0))
a2_range = st.sidebar.slider("Spin $\\chi_2$", min_value=0.0, max_value=0.99, value=(0.0, 0.99))

###############################################
#                Main Content                 #
###############################################

st.markdown(
    """
    # Exploring Distribution of Remnant Properties

    ## Introduction

    This application allows users to configure the distributions for properties
    of black hole binaries and visualize the distribution of their remnant
    properties. By adjusting mass, spin, and alignment settings, users can generate
    samples and study the properties of remnant black holes.

    If you found hierarchical formation of black holes interesting and want to learn more,
    please refer to our paper:

    [1] Carlos Araújo Álvarez, Henry W. Y. Wong, Juan Calderón Bustillo. "Kicking Time
        Back in Black Hole Mergers: Ancestral Masses, Spins, Birth Recoils, and
        Hierarchical-formation Viability of GW190521." The Astrophysical Journal
        977.2 (2024): 220.
    """
)


if "figs" not in st.session_state:
    st.session_state["figs"] = {}
    for col, label in {
        C.BH_KICK: "Birth Recoil k<sub>f</sub> [km s<sup>-1</sup>]",
        C.BH_SPIN: "Spin χ<sub>f</sub> [-]",
        C.BH_MASS: "Mass m<sub>f</sub> [M<sub>Sun</sub>]",
    }.items():
        st.session_state.figs[col] = go.Figure()
        st.session_state.figs[col].update_layout(
            title="Probability Density Function",
            xaxis_title=label,
            yaxis_title="Density",
            showlegend=True,
            barmode="overlay",
        )

if st.sidebar.button("Run"):

    app_dir = os.path.dirname(__file__)
    prior = "ppq6" if spin_setting == "Precessing spin" else "apq6"
    df = pd.read_feather(f"{app_dir}/{prior}_st.ipc")

    with st.spinner("Loading... Please wait..."):
        _df = df.loc[
            (df[C.HEAVIER_BH_MASS] <= m1_range[1])
            & (df[C.HEAVIER_BH_MASS] >= m1_range[0])
            & (df[C.LIGHTER_BH_MASS] <= m2_range[1])
            & (df[C.LIGHTER_BH_MASS] >= m2_range[0])
            & (df[C.HEAVIER_BH_SPIN] <= a1_range[1])
            & (df[C.HEAVIER_BH_SPIN] >= a1_range[0])
            & (df[C.LIGHTER_BH_SPIN] <= a2_range[1])
            & (df[C.LIGHTER_BH_SPIN] >= a2_range[0])
            & (df[C.MASS_RATIO] <= q_range[1])
            & (df[C.MASS_RATIO] >= q_range[0])
        ]

        if _df.shape[0] > 10000:
            _df = _df.sample(10000, replace=True)
        elif _df.empty:
            st.error("No samples available. Please set a new config.")

        st.write("## Visualization of Remnant Properties")
        for col in [C.BH_KICK, C.BH_SPIN, C.BH_MASS]:
            viz.add_pdf(st.session_state.figs[col], _df[col], prior_name)
            st.plotly_chart(st.session_state.figs[col])
            if col == C.BH_KICK:
                st.markdown(
                    """
                    Birth Recoil $k_f$ is the velocity of the remnant black hole after the merger.
                    This property is important because it determines whether the remnant black hole
                    can remain within its environment and eventually undergo a subsequent merger that
                    may be observed by detectors on Earth. For example, the typical escape velocity of
                    a globular cluster is only 50 km/s (Baumgardt & Hilker, 2018).
                    """
                )

    st.success("Simulation completed!")
