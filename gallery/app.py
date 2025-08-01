import os

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from archeo.constants import Columns as C
from archeo.constants import Suffixes as S
from archeo.frontend import viz


st.set_page_config(page_title="Exploring Distribution of Remnant Properties", layout="wide")

###############################################
#                Sidebar Content              #
###############################################

st.sidebar.markdown("## Configure Your Ancestral Priors")

# Store slider values in session state
prior_name = st.sidebar.text_input("Name of the prior", "My prior")
spin_setting = st.sidebar.selectbox("Spin setting", ["Precessing spin", "Aligned spin"])
q_range = st.sidebar.slider("Mass ratio $q$", min_value=1.0, max_value=6.0, value=(1.0, 6.0))

st.sidebar.markdown("#### Primary Black Hole Settings")
m1_range = st.sidebar.slider(r"Mass $m_1$ [$M_\odot$]", min_value=5.0, max_value=200.0, value=(5.0, 200.0))
a1_range = st.sidebar.slider("Spin $\\chi_1$", min_value=0.0, max_value=0.99, value=(0.0, 0.99))
st.sidebar.markdown(
    """
    Spin $\\chi$ describes the rotation of the black hole,
    ranging from 0 (non-rotating) to 0.99.
    Spin influences the dynamics of mergers and gravitational wave signals, etc.
    """
)

st.sidebar.markdown("#### Secondary Black Hole Settings")
m2_range = st.sidebar.slider(r"Mass $m_2$ [$M_\odot$]", min_value=5.0, max_value=200.0, value=(5.0, 200.0))
a2_range = st.sidebar.slider("Spin $\\chi_2$", min_value=0.0, max_value=0.99, value=(0.0, 0.99))

###############################################
#                Main Content                 #
###############################################

st.markdown(
    """
    # Exploring Distribution of Remnant Properties

    ## Introduction

    This application allows users to explore the distributions for properties
    of binary black hole (BBH) mergers and visualize the characteristics of their remnant
    BHs. By adjusting parameters such as mass, spin, and alignment, users can generate
    samples to study how these factors influence the final mass, spin, and recoil velocity
    of the remnant BH. Understanding these properties is crucial for studying
    the hierarchical formation of BHs.
    """
)


if "figs" not in st.session_state:
    st.session_state["figs"] = {}
    for col, label in {
        S.FINAL(C.KICK): "Birth Recoil k<sub>f</sub> [km s<sup>-1</sup>]",
        S.FINAL(C.SPIN_MAG): "Spin χ<sub>f</sub> [-]",
        S.FINAL(C.MASS): "Mass m<sub>f</sub> [M<sub>Sun</sub>]",
    }.items():
        st.session_state.figs[col] = go.Figure()
        st.session_state.figs[col].update_layout(
            title="Probability Density Function",
            xaxis_title=label,
            yaxis_title="Density",
            showlegend=True,
            barmode="overlay",
        )


info: dict[str, str] = {
    S.FINAL(
        C.KICK
    ): """
    The Birth Recoil $k_f$ is the velocity of the remnant BH after the merger.
    This property is crucial as it determines whether the remnant remains gravitationally
    bound to its environment or is ejected, affecting its potential for future mergers
    that could be detected on Earth. For instance, the typical escape velocity of a
    globular cluster is only $50$ $km$ $s^{-1}$ (Baumgardt & Hilker, 2018), meaning that BHs
    with significantly higher recoil velocities are likely to be ejected from such environments
    instead of forming hierarchical mergers.
    """,
    S.FINAL(
        C.SPIN_MAG
    ): r"""
    The dimensionless spin of the remnant BH after the merger, denoted as $\chi_f$,
    reflects its rotational properties. Supposed in the visualization, we should see remnant
    BHs exhibit a preference for high spins (a peak around $0.7$ for precessing case).
    In contrast, first-generation BHs formed from isolated stellar binaries are expected
    to have masses $\lesssim 50$ $M_{\odot}$ and exhibit relatively low spins aligned to the orbital
    (Pierra, Mastrogiovanni & Perrières, 2024).
    """,
    S.FINAL(
        C.MASS
    ): r"""
    The remnant mass $m_f$ of the final BH is determined by the masses of the merging black
    holes and the energy radiated away. While we mentioned most BHs formed from stellar
    collapse are expected to have masses below $50$ $M_{\odot}$, BBH mergers can produce significantly
    more massive remnants. Notably, we should observe samples in the pair-instability supernova (PISN)
    mass gap $M \in [65, 130]$ $M_{\odot}$, where BHs cannot form directly from stellar evolution.
    This suggests that second-generation BHs, formed from previous mergers, provide a key pathway
    for populating this otherwise empty region of the BH mass spectrum.
    """,
}

if st.sidebar.button("Run"):

    app_dir = os.path.dirname(__file__)
    prior = "ppq6" if spin_setting == "Precessing spin" else "apq6"
    df = pd.read_feather(f"{app_dir}/{prior}_st.ipc")

    with st.spinner("Loading... Please wait..."):
        _df = df.loc[
            (df[S.PRIMARY(C.MASS)] <= m1_range[1])
            & (df[S.PRIMARY(C.MASS)] >= m1_range[0])
            & (df[S.SECONDARY(C.MASS)] <= m2_range[1])
            & (df[S.SECONDARY(C.MASS)] >= m2_range[0])
            & (df[S.PRIMARY(C.SPIN_MAG)] <= a1_range[1])
            & (df[S.PRIMARY(C.SPIN_MAG)] >= a1_range[0])
            & (df[S.SECONDARY(C.SPIN_MAG)] <= a2_range[1])
            & (df[S.SECONDARY(C.SPIN_MAG)] >= a2_range[0])
            & (df[C.MASS_RATIO] <= q_range[1])
            & (df[C.MASS_RATIO] >= q_range[0])
        ]

        if _df.shape[0] > 10000:
            _df = _df.sample(10000, replace=True)
        elif _df.empty:
            st.error("No samples available. Please set a new config.")

        st.write("## Visualization of Remnant Properties")
        for col in [S.FINAL(C.KICK), S.FINAL(C.SPIN_MAG), S.FINAL(C.MASS)]:
            viz.add_pdf(st.session_state.figs[col], _df[col], prior_name)
            st.plotly_chart(st.session_state.figs[col])
            if col in info:
                st.markdown(info[col])

        st.markdown(
            """
            ## More Information

            If you found hierarchical formation of BHs interesting and want to learn more,
            please refer to our paper:

            [1] Carlos Araújo Álvarez, Henry W. Y. Wong, Juan Calderón Bustillo. "Kicking Time
                Back in Black Hole Mergers: Ancestral Masses, Spins, Birth Recoils, and
                Hierarchical-formation Viability of GW190521." The Astrophysical Journal
                977.2 (2024): 220.

            Also, please let us know if you found this application has any wrong information.
            """
        )

    st.success("Simulation completed!")
