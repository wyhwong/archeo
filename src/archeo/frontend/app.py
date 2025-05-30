import plotly.graph_objects as go
import streamlit as st

import archeo
from archeo.constants import Columns as C
from archeo.constants import Fits
from archeo.constants import Suffixes as S
from archeo.frontend import viz
from archeo.schema import Domain, PriorConfig


st.set_page_config(page_title="Exploring Distribution of Remnant Properties", layout="wide")

###############################################
#                Sidebar Content              #
###############################################

st.sidebar.markdown("## Configure Your Ancestral Priors")

# Store slider values in session state
prior_name = st.sidebar.text_input("Name of the prior", "My prior")
spin_setting = st.sidebar.selectbox("Spin setting", ["Aligned spin", "Positively aligned spin", "Precessing spin"])
n_samples = st.sidebar.slider("Number of samples", min_value=1000, max_value=10000, value=1000)
q_range = st.sidebar.slider("Mass ratio $q$", min_value=1.0, max_value=6.0, value=(1.0, 6.0))
is_uniform_q = st.sidebar.checkbox("Is uniform in mass ratio", value=False)
st.sidebar.markdown(
    "If this box is not checked, we sample the masses uniformly in the primary and secondary black hole masses."
)

st.sidebar.markdown("#### Primary Black Hole Settings")
m1_range = st.sidebar.slider("Mass $m_1$", min_value=5.0, max_value=200.0, value=(5.0, 200.0))
a1_range = st.sidebar.slider("Spin $\\chi_1$", min_value=0.0, max_value=0.99, value=(0.0, 0.99))

if spin_setting == "Precessing spin":
    phi1_range = st.sidebar.slider("Azimuthal angle $\\phi_1$", min_value=0.0, max_value=2.0, value=(0.0, 2.0))
    theta1_range = st.sidebar.slider("Polar angle $\\theta_1$", min_value=0.0, max_value=1.0, value=(0.0, 1.0))
else:
    phi1_range = theta1_range = (0.0, 0.0)

st.sidebar.markdown("#### Secondary Black Hole Settings")
m2_range = st.sidebar.slider("Mass $m_2$", min_value=5.0, max_value=200.0, value=(5.0, 200.0))
a2_range = st.sidebar.slider("Spin $\\chi_2$", min_value=0.0, max_value=0.99, value=(0.0, 0.99))

if spin_setting == "Precessing spin":
    phi2_range = st.sidebar.slider("Azimuthal angle $\\phi_2$", min_value=0.0, max_value=2.0, value=(0.0, 2.0))
    theta2_range = st.sidebar.slider("Polar angle $\\theta_2$", min_value=0.0, max_value=1.0, value=(0.0, 1.0))
else:
    phi2_range = theta2_range = (0.0, 0.0)

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
        C.KICK: "Birth Recoil k<sub>f</sub> [km s<sup>-1</sup>]",
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

if st.sidebar.button("Run"):

    fits = Fits.NRSUR7DQ4REMNANT if spin_setting == "Precessing spin" else Fits.NRSUR3DQ8REMNANT
    prior_config = PriorConfig(
        n_samples=n_samples,
        fits=fits,
        is_spin_aligned=spin_setting in ["Aligned spin", "Positively aligned spin"],
        is_only_up_aligned_spin=spin_setting == "Positively aligned spin",
        m_1=Domain(*m1_range),
        m_2=Domain(*m2_range),
        a_1=Domain(*a1_range),
        a_2=Domain(*a2_range),
        mass_ratio=Domain(*q_range),
        is_uniform_in_mass_ratio=is_uniform_q,
        is_mahapatra=False,
        phi_1=Domain(*phi1_range),
        phi_2=Domain(*phi2_range),
        theta_1=Domain(*theta1_range),
        theta_2=Domain(*theta2_range),
    )

    with st.spinner("Running simulation... Please wait..."):
        df = archeo.Prior.from_config(prior_config)

        st.write("## Visualization of Remnant Properties")
        for col in [C.KICK, S.FINAL(C.SPIN_MAG), S.FINAL(C.MASS)]:
            viz.add_pdf(st.session_state.figs[col], df[col], prior_name)
            st.plotly_chart(st.session_state.figs[col])

    st.success("Simulation completed!")
