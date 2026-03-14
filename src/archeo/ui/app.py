import plotly.graph_objects as go
import streamlit as st

from archeo.constants.enum import Fits
from archeo.data_structures.distribution import Uniform
from archeo.data_structures.math import Domain
from archeo.data_structures.physics.binary import BinaryGenerator
from archeo.data_structures.physics.black_hole import BlackHoleGenerator
from archeo.postprocessing.dataframe import convert_simulated_binaries_to_dataframe
from archeo.simulation.simulate_merger import simulate_black_hole_mergers
from archeo.ui.visualization import add_pdf


st.set_page_config(page_title="Exploring Distribution of Remnant Properties", layout="wide")

###############################################
#                Sidebar Content              #
###############################################

st.sidebar.markdown("## Configure Your Ancestral Priors")

# Store slider values in session state
prior_name = st.sidebar.text_input("Name of the prior", "My prior")
spin_setting = st.sidebar.selectbox("Spin setting", ["Aligned spin", "Precession spin"])
n_samples = st.sidebar.slider("Number of samples", min_value=1000, max_value=10000, value=1000)
n_workers = st.sidebar.slider("Number of workers", min_value=1, max_value=16, value=1)
q_range = st.sidebar.slider("Mass ratio $q$", min_value=1.0, max_value=6.0, value=(1.0, 6.0))

st.sidebar.markdown("#### Primary Black Hole Settings")
m1_range = st.sidebar.slider("Mass $m_1$", min_value=5.0, max_value=200.0, value=(5.0, 200.0))
a1_range = st.sidebar.slider("Spin $\\chi_1$", min_value=0.0, max_value=0.99, value=(0.0, 0.99))

if spin_setting == "Precession spin":
    phi1_range = st.sidebar.slider("Azimuthal angle $\\phi_1$", min_value=0.0, max_value=2.0, value=(0.0, 2.0))
    theta1_range = st.sidebar.slider("Polar angle $\\theta_1$", min_value=0.0, max_value=1.0, value=(0.0, 1.0))
else:
    phi1_range = theta1_range = (0.0, 0.0)

st.sidebar.markdown("#### Secondary Black Hole Settings")
m2_range = st.sidebar.slider("Mass $m_2$", min_value=5.0, max_value=200.0, value=(5.0, 200.0))
a2_range = st.sidebar.slider("Spin $\\chi_2$", min_value=0.0, max_value=0.99, value=(0.0, 0.99))

if spin_setting == "Precession spin":
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
        "k_f": "Birth Recoil k<sub>f</sub> [km s<sup>-1</sup>]",
        "a_f": "Spin χ<sub>f</sub> [-]",
        "m_f": "Mass m<sub>f</sub> [M<sub>Sun</sub>]",
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

    fits = Fits.NRSUR7DQ4REMNANT if spin_setting == "Precession spin" else Fits.NRSUR3DQ8REMNANT
    generator_bh1 = BlackHoleGenerator(
        mass_distribution=Uniform(low=m1_range[0], high=m1_range[1]),
        spin_magnitude_distribution=Uniform(low=a1_range[0], high=a1_range[1]),
        phi_distribution=Uniform(low=phi1_range[0], high=phi1_range[1]),
        theta_distribution=Uniform(low=theta1_range[0], high=theta1_range[1]),
    )
    generator_bh2 = BlackHoleGenerator(
        mass_distribution=Uniform(low=m2_range[0], high=m2_range[1]),
        spin_magnitude_distribution=Uniform(low=a2_range[0], high=a2_range[1]),
        phi_distribution=Uniform(low=phi2_range[0], high=phi2_range[1]),
        theta_distribution=Uniform(low=theta2_range[0], high=theta2_range[1]),
    )
    binary_generator = BinaryGenerator(
        primary_black_hole_source=generator_bh1,
        secondary_black_hole_source=generator_bh2,
        mass_ratio_domain=Domain(low=q_range[0], high=q_range[1]),
        is_aligned_spin=(spin_setting == "Aligned spin"),
    )

    with st.spinner("Running simulation... Please wait..."):

        black_hole_mergers = simulate_black_hole_mergers(binary_generator, fits, n_samples, n_workers)
        df = convert_simulated_binaries_to_dataframe(black_hole_mergers)

        st.write("## Visualization of Remnant Properties")
        for col in ["k_f", "a_f", "m_f"]:
            add_pdf(st.session_state.figs[col], df[col], prior_name)
            st.plotly_chart(st.session_state.figs[col])

    st.success("Simulation completed!")
