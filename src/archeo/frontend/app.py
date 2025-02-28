import streamlit as st

import archeo
from archeo.constants import Columns as C
from archeo.constants import Fits
from archeo.frontend import viz
from archeo.schema import Domain, PriorConfig


st.sidebar.markdown("## Configure Your Ancestor Priors")

# Store slider values in session state
spin_setting = st.sidebar.selectbox("Spin setting", ["Aligned spin", "Positively aligned spin", "Precessing spin"])
n_samples = st.sidebar.slider("Number of samples", min_value=1000, max_value=100000, value=5000)
q_range = st.sidebar.slider("Mass ratio $q$", min_value=1.0, max_value=6.0, value=(1.0, 6.0))
is_uniform_q = st.sidebar.checkbox("Is uniform in mass ratio", value=False)

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


# Create the prior config object
@st.cache_data(max_entries=5)
def get_prior_config():
    fits = Fits.NRSUR7DQ4REMNANT if spin_setting == "Precessing spin" else Fits.NRSUR3DQ8REMNANT
    return PriorConfig(
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


if st.sidebar.button("Run"):

    with st.spinner("Running simulation... Please wait..."):
        df = archeo.Prior.from_config(get_prior_config())

        st.write("## Visualization of Remnant Properties")
        st.plotly_chart(viz.plot_pdf(df, C.BH_KICK, label="Birth Recoil"))
        st.plotly_chart(viz.plot_pdf(df, C.BH_SPIN, label="Spin"))
        st.plotly_chart(viz.plot_pdf(df, C.BH_MASS, label="Mass"))

    st.success("Simulation completed!")
