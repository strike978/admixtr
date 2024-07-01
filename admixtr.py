

import streamlit as st


def nav_to(url):
    # Use the HTML meta refresh tag for automatic redirection
    st.markdown(
        f'<meta http-equiv="refresh" content="0;url={url}" />', unsafe_allow_html=True
    )


# Usage
nav_to("https://admixr.com")
