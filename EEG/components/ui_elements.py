# components/ui_elements.py

import streamlit as st

class UIElements:
    """
    Class to handle UI elements like logos and other visual components.
    """

    @staticmethod
    def display_usach_logo():
        """
        Displays the USACH logo.
        """
        png_icon = "https://c7.alamy.com/comp/2R8DFAF/brain-waves-pulse-in-human-head-scan-one-line-vector-illustration-2R8DFAF.jpg"
        st.logo(png_icon)
        st.markdown("""
        <style>
            [alt=Logo] {
                height: 5rem;
            }
        </style>
        """, unsafe_allow_html=True)
