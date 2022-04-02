import streamlit as st

from src.data.NFPD_fish_data import retrieve_fish_counts
from src.app.app import App, AppConfig, AppModes


@st.cache(allow_output_mutation=True)
def _get_cached_data() -> AppConfig:
    return AppConfig(
        mode=AppModes.data_plotter,
        fish_count_data=retrieve_fish_counts(),
    )


def main():
    st.set_page_config(layout="wide")
    st.markdown("# anglersim")

    app_config = _get_cached_data()
    
    selected_mode = st.sidebar.selectbox(
        "Choose the app mode",
        [m.value for m in AppModes],
    )
    app_config.mode = AppModes(selected_mode)
    
    app = App(app_config)
    app.run()
    

if __name__=="__main__":
    main()