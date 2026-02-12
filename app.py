"""Dashboard Streamlit - Sistema de Forecast Multi-Produto SKU."""

import sys
from pathlib import Path

# Adicionar raiz ao path
sys.path.insert(0, str(Path(__file__).parent))

import streamlit as st
from dashboard.styles import inject_css
from dashboard.components import sidebar_logo
from data.synthetic_generator import generate_synthetic_data

# Configuracao da pagina
st.set_page_config(
    page_title="Forecast Pro - Multi-Produto SKU",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Injetar CSS customizado
inject_css()


@st.cache_data(ttl=3600)
def load_data():
    """Gera e cacheia dados sinteticos."""
    return generate_synthetic_data()


def main():
    # Sidebar
    with st.sidebar:
        sidebar_logo()
        st.markdown("---")

        page = st.radio(
            "Navegacao",
            [
                "Overview",
                "Data Explorer",
                "Similaridade",
                "Forecasting",
                "Comparacao de Modelos",
            ],
            index=0,
        )

        st.markdown("---")

        # Opcoes de dados
        with st.expander("Configuracoes de Dados"):
            if st.button("Regenerar Dados", use_container_width=True):
                st.cache_data.clear()
                st.rerun()

            st.markdown(
                f"<div style='color:#8d99ae; font-size:0.8rem; margin-top:0.5rem;'>"
                f"Dados sinteticos com 20 SKUs e 2.5 anos de historico"
                f"</div>",
                unsafe_allow_html=True,
            )

        st.markdown("---")
        st.markdown(
            "<div style='text-align:center; color:#4a5568; font-size:0.7rem; padding:1rem 0;'>"
            "Forecast Pro v1.0<br>"
            "Sistema de Forecast Multi-Produto"
            "</div>",
            unsafe_allow_html=True,
        )

    # Carregar dados
    df = load_data()

    # Renderizar pagina selecionada
    if page == "Overview":
        from dashboard.page_overview import render
        render(df)

    elif page == "Data Explorer":
        from dashboard.page_data_explorer import render
        render(df)

    elif page == "Similaridade":
        from dashboard.page_similarity import render
        render(df)

    elif page == "Forecasting":
        from dashboard.page_forecasting import render
        render(df)

    elif page == "Comparacao de Modelos":
        from dashboard.page_model_comparison import render
        render(df)


if __name__ == "__main__":
    main()
