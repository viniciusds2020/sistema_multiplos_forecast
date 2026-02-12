"""CSS customizado para dashboard estilo React moderno."""

MAIN_CSS = """
<style>
    /* Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

    /* Global */
    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
    }

    /* Main container */
    .main .block-container {
        padding: 1.5rem 2rem 2rem 2rem;
        max-width: 1400px;
    }

    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0f0f23 0%, #1a1a3e 50%, #16213e 100%);
    }

    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p,
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] span,
    [data-testid="stSidebar"] label {
        color: #e2e8f0 !important;
    }

    [data-testid="stSidebar"] .stSelectbox label,
    [data-testid="stSidebar"] .stMultiSelect label,
    [data-testid="stSidebar"] .stSlider label {
        color: #cbd5e1 !important;
        font-weight: 500;
    }

    /* Header styling */
    h1 {
        color: #1a1a2e !important;
        font-weight: 800 !important;
        letter-spacing: -0.5px;
    }

    h2 {
        color: #2b2d42 !important;
        font-weight: 700 !important;
    }

    h3 {
        color: #3a3d5c !important;
        font-weight: 600 !important;
    }

    /* KPI Card */
    .kpi-card {
        background: white;
        border-radius: 16px;
        padding: 1.5rem;
        box-shadow: 0 1px 3px rgba(0,0,0,0.06), 0 4px 12px rgba(0,0,0,0.04);
        border: 1px solid #f0f0f5;
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }

    .kpi-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 16px rgba(0,0,0,0.1);
    }

    .kpi-label {
        font-size: 0.8rem;
        font-weight: 600;
        color: #8d99ae;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin-bottom: 0.3rem;
    }

    .kpi-value {
        font-size: 2rem;
        font-weight: 800;
        color: #1a1a2e;
        line-height: 1.1;
    }

    .kpi-delta {
        font-size: 0.85rem;
        font-weight: 600;
        margin-top: 0.3rem;
    }

    .kpi-delta-positive { color: #06d6a0; }
    .kpi-delta-negative { color: #ef476f; }
    .kpi-delta-neutral { color: #8d99ae; }

    /* Chart container */
    .chart-container {
        background: white;
        border-radius: 16px;
        padding: 1.5rem;
        box-shadow: 0 1px 3px rgba(0,0,0,0.06), 0 4px 12px rgba(0,0,0,0.04);
        border: 1px solid #f0f0f5;
        margin-bottom: 1rem;
    }

    .chart-title {
        font-size: 1rem;
        font-weight: 700;
        color: #2b2d42;
        margin-bottom: 0.8rem;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid #f0f0f5;
    }

    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 0.5rem;
        background: #f8f9fc;
        border-radius: 12px;
        padding: 0.3rem;
    }

    .stTabs [data-baseweb="tab"] {
        border-radius: 10px;
        font-weight: 600;
        font-size: 0.9rem;
        padding: 0.5rem 1.2rem;
    }

    .stTabs [aria-selected="true"] {
        background: white !important;
        box-shadow: 0 2px 8px rgba(0,0,0,0.08);
    }

    /* Buttons */
    .stButton > button {
        background: linear-gradient(135deg, #4361ee 0%, #3a0ca3 100%);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 0.6rem 1.5rem;
        font-weight: 600;
        transition: all 0.3s ease;
        box-shadow: 0 2px 8px rgba(67, 97, 238, 0.3);
    }

    .stButton > button:hover {
        transform: translateY(-1px);
        box-shadow: 0 4px 16px rgba(67, 97, 238, 0.4);
    }

    /* Dataframe */
    .stDataFrame {
        border-radius: 12px;
        overflow: hidden;
    }

    /* Metric highlight */
    .metric-highlight {
        background: linear-gradient(135deg, #4361ee15 0%, #3a0ca315 100%);
        border-radius: 12px;
        padding: 1rem;
        border-left: 4px solid #4361ee;
    }

    /* Status badge */
    .status-badge {
        display: inline-block;
        padding: 0.25rem 0.75rem;
        border-radius: 20px;
        font-size: 0.75rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }

    .badge-success { background: #06d6a020; color: #06d6a0; }
    .badge-warning { background: #ffd16620; color: #e6a817; }
    .badge-danger { background: #ef476f20; color: #ef476f; }
    .badge-info { background: #118ab220; color: #118ab2; }

    /* Progress bar */
    .stProgress > div > div {
        background: linear-gradient(90deg, #4361ee, #06d6a0);
        border-radius: 10px;
    }

    /* Expander */
    .streamlit-expanderHeader {
        font-weight: 600;
        border-radius: 10px;
    }

    /* Separator */
    .section-divider {
        border: none;
        height: 2px;
        background: linear-gradient(90deg, #4361ee40, transparent);
        margin: 1.5rem 0;
    }

    /* Logo area */
    .logo-container {
        text-align: center;
        padding: 1rem 0 1.5rem 0;
    }

    .logo-title {
        font-size: 1.3rem;
        font-weight: 800;
        background: linear-gradient(135deg, #4361ee, #06d6a0);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }

    .logo-subtitle {
        font-size: 0.75rem;
        color: #8d99ae;
        margin-top: 0.2rem;
    }
</style>
"""


def inject_css():
    """Injeta CSS customizado na pagina Streamlit."""
    import streamlit as st
    st.markdown(MAIN_CSS, unsafe_allow_html=True)
