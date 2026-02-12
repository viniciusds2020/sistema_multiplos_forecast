"""Configuracoes globais do sistema de forecast multi-produto."""

from pathlib import Path
import datetime

# Paths
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
ASSETS_DIR = BASE_DIR / "assets"

# Dados sinteticos
NUM_SKUS = 20
DATE_START = datetime.date(2023, 7, 1)
DATE_END = datetime.date(2025, 12, 31)
FORECAST_HORIZON = 30  # dias
TRAIN_TEST_SPLIT_DAYS = 60  # ultimos 60 dias para teste

# SKU Nomes e categorias
SKU_CATEGORIES = {
    "estavel": ["Arroz_5kg", "Feijao_1kg", "Acucar_1kg", "Oleo_Soja_900ml"],
    "sazonal": ["Protetor_Solar", "Sorvete_2L", "Chocolate_Pascoa", "Panetone_500g"],
    "tendencia_alta": ["Whey_Protein", "Leite_Vegetal_1L", "Granola_Organica", "Cafe_Especial"],
    "tendencia_baixa": ["DVD_Virgem", "Filme_Fotografico", "Fax_Toner", "CD_Audio"],
    "intermitente": ["Peca_Reposicao_A", "Equipamento_Industrial", "Reagente_Lab", "Filtro_Especial"],
}

# Safra (calendarios de colheita)
SAFRA_CALENDARS = {
    "soja": {"plantio": (9, 11), "crescimento": (12, 1), "colheita": (2, 4), "entressafra": (5, 8)},
    "milho_safrinha": {"plantio": (1, 3), "crescimento": (4, 5), "colheita": (6, 8), "entressafra": (9, 12)},
    "cana": {"plantio": (1, 3), "crescimento": (4, 6), "colheita": (4, 11), "entressafra": (12, 12)},
}

# Modelos
AVAILABLE_MODELS = [
    "XGBoost",
    "LightGBM",
    "AutoARIMA",
    "Prophet",
    "Chronos",
    "CrostonSBA",
]

# Metricas
METRICS = ["MAE", "RMSE", "MAPE", "WAPE", "Bias"]

# Paleta de cores do dashboard
COLORS = {
    "primary": "#4361ee",
    "secondary": "#3a0ca3",
    "success": "#06d6a0",
    "danger": "#ef476f",
    "warning": "#ffd166",
    "info": "#118ab2",
    "dark": "#1a1a2e",
    "dark_secondary": "#16213e",
    "light": "#f8f9fa",
    "white": "#ffffff",
    "text": "#2b2d42",
    "text_muted": "#8d99ae",
}

MODEL_COLORS = {
    "XGBoost": "#4361ee",
    "LightGBM": "#06d6a0",
    "AutoARIMA": "#ef476f",
    "Prophet": "#ffd166",
    "Chronos": "#118ab2",
    "CrostonSBA": "#7209b7",
    "Agregado": "#3a0ca3",
}

# Clustering
MAX_CLUSTERS = 8
MIN_CLUSTERS = 2
DTW_WINDOW_RATIO = 0.1

RANDOM_SEED = 42
