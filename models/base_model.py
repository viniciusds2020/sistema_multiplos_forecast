"""Interface abstrata para modelos de forecast."""

from abc import ABC, abstractmethod
import pandas as pd


class BaseForecaster(ABC):
    """Classe base para todos os modelos de forecasting."""

    name: str = "Base"
    supports_exogenous: bool = False
    supports_intermittent: bool = False

    @abstractmethod
    def fit(self, y: pd.Series, X: pd.DataFrame = None) -> "BaseForecaster":
        """Treina o modelo.

        Args:
            y: Serie temporal de demanda (index=date).
            X: Features exogenas opcionais.
        """
        ...

    @abstractmethod
    def predict(self, horizon: int, X_future: pd.DataFrame = None) -> pd.DataFrame:
        """Gera previsoes.

        Args:
            horizon: Numero de passos a frente.
            X_future: Features exogenas futuras.

        Returns:
            DataFrame com colunas: [ds, yhat, yhat_lower, yhat_upper]
        """
        ...

    def get_params(self) -> dict:
        """Retorna parametros do modelo."""
        return {}
