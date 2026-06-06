import pandas as pd


class ModelPredictor:
    """
    Classe responsável por carregar o modelo treinado e realizar predições de novos dados.
    """

    def __init__(self, model_path: str):
        """
        Carrega o modelo serializado do caminho fornecido.
        """
        # TODO: Implementar o carregamento do arquivo serializado
        self.model = None

    def predict(self, X: pd.DataFrame):
        """
        Realiza as predições de probabilidade ou classe sobre novas ocorrências.
        """
        # TODO: Implementar a inferência
        pass
