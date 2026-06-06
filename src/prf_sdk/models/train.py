import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline


class ModelTrainer:
    """
    Classe responsável por orquestrar a preparação, treinamento e persistência do modelo.
    """

    def __init__(self, pipeline: Pipeline):
        """
        Inicializa o treinador com um Pipeline do scikit-learn.
        """
        self.pipeline = pipeline
        self.model = None

    def train(self, X_train: pd.DataFrame, y_train: pd.Series):
        """
        Treina o modelo utilizando os dados de treino fornecidos.
        """
        # TODO: Implementar o ajuste do pipeline nos dados de treino
        pass

    def save_artifacts(self, output_path: str):
        """
        Salva o pipeline de modelo treinado em um arquivo .pkl.
        """
        # TODO: Implementar a serialização com joblib ou pickle
        pass
