import matplotlib.pyplot as plt
import seaborn as sns


def set_plotting_theme():
    """
    Aplica um padrão visual unificado para todos os plots do projeto.
    """
    sns.set_theme(style="darkgrid")
    plt.rcParams["figure.figsize"] = (10, 6)
    plt.rcParams["axes.titlesize"] = 14
    plt.rcParams["axes.labelsize"] = 12
