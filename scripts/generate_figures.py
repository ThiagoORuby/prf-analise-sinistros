import pandas as pd

from prf_sdk.settings import settings
from prf_sdk.utils.plots import (
    build_fatality_map_figure,
    build_h1b_fatality_figure,
    build_lorenz_spatial_figure,
    build_regional_h5_figure,
    build_stl_seasonality_figure,
)


DATA_PATH = settings.BASE_DIR / "data/processed/datatran_2022_2026_processed_v1.csv"
FIGURES_DIR = settings.BASE_DIR / "docs/figures"


def _run(label: str, fn, *args, **kwargs):
    print(f"  {label}...", end=" ", flush=True)
    try:
        result = fn(*args, **kwargs)
        print(f"OK → {result}")
    except Exception as exc:
        print(f"ERRO: {exc}")
        raise


def main():
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Carregando dados de {DATA_PATH} ...")
    df = pd.read_csv(DATA_PATH, low_memory=False)
    print(f"  {len(df):,} registros carregados.\n")

    print("=== Figuras do artigo ===")
    _run(
        "fig1.jpg  (H1b taxa por periodo)",
        build_h1b_fatality_figure,
        df,
        FIGURES_DIR / "fig1.jpg",
    )
    _run(
        "fig2.jpg  (STL sazonalidade)",
        build_stl_seasonality_figure,
        df,
        FIGURES_DIR / "fig2.jpg",
    )
    _run(
        "fig3.jpg  (Lorenz espacial)",
        build_lorenz_spatial_figure,
        df,
        FIGURES_DIR / "fig3.jpg",
    )
    _run(
        "fig4.jpg  (Regional H5)",
        build_regional_h5_figure,
        df,
        FIGURES_DIR / "fig4.jpg",
    )
    _run(
        "fig_mapa.png  (mapa de calor sinistros fatais)",
        build_fatality_map_figure,
        df,
        FIGURES_DIR / "fig_mapa.png",
    )

    print("\n=== Figuras de análise H2 ===")
    from prf_sdk.analysis.hypotheses.h2 import run_h2_analysis

    _run(
        "h2_periodos_fatalidade.png + h2_stl_decomposicao.png",
        run_h2_analysis,
        data_path=DATA_PATH,
        output_dir=FIGURES_DIR,
    )

    print("\nTodas as figuras geradas com sucesso.")


if __name__ == "__main__":
    main()
