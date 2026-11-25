"""Treina e compara modelos no California Housing; gera modelo e evidências."""
from pathlib import Path
import json
import warnings
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
MODELS = ROOT / "models"
REPORTS = ROOT / "reports"
PLOTS = REPORTS / "graficos"
for p in (DATA, MODELS, REPORTS, PLOTS):
    p.mkdir(parents=True, exist_ok=True)
SEED = 42
TARGET = "MedHouseVal"

def main():
    print("Baixando/abrindo California Housing (dataset real público)...")
    ds = fetch_california_housing(as_frame=True, download_if_missing=True)
    df = ds.frame.copy()
    df.to_csv(DATA / "california_housing.csv", index=False)
    X, y = df.drop(columns=TARGET), df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=SEED
    )
    pipelines = {
        "Ridge": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", Ridge(alpha=1.0)),
        ]),
        "RandomForestRegressor": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", RandomForestRegressor(
                n_estimators=150, min_samples_leaf=2, random_state=SEED, n_jobs=-1
            )),
        ]),
    }
    metrics, fitted = {}, {}
    for name, pipe in pipelines.items():
        print(f"Treinando {name}...")
        pipe.fit(X_train, y_train)
        pred = pipe.predict(X_test)
        metrics[name] = {
            "MAE": float(mean_absolute_error(y_test, pred)),
            "RMSE": float(mean_squared_error(y_test, pred) ** 0.5),
            "R2": float(r2_score(y_test, pred)),
        }
        fitted[name] = pipe
    chosen = min(metrics, key=lambda name: metrics[name]["MAE"])
    joblib.dump(fitted[chosen], MODELS / "model.joblib")
    joblib.dump(fitted, MODELS / "modelos_comparados.joblib")
    pd.DataFrame(metrics).T.to_csv(REPORTS / "comparacao_modelos.csv", index_label="Modelo")
    payload = {
        "dataset": "California Housing (scikit-learn)",
        "target": TARGET,
        "target_unit": "US$ 100,000",
        "test_size": 0.2,
        "random_state": SEED,
        "rows": int(len(df)),
        "models": metrics,
        "selected_model": chosen,
        "selected_metrics": metrics[chosen],
        "note": "Resultados calculados pelo script no conjunto de teste; não são valores pré-preenchidos."
    }
    (REPORTS / "metricas.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    df.describe().to_csv(REPORTS / "estatisticas_descritivas.csv")
    corr = df.corr(numeric_only=True)[TARGET].drop(TARGET).sort_values(ascending=False)
    findings = [
        f"A base possui {len(df):,} registros, {X.shape[1]} atributos e o alvo {TARGET}.",
        f"Valores ausentes: {int(df.isna().sum().sum())}; linhas duplicadas: {int(df.duplicated().sum())}.",
        f"Maior correlação positiva linear com o alvo: {corr.index[0]} (r={corr.iloc[0]:.3f}).",
        f"Maior correlação negativa linear com o alvo: {corr.index[-1]} (r={corr.iloc[-1]:.3f}).",
        "Correlação descreve associação linear e não comprova causalidade."
    ]
    (REPORTS / "achados_eda.md").write_text(
        "# Achados da análise exploratória\n\n" + "\n".join(f"- {x}" for x in findings) + "\n",
        encoding="utf-8"
    )
    plt.figure(figsize=(8, 5))
    df[TARGET].hist(bins=40)
    plt.title("Distribuição do valor mediano de imóveis")
    plt.xlabel("Valor mediano (unidades de US$ 100 mil)")
    plt.ylabel("Frequência")
    plt.tight_layout()
    plt.savefig(PLOTS / "distribuicao_alvo.png", dpi=160)
    plt.close()
    plt.figure(figsize=(8, 5))
    plt.scatter(df["MedInc"], df[TARGET], alpha=.15, s=8)
    plt.title("Renda mediana vs. valor mediano")
    plt.xlabel("Renda mediana (unidade original do dataset)")
    plt.ylabel("Valor mediano (US$ 100 mil)")
    plt.tight_layout()
    plt.savefig(PLOTS / "renda_vs_valor.png", dpi=160)
    plt.close()
    plt.figure(figsize=(9, 6))
    c = df.corr(numeric_only=True)
    plt.imshow(c, aspect="auto", vmin=-1, vmax=1)
    plt.colorbar(label="Correlação")
    plt.xticks(range(len(c.columns)), c.columns, rotation=45, ha="right")
    plt.yticks(range(len(c.columns)), c.columns)
    plt.title("Matriz de correlação")
    plt.tight_layout()
    plt.savefig(PLOTS / "matriz_correlacao.png", dpi=160)
    plt.close()
    print(f"Modelo escolhido pelo menor MAE: {chosen}")
    print(json.dumps(metrics, indent=2))
    print("Modelo, métricas, estatísticas e gráficos salvos.")

if __name__ == "__main__":
    main()
