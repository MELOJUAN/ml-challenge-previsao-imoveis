"""Interface Streamlit para estimativa educacional de valor mediano por distrito."""
from pathlib import Path
import json
import joblib
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "models" / "model.joblib"
METRICS_PATH = ROOT / "reports" / "metricas.json"

st.set_page_config(page_title="Estimador de Imóveis", page_icon="🏠", layout="centered")
st.title("🏠 Previsão de Preços de Imóveis")
st.write("Informe características de uma região residencial para estimar o valor mediano.")
st.warning("Protótipo educacional baseado em dados históricos da Califórnia (EUA), não em imóveis brasileiros. Não é avaliação imobiliária.")

if not MODEL_PATH.exists():
    st.error("O modelo ainda não foi gerado. No GitHub, confira a execução do workflow 'Treinar modelo e gerar relatórios'.")
    st.info("Se estiver executando localmente: `python train.py`")
    st.stop()

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

model = load_model()
with st.form("inputs"):
    st.subheader("Características da região (California Housing)")
    a, b = st.columns(2)
    with a:
        medinc = st.number_input("Renda mediana (MedInc)", 0.0, 20.0, 3.5, 0.1)
        houseage = st.number_input("Idade mediana das casas (anos)", 0.0, 100.0, 25.0, 1.0)
        rooms = st.number_input("Média de cômodos por domicílio", 0.1, 100.0, 5.0, 0.1)
        bedrooms = st.number_input("Média de quartos por domicílio", 0.0, 20.0, 1.0, 0.1)
    with b:
        population = st.number_input("População do distrito", 0.0, 100000.0, 1000.0, 50.0)
        occupancy = st.number_input("Média de moradores por domicílio", 0.1, 100.0, 3.0, 0.1)
        latitude = st.number_input("Latitude", 32.0, 42.5, 37.0, 0.1)
        longitude = st.number_input("Longitude", -125.0, -114.0, -120.0, 0.1)
    submit = st.form_submit_button("Fazer previsão", type="primary", use_container_width=True)

if submit:
    row = pd.DataFrame([{
        "MedInc": medinc, "HouseAge": houseage, "AveRooms": rooms,
        "AveBedrms": bedrooms, "Population": population, "AveOccup": occupancy,
        "Latitude": latitude, "Longitude": longitude
    }])
    if rooms <= 0 or occupancy <= 0:
        st.error("Cômodos e ocupação devem ser positivos.")
    else:
        # MedHouseVal está em unidades de US$ 100.000.
        estimate = float(model.predict(row)[0]) * 100_000
        st.success(f"Valor mediano estimado: **US$ {estimate:,.0f}**")
        st.caption("Estimativa agregada por distrito censitário, não de uma casa específica.")

if METRICS_PATH.exists():
    info = json.loads(METRICS_PATH.read_text(encoding="utf-8"))
    st.divider()
    st.caption(f"Modelo: {info.get('selected_model', '—')}")
    m = info.get("selected_metrics", {})
    st.caption(f"MAE de teste: {m.get('MAE', '—')} | RMSE: {m.get('RMSE', '—')} | R²: {m.get('R2', '—')}")
