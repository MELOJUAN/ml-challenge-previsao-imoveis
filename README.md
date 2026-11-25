# ML Challenge — Previsão de Preços de Imóveis

**Trilha A — Regressão** | Projeto individual de Machine Learning.

> Este projeto é educacional. Utiliza o California Housing, uma base histórica dos EUA (censo de 1990), para prever o valor mediano por distrito censitário. Não estima o preço de um imóvel individual e não representa o mercado brasileiro.

## Entrega e automação

Este repositório inclui código de treinamento, aplicação Streamlit, documentação, testes básicos e workflow de GitHub Actions. Ao enviar o projeto para o GitHub, o workflow **Treinar modelo e gerar relatórios** baixa a base, treina os modelos e gera o modelo salvo, métricas e gráficos. A ação tenta commitar os artefatos gerados de volta ao repositório.

1. Crie um repositório público no GitHub.
2. Envie os arquivos desta pasta.
3. Abra a aba **Actions** e acompanhe o workflow.
4. Quando terminar, confira `models/` e `reports/` para os resultados calculados.

Se o workflow não iniciar, habilite Actions no repositório e confira se a permissão de workflow para escrita está disponível. Os resultados só aparecem após a execução bem-sucedida.

## Problema e variável-alvo

Estimar `MedHouseVal`, o valor mediano de imóveis por distrito, usando oito atributos demográficos/habitacionais. O alvo é expresso em unidades de **US$ 100.000**.

## Dataset

- Nome: California Housing
- Distribuído por: scikit-learn
- Documentação: https://scikit-learn.org/stable/datasets/real_world.html#california-housing-dataset
- Fonte original: California census data, 1990
- O script salva a base baixada em `data/california_housing.csv`.
- Consulte a documentação da fonte para detalhes de uso e licença.

## Modelos e metodologia

- Divisão treino/teste: 80%/20%, `random_state=42`.
- Pré-processamento dentro de Pipeline: imputação por mediana; Ridge também usa StandardScaler.
- Modelos comparados: Ridge e RandomForestRegressor.
- Métricas: MAE, RMSE e R².
- Seleção: menor MAE no conjunto de teste (para demonstração acadêmica; em projeto mais robusto, use validação cruzada no treino e mantenha o teste apenas para avaliação final).

## Estrutura

```text
.
├── app.py
├── train.py
├── requirements.txt
├── README.md
├── .gitignore
├── data/                 # CSV real baixado pelo treino
├── models/               # model.joblib e modelos_comparados.joblib
├── reports/              # métricas, achados, estatísticas e gráficos
├── screenshots/           # captura da interface (adicionar após abrir o app)
├── tests/
└── .github/workflows/
    └── train-and-commit.yml
```

## Execução local (se tiver computador)

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
python train.py
streamlit run app.py
```

A primeira execução precisa de internet para obter a base. O app carrega o arquivo `models/model.joblib` gerado pelo treino.

## Resultados reais

As métricas são geradas em `reports/metricas.json` e `reports/comparacao_modelos.csv`. Não há valores fictícios pré-preenchidos: confira os arquivos após o workflow concluir.

## Análise exploratória

O treino gera:
- estatísticas descritivas;
- pelo menos três achados calculados na base;
- distribuição do alvo;
- gráfico renda mediana × valor mediano;
- matriz de correlação.

Interprete os gráficos e os achados no relatório da entrega. Correlação não implica causalidade.

## Limitações

1. Dados antigos (1990) e geografia dos EUA.
2. Alvo agregado por distrito, não preço individual.
3. A base pode refletir desigualdades e vieses históricos.
4. Não usar para decisões financeiras ou avaliação real.
5. A aplicação é uma demonstração acadêmica.

## Uso de IA

Declare de forma transparente as ferramentas de IA utilizadas para apoiar estruturação, revisão ou programação, conforme as regras da disciplina. Revise o código, entenda as decisões e esteja preparado para explicar a metodologia.

## Checklist de entrega

- [ ] Workflow de treinamento concluído com sucesso.
- [ ] `models/model.joblib` foi gerado.
- [ ] `reports/metricas.json` contém métricas calculadas.
- [ ] Há 3 gráficos em `reports/graficos/`.
- [ ] README revisado e nome/identificação acadêmica preenchidos, se solicitado.
- [ ] Captura da interface adicionada em `screenshots/`.
- [ ] Repositório público e link conferido.
