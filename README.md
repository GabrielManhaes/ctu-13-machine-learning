# Projeto Final de Engenharia de Computação - PUC-Rio

## CTU-13

Dataset[1] criado pela Czech Technical University, misturando amostras reais de tráfego de botnets (Neris, Rbot, Virut, etc.) com tráfego não-malicioso e background.

## Conteúdo do Repositório

### `models/`

Nesta pasta, estão presentes os modelos implementados:
* Autoencoder
* Variational Autoencoder
* Stacked Autoencoders

### `preprocessing.py`

Neste arquivo, estão presentes as etapas de concatenação, limpeza de dados, encoding, feature engineering e separação em treino e teste.

### `train.pkl` e `test.pkl`

Pickled dataframe das duas saídas do arquivo `preprocessing.py`. Originalmente, o dataset CTU-13 ocupa em torno de 2.54 GB, enquanto os dois arquivos combinados pesam apenas 27.6 MB.

### `prediction.ipynb`

Neste  arquivo, estão presentes as etapas de scaling, treino e predição, gerando como saída Precision, Recall, F1-score, AUC e Confusion Matrix para cada modelo.

## Referências

1. "An Empirical Comparison of Botnet Detection Methods" - GARCÍA et al., 2014
