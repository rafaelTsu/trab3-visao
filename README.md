# Trabalho Final — Segmentação Semântica (Oxford-IIIT Pet)

Disciplina de Visão Computacional — UFMS. Comparativo de **três arquiteturas de segmentação
implementadas do zero** (FCN-8s, U-Net, DeepLabV3+) sobre o dataset **Oxford-IIIT Pet**, com
experimentos controlados de função de perda, *data augmentation*, *Batch Normalization* e
*transfer learning*.

**Integrantes:** Jerfferson Jorge e Rafael Tsutomu.

## Arquivos

| Arquivo | Descrição |
|---|---|
| `trabalho_final.ipynb` | Notebook principal: dados, modelos, treino, experimentos e avaliação |
| `RELATORIO.md` | Relatório técnico |
| `PLANEJAMENTO.md` | Planejamento do projeto e do diretório |
| `requirements.txt` | Dependências |
| `imagens/` | Figuras geradas pelo notebook |
| `PETS_RESULTS.csv` | Métricas por imagem no conjunto de teste (gerado ao executar) |

## Como executar

### Google Colab / Kaggle (recomendado — usa GPU grátis)

1. Faça upload de `trabalho_final.ipynb` (ou abra direto do GitHub/Drive).
2. `Runtime → Change runtime type → GPU`.
3. `Runtime → Run all`. O dataset Oxford-IIIT Pet (~800 MB) é baixado automaticamente na
   primeira execução.

### Local

```bash
pip install -r requirements.txt
jupyter notebook trabalho_final.ipynb
```

Sem GPU, o treino completo é lento. Para um teste rápido, reduza `EPOCHS` na **Seção 0** (ex.: `EPOCHS = 3`).

## Parâmetros úteis (Seção 0 do notebook)

- `IMG_SIZE` — resolução de treino (`128` para iterar; `256` para o resultado final).
- `EPOCHS` — número de épocas do comparativo (`ABLATION_EPOCHS` é derivado dele).
- `BATCH_SIZE`, `LR` — tamanho de batch e taxa de aprendizado.
- `SEED = 42` — semente fixa para reprodutibilidade.

## Reprodutibilidade

`SEED = 42` é fixado em `random`, `numpy` e `torch`, e o split treino/validação (80/20) usa um
gerador semeado. Rode as células em ordem.

## O que é feito do zero vs. pronto

Implementados **do zero**: as três arquiteturas, as perdas (Dice e CE+Dice), as métricas
(IoU/mIoU, Dice, pixel accuracy) e o laço de treino. **Bibliotecas** apenas para: primitivas `nn.*`,
download do dataset, otimizador/scheduler e pesos pré-treinados (somente no experimento de
*transfer learning*). Detalhes em `PLANEJAMENTO.md`.
