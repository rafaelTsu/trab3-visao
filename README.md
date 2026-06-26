# Trabalho Final — Segmentação Semântica

Disciplina de Visão Computacional — UFMS. Comparativo de **três arquiteturas de segmentação** (FCN-8s, U-Net, DeepLabV3+) sobre o dataset **Oxford-IIIT Pet**, com
experimentos controlados de função de perda, *data augmentation*, *Batch Normalization* e *transfer learning*.

**Alunos:** Jerfferson Jorge e Rafael Tsutomu.

## Arquivos

| Arquivo | Descrição |
|---|---|
| `trabalho_final.ipynb` | Notebook principal: dados, modelos, treino, experimentos e avaliação |
| `RELATORIO.md` | Relatório técnico |
| `requirements.txt` | Dependências |
| `imagens/` | Figuras geradas pelo notebook |

## Como executar

### Google Colab / Kaggle

1. Faça upload de `trabalho_final.ipynb`.
2. `Runtime → Change runtime type → GPU`.
3. `Runtime → Run all`. O dataset Oxford-IIIT Pet é baixado automaticamente na
   primeira execução.

### Local

```bash
pip install -r requirements.txt
jupyter notebook trabalho_final.ipynb
```

Sem GPU, o treino completo é lento.

## Parâmetros úteis

- `IMG_SIZE` — resolução de treino (`128` para iterar; `256` para o resultado final).
- `EPOCHS` — número de épocas do comparativo (`ABLATION_EPOCHS` é derivado dele).
- `BATCH_SIZE`, `LR` — tamanho de batch e taxa de aprendizado.
- `SEED = 42` — semente fixa para reprodutibilidade.

## Reprodutibilidade

`SEED = 42` é fixado em `random`, `numpy` e `torch`, e o split treino/validação (80/20) usa um
gerador semeado. Rode as células em ordem.
