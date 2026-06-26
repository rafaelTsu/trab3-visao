# Planejamento do Projeto — Trabalho Final de Visão Computacional

## Visão geral

- **Tarefa:** Segmentação semântica de imagens.
- **Abordagem:** comparativo de **3 arquiteturas implementadas do zero** — **FCN-8s**, **U-Net** e **DeepLabV3+** — mais experimentos controlados (ablações).
- **Dataset:** Oxford-IIIT Pet (trimap de 3 classes: animal, fundo, borda), baixado automaticamente pelo `torchvision`.
- **Hardware-alvo:** Google Colab / Kaggle (GPU).
- **Grupo:** Jerfferson Jorge e Rafael Tsutomu.
- **Formato do código:** um único notebook (`trabalho_final.ipynb`), no estilo "do Zero" dos notebooks do professor.

A escolha de segmentação dá continuidade ao Trabalho 2 (classificação com CNN do zero), reutiliza as
convenções do professor (PyTorch, arquiteturas construídas na mão, células de verificação) e usa três
arquiteturas que formam uma linha evolutiva (FCN → U-Net → DeepLabV3+), com métricas próprias de
segmentação (IoU, Dice, pixel accuracy).

## O que é feito do zero vs. o que usamos pronto

| Componente | Do zero (na mão) | Pronto (biblioteca) |
|---|---|---|
| **As 3 arquiteturas** (FCN-8s, U-Net, DeepLabV3+) | blocos e `forward` de cada uma (`DoubleConv`/skips na U-Net; fusão de skips no FCN; `ASPP`+atrous no DeepLabV3+; `BasicBlock` da ResNet), init Kaiming | primitivas `nn.Conv2d/BatchNorm2d/ReLU/ConvTranspose2d/MaxPool2d` |
| **Loop de treino/validação** | `train_one_epoch`, `evaluate`, `fit` (history + checkpoint do melhor mIoU) | `torch.optim.Adam`, `CosineAnnealingLR` |
| **Funções de perda** | **Dice loss** e **combo CE+Dice** | `nn.CrossEntropyLoss` |
| **Métricas** | **IoU/mIoU**, **Dice**, **pixel accuracy** (via matriz de confusão) | — |
| **Dataset / transforms conjuntas** | `JointTransform` (img+máscara sincronizadas), remap do trimap | `torchvision.datasets.OxfordIIITPet`, `transforms.functional` |
| **Transfer learning (1 ablação)** | acoplamento do encoder ResNet ao decoder U-Net | `torchvision.models.resnet34` (somente pesos ImageNet) |
| **Gráficos** | montagem das figuras | `matplotlib` |

## Estrutura de arquivos

```
trab3/
├── PLANEJAMENTO.md       # este arquivo
├── README.md             # instruções de execução (Colab + local)
├── RELATORIO.md          # relatório técnico (6 itens do enunciado)
├── requirements.txt
├── trabalho_final.ipynb  # ÚNICO notebook (dados, 3 modelos, treino, experimentos, avaliação)
└── imagens/              # figuras geradas para o relatório
```

## Seções do notebook

| Seção | Conteúdo |
|---|---|
| 0. Setup | imports, dispositivo, `SEED=42`, hiperparâmetros globais |
| 1. Dados | OxfordIIITPet, remap do trimap (1/2/3→0/1/2), split 80/20 (seed=42), dataloaders, amostras |
| 2. Métricas | IoU/mIoU, Dice, pixel accuracy (do zero) + teste trivial |
| 3. Perdas | CrossEntropy (base), Dice e CE+Dice (do zero) |
| 4. Motor de treino | `train_one_epoch`, `evaluate`, `fit` (history + checkpoint do melhor mIoU) |
| 5. U-Net (do zero) | + verificação de shapes/params + overfit de 1 batch |
| 6. FCN-8s (do zero) | + verificação |
| 7. DeepLabV3+ (do zero) | backbone ResNet atrous + ASPP + decoder + verificação |
| 8. Comparativo | treina as 3 no mesmo setup; tabela + curvas |
| 9. Ablações | perda, data augmentation, BatchNorm, transfer learning |
| 10. Avaliação final | melhor modelo no teste oficial; matriz de confusão, distribuição por imagem e grade de predições |

## Plano de experimentos

**Principal — comparativo de arquiteturas:** FCN-8s vs U-Net vs DeepLabV3+, mesmo setup (CE,
mesmas épocas, sem augmentation, `seed=42`). Comparados por mIoU/Dice/pixel acc, nº de parâmetros e
tempo de treino.

**Ablações (na melhor arquitetura do comparativo):**
1. Função de perda: CE vs Dice vs CE+Dice.
2. Data augmentation: com vs sem (flip, affine, color jitter — só no treino).
3. BatchNorm: com vs sem (U-Net, via `use_bn`).
4. Transfer learning: encoder ResNet-34 pré-treinado (ImageNet) vs do zero.

**Hiperparâmetros base:** `Adam lr=1e-3`, `batch_size=16`, `epochs=30` (reduzir para teste rápido),
`CosineAnnealingLR`, entrada `128×128` (usar `256` para o resultado final).

## Métricas (adequadas a segmentação)

IoU por classe e mIoU (principal), Dice coefficient, pixel accuracy, curvas de loss/mIoU por época e
exemplos visuais (imagem | ground-truth | predição).

## Referências principais

- Long, Shelhamer, Darrell (2015) — **FCN**.
- Ronneberger, Fischer, Brox (2015) — **U-Net**.
- Chen et al. (2018) — **DeepLabV3+**.
- Milletari et al. (2016) — **Dice loss (V-Net)**; Sudre et al. (2017) — Generalised Dice.
- He et al. (2016) — **ResNet**; Ioffe & Szegedy (2015) — **BatchNorm**; Kingma & Ba (2015) — **Adam**.
- Parkhi et al. (2012) — **Oxford-IIIT Pet**.
- Documentação: PyTorch, torchvision.

## Como validamos (corretude)

- Verificação de shape entrada→saída e contagem de parâmetros de cada arquitetura.
- Teste trivial das métricas (predição = GT ⇒ IoU = Dice = 1).
- Overfit de 1 batch (a perda deve cair para ~0).
- `seed=42` fixo em `random`/`numpy`/`torch` para reprodutibilidade.
- Execução de ponta a ponta no Colab (notebook roda do início ao fim; figuras geradas).
