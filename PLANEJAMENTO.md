# Planejamento — Trabalho Final de Visão Computacional

## Segmentação Semântica no Oxford-IIIT Pet

**Disciplina:** Visão Computacional — UFMS
**Integrantes:** Rafael Tsutomu e Jerfferson Jorge

---

## Contexto

O trabalho final (ver `trab3/trabalho_final.pdf`) é um projeto prático de Deep Learning
em uma de três tarefas (classificação, detecção ou segmentação), acompanhado de um
relatório técnico no formato da seção experimental de um artigo, com 6 itens
obrigatórios: **Base de imagens, Metodologia, Experimentos, Resultados, Discussão e
Referências**.

Escolhemos **segmentação semântica** por ser distinta da tarefa do Trabalho 2
(classificação) e por ser bem coberta pelos exemplos do professor
(`codigo_exemplo/segmentation.ipynb` — U-Net; `codigo_exemplo/deeplabv3plus.ipynb` —
DeepLabV3+).

### Critérios herdados do Trabalho 2

1. Split fixo e reprodutível (`SEED = 42`).
2. Conjunto de teste usado **apenas na avaliação final**.
3. **Todos os modelos treinados do zero** (sem pesos pré-treinados / sem transfer learning).
4. Estrutura baseline + experimentos controlados + análise por classe + relatório em Markdown.

---

## Escolhas técnicas

| Item | Escolha | Justificativa |
|---|---|---|
| Tarefa | Segmentação semântica | Distinta do Trab 2; coberta pelos exemplos do professor. |
| Dataset | **Oxford-IIIT Pet** (máscaras) | Público, `torchvision.datasets.OxfordIIITPet(target_types="segmentation")`, ~7.349 imagens, leve para Colab. |
| Classes | 3: animal / fundo / contorno | Trimap remapeado `{1,2,3}→{0,1,2}`. Classe *contorno* é minoritária → desbalanceamento análogo ao Trab 2. |
| Arquiteturas | **U-Net do zero** (baseline) + **DeepLabV3+ do zero** (backbone ResNet-50 sem pesos ImageNet) | Comparação arquitetural (Experimento 1). |
| Perda | CrossEntropy / Dice / CE+Dice | Dice ajuda a classe minoritária (contorno). |
| Otimizador | Adam (lr=1e-3) | Padrão dos exemplos do professor. |
| Métricas | mean IoU, IoU por classe, Dice, pixel accuracy | Exatamente as sugeridas no enunciado para segmentação. |
| Execução | Notebook `.ipynb` no Google Colab (GPU) | Confirmado. |

---

## Arquivos do projeto

```
trab3/
├── PLANEJAMENTO.md          # este documento
├── trabalho_final.pdf       # enunciado
├── segmentacao_pets.ipynb   # notebook principal (Colab/GPU), modelos do zero
├── RELATORIO.md             # relatório técnico (6 seções do enunciado)
└── imagens/                 # figuras geradas (amostras, curvas, IoU por classe, overlays)
```

---

## Estrutura do notebook `segmentacao_pets.ipynb`

1. **Setup** — imports, device GPU, `SEED=42` em `random`/`numpy`/`torch`.
2. **Dataset** — `OxfordIIITPet(target_types="segmentation")`; split oficial
   3.680 trainval / 3.669 teste; trainval → treino/val 90/10 com seed fixa.
3. **Transforms conjuntos (imagem+máscara)** — Resize 128×128 (máscara *nearest*),
   normalização ImageNet; augmentation geométrica igual para imagem e máscara
   (`RandomResizedCrop`, `RandomHorizontalFlip`, `RandomRotation`) + `ColorJitter`
   só na imagem.
4. **Visualização** — amostras com máscara em overlay + distribuição de pixels por classe.
5. **Métricas** — `compute_iou` (mIoU e por classe), `dice_coefficient`, `pixel_accuracy`.
6. **U-Net do zero** (baseline) — blocos down/up (base: `segmentation.ipynb`).
7. **DeepLabV3+ do zero** — backbone ResNet-50 do zero (Kaiming init, sem ImageNet),
   ASPP + decoder (base: `deeplabv3plus.ipynb`).
8. **Helpers de treino** — `train_one_epoch`, `evaluate`, dict `history`.
9. **Experimentos controlados** (espelhando o Trab 2):
   - **Exp 1 — Arquitetura:** U-Net vs DeepLabV3+ (mesmo treino/épocas).
   - **Exp 2 — Função de perda:** CrossEntropy vs Dice vs CE+Dice.
   - **Exp 3 — Data augmentation:** com vs sem.
   - *Sem experimento de transfer learning (restrição: tudo treinado do zero).*
10. **Avaliação final no teste** — melhor modelo; mIoU/Dice/pixel acc; IoU por classe;
    predições visuais (original | máscara real | predição).
11. **Conclusões**.

### Hiperparâmetros base
- Entrada 128×128; `batch_size` 16; Adam `lr=1e-3`; ~30 épocas; semente fixa.

---

## Relatório `RELATORIO.md` (mapeado aos 6 itens do enunciado)

1. **Base de imagens** — nome (Oxford-IIIT Pet), fonte (Parkhi et al., 2012), quantidade
   (7.349), 3 classes do trimap, divisão treino/val/teste, exemplos visuais.
2. **Metodologia** — tarefa (segmentação), arquiteturas (U-Net e DeepLabV3+ do zero),
   pré-processamento, augmentation, hiperparâmetros, ferramentas (PyTorch/torchvision).
3. **Experimentos** — tabela-resumo das 3 comparações controladas.
4. **Resultados** — tabelas (mIoU, Dice, pixel acc, IoU por classe), curvas de treino,
   IoU por classe, predições visuais.
5. **Discussão** — pontos fortes, limitações (ex.: contorno fino perdido em 128×128),
   dificuldades (sincronizar augmentation imagem/máscara, memória/tempo do treino do
   zero), melhorias futuras (resolução maior, Lovász loss, pós-processamento CRF).
6. **Referências** — lista abaixo.

---

## Referências

- Ronneberger et al. (2015) — *U-Net: Convolutional Networks for Biomedical Image Segmentation*.
- Chen et al. (2018) — *Encoder-Decoder with Atrous Separable Convolution (DeepLabV3+)*.
- Long et al. (2015) — *Fully Convolutional Networks for Semantic Segmentation (FCN)*.
- He et al. (2016) — *Deep Residual Learning (ResNet)* — backbone treinado do zero.
- Milletari et al. (2016) — *V-Net / Dice loss*.
- Parkhi et al. (2012) — *Cats and Dogs (Oxford-IIIT Pet dataset)*.
- Ioffe & Szegedy (2015) — *Batch Normalization*; Kingma & Ba (2015) — *Adam*.
- Documentação PyTorch / torchvision.

---

## Verificação (teste de ponta a ponta)

1. Abrir `segmentacao_pets.ipynb` no Google Colab com runtime **GPU**.
2. *Run all* — download automático do dataset; conferir split (3.680/3.669) e amostras.
3. Treinar baseline U-Net e checar curvas (`history` registra loss e mIoU por época).
4. Rodar os 3 experimentos; conferir tabela-resumo (mIoU/Dice/pixel acc).
5. Avaliação final no teste **somente ao fim**; gerar overlays em `imagens/`.
6. Confirmar que **nenhum** modelo usa pesos pré-treinados (tudo do zero).
7. Revisar `RELATORIO.md` cobrindo os 6 itens e validar as figuras referenciadas.
