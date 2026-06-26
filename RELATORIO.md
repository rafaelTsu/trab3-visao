# Segmentação Semântica de Animais de Estimação com FCN-8s, U-Net e DeepLabV3+

**Disciplina:** Visão Computacional — UFMS

**Integrantes:** Jerfferson Jorge e Rafael Tsutomu

**Trabalho Final**

---

## 1. Introdução

### Descrição do problema
O objetivo é realizar **segmentação semântica** de imagens de animais de estimação: atribuir a
**cada pixel** uma de três classes — **animal**, **fundo** e **borda** (contorno do animal).
Diferente da classificação de imagens (foco do Trabalho 2), a segmentação exige predição densa, com
preservação da estrutura espacial; os principais desafios são a recuperação de resolução após o
*encoder* e a delimitação correta dos contornos.

### Objetivo do trabalho
1. Implementar **do zero** três arquiteturas representativas de segmentação — **FCN-8s**, **U-Net** e
   **DeepLabV3+** — e compará-las sob o mesmo protocolo de treino.
2. Avaliar quatro fatores por meio de ablações na melhor arquitetura: **função de perda**,
   **data augmentation**, **Batch Normalization** e **transfer learning**.
3. Reportar métricas adequadas a segmentação (mIoU, Dice, *pixel accuracy*), com análise por classe e
   por imagem, além de exemplos visuais das predições.

---

## 2. Base de imagens

- **Nome / fonte:** **Oxford-IIIT Pet** (Parkhi et al., 2012), carregado via
  `torchvision.datasets.OxfordIIITPet` com `target_types="segmentation"` (download automático).
- **Quantidade:** 7.349 imagens de 37 raças de cães e gatos, cada uma com uma máscara *trimap*.
- **Classes (3):** os valores originais do trimap — `1` (animal), `2` (fundo) e `3` (borda) — são
  **remapeados para `0` (animal), `1` (fundo) e `2` (borda)** para uso com `CrossEntropyLoss`. A
  classe **borda** é a mais rara e a mais difícil (poucos pixels, alta ambiguidade).
- **Divisão dos dados:** usamos o split oficial `trainval` e `test`. Do `trainval` separamos
  **treino/validação 80/20** com gerador semeado (`seed=42`); o conjunto de **teste** foi usado
  apenas na avaliação final.

| Conjunto | Imagens |
|---|---|
| Treino | 2.944 |
| Validação | 736 |
| Teste | 3.669 |

**Exemplos** (imagem | máscara *ground-truth* | sobreposição):

![Amostras do dataset](imagens/amostras.png)

### Pré-processamento
- Redimensionamento para **128×128**. Máscaras redimensionadas com interpolação **nearest** (para não
  gerar rótulos inválidos por interpolação).
- **Normalização ImageNet** (média `[0.485, 0.456, 0.406]`, desvio `[0.229, 0.224, 0.225]`),
  necessária para o encoder pré-treinado e adotada de forma consistente em todos os modelos.

---

## 3. Metodologia

### Tipo de tarefa
Segmentação semântica multiclasse (3 classes), predição densa pixel a pixel.

### Arquiteturas (todas implementadas do zero)
- **FCN-8s** (Long et al., 2015) — *baseline*. Encoder convolucional estilo VGG; classificação por
  `Conv1×1` em baixa resolução; *upsampling* por `ConvTranspose2d` com **fusão aditiva** dos mapas
  intermediários `pool3` e `pool4` (*skip connections* grosseiras). **20,50 M** parâmetros.
- **U-Net** (Ronneberger et al., 2015) — *encoder-decoder* simétrico com **skip connections por
  concatenação** em todos os níveis (`DoubleConv` + `Down`/`Up`). **31,04 M** parâmetros.
- **DeepLabV3+** (Chen et al., 2018) — *backbone* ResNet (do zero) com **atrous convolutions**
  (*output stride* 16), **ASPP** (contexto multi-escala) e **decoder** com fusão de *low-level
  features*. **16,60 M** parâmetros.

A inicialização de pesos é **Kaiming (He)** em todas as convoluções.

### Data augmentation
Aplicado **apenas no treino** e **sincronizado** entre imagem e máscara (classe `JointTransform`):
flip horizontal, `RandomAffine` (rotação ±15°, translação ±10%) e `ColorJitter` (brilho/contraste/
saturação, **só na imagem**).

### Funções de perda
- **CrossEntropy** (`nn.CrossEntropyLoss`) — baseline.
- **Dice loss** (implementada do zero) — otimiza diretamente a sobreposição.
- **Combo CE+Dice** (do zero) — soma das duas.

### Hiperparâmetros principais
| Hiperparâmetro | Valor |
|---|---|
| Otimizador | Adam |
| Taxa de aprendizado | `1e-3` |
| Scheduler | CosineAnnealingLR |
| Batch size | 16 |
| Épocas (comparativo) | 30 |
| Épocas (ablações) | 15 |
| Resolução | 128×128 |
| Semente | 42 |

A seleção do melhor modelo usa **checkpoint pelo maior mIoU de validação**.

### Ferramentas e bibliotecas
PyTorch (CUDA), torchvision (dataset, transforms, pesos pré-treinados da ResNet-34), NumPy e
Matplotlib. **Implementados do zero:** as três arquiteturas (blocos e `forward`), as perdas Dice e
CE+Dice, as métricas (IoU/mIoU, Dice, *pixel accuracy* via matriz de confusão) e o laço de treino.
**De biblioteca:** primitivas `nn.*`, download do dataset, otimizador/scheduler e pesos pré-treinados
(somente na ablação de *transfer learning*). Detalhes em `PLANEJAMENTO.md`.

---

## 4. Experimentos

### 4.1 Comparativo de arquiteturas
As três arquiteturas treinadas no **mesmo protocolo**: perda CrossEntropy, 30 épocas, sem
augmentation, `seed=42`. Comparadas por mIoU/Dice/pixel accuracy, número de parâmetros e tempo de
treino.

### 4.2 Ablações (na melhor arquitetura)
Variando **um fator por vez**, com 15 épocas e `seed=42`:
1. **Função de perda:** CE vs Dice vs CE+Dice.
2. **Data augmentation:** com vs sem.
3. **Batch Normalization:** com vs sem (U-Net, via `use_bn`).
4. **Transfer learning:** encoder ResNet-34 pré-treinado (ImageNet) vs do zero.

---

## 5. Resultados

### 5.1 Comparativo de arquiteturas (validação, 30 épocas)

| Arquitetura | Params (M) | mIoU | Dice | Pixel acc | Tempo (s) |
|---|---|---|---|---|---|
| FCN-8s | 20,50 | 0,6788 | 0,7891 | 0,8731 | 710 |
| **U-Net** | 31,04 | **0,7282** | **0,8295** | **0,8961** | 1460 |
| DeepLabV3+ | 16,60 | 0,6720 | 0,7799 | 0,8640 | 583 |

![Comparativo de arquiteturas](imagens/comparativo_arquiteturas.png)

- A **U-Net venceu** em todas as métricas (mIoU 0,7282), com curva de validação consistentemente
  acima das demais a partir da época ~8.
- O **DeepLabV3+** convergiu rápido no início, mas sua **perda de validação volta a subir após a
  época ~12** (sinal de *overfitting*), terminando ligeiramente atrás da FCN-8s em mIoU apesar de ser
  o modelo com **menos parâmetros** (16,60 M) e o **mais rápido** de treinar (583 s).
- A **FCN-8s** converge de forma mais lenta e ruidosa (queda de mIoU na época 19), mas estabiliza em
  0,6788.
- O **custo de treino** acompanha a profundidade efetiva: U-Net foi a mais cara (1460 s), cerca do
  dobro das outras duas.

### 5.2 Ablações (mIoU de validação, U-Net, 15 épocas)

| Experimento | mIoU |
|---|---|
| Perda: CE | 0,7139 |
| **Perda: Dice** | **0,7242** |
| Perda: CE+Dice | 0,7186 |
| Augmentation: não | 0,7070 |
| Augmentation: sim | 0,7114 |
| BatchNorm: sim | 0,7040 |
| BatchNorm: não | 0,6880 |
| Transfer: do zero | 0,6875 |
| **Transfer: pré-treinado** | **0,7507** |

![Ablações](imagens/ablacoes.png)

- **Função de perda:** **Dice** (0,7242) superou CE+Dice (0,7186) e CE puro (0,7139). A perda baseada
  em sobreposição ajudou a métrica de IoU.
- **Data augmentation:** ganho pequeno mas positivo (0,7114 vs 0,7070). Com apenas 15 épocas e dataset
  pequeno, o efeito regularizador ainda é modesto.
- **Batch Normalization:** **acelerou e melhorou** o treino (0,7040 com BN vs 0,6880 sem BN).
- **Transfer learning:** **maior impacto de todos**. O encoder ResNet-34 pré-treinado no ImageNet
  atingiu **0,7507 em apenas 15 épocas** — superando inclusive a U-Net do zero treinada por 30 épocas
  (0,7282). O encoder ResNet-34 do zero (sem pré-treino) ficou em 0,6875, o que isola o ganho como
  efeito do **pré-treino**, não da arquitetura do encoder.

### 5.3 Avaliação final no conjunto de teste

Melhor arquitetura do comparativo: **U-Net** (melhores pesos por mIoU de validação), avaliada no
**conjunto de teste oficial** (3.669 imagens, nunca usado em treino/validação).

| Métrica | Valor |
|---|---|
| mIoU | 0,7408 |
| Dice | 0,8395 |
| Pixel accuracy | 0,9010 |

**IoU por classe:**

| Classe | IoU |
|---|---|
| animal | 0,8158 |
| fundo | 0,8995 |
| **borda** | **0,5071** |

A classe **fundo** é a mais fácil (0,8995) e a **borda** é, de longe, a mais difícil (0,5071),
puxando o mIoU para baixo.

**Matriz de confusão** (normalizada por linha / classe verdadeira, em nível de pixel):

![Matriz de confusão](imagens/matriz_confusao.png)

A matriz confirma o diagnóstico: as classes **animal** e **fundo** têm alta taxa na diagonal,
enquanto os pixels de **borda** são frequentemente confundidos com **animal** e **fundo** — esperado,
pois a borda é uma faixa fina de transição entre as outras duas classes.

**Distribuição por imagem** (a partir de `PETS_RESULTS.csv`, 3.669 imagens):

| Estatística (mIoU por imagem) | Valor |
|---|---|
| Média | 0,7356 |
| Mediana | 0,7625 |
| Desvio padrão | 0,1118 |
| Q25 – Q75 | 0,6859 – 0,8145 |
| Mínimo / Máximo | 0,0083 / 0,9098 |
| Imagens com mIoU > 0,7 | 2.613 (71,2%) |
| Imagens com mIoU > 0,8 | 1.198 (32,7%) |
| Imagens com mIoU < 0,5 | 153 (4,2%) |

A maioria das imagens (71,2%) tem mIoU acima de 0,7 e apenas 4,2% ficam abaixo de 0,5, indicando
desempenho robusto com poucos casos de falha grave.

**Exemplos de predições** (imagem | *ground-truth* | predição):

![Predições no teste](imagens/predicoes_teste.png)

As predições reproduzem bem a silhueta do animal e o fundo; os erros se concentram na **faixa de
borda** e em cenas com **camuflagem/fundo complexo** (ex.: a 4ª linha da figura), coerente com a baixa
IoU da classe borda e com os casos de mIoU mínimo na distribuição por imagem.

---

## 6. Discussão

### Pontos fortes da solução
- **Comparativo justo e reprodutível:** três arquiteturas implementadas do zero, mesmo protocolo,
  mesma semente. A **U-Net** se mostrou a melhor escolha *do zero* para este dataset pequeno, graças
  às *skip connections* densas que recuperam detalhes espaciais.
- **Transfer learning** foi a alavanca mais eficaz: o encoder ResNet-34 pré-treinado atingiu o melhor
  mIoU (0,7507) na metade das épocas, confirmando o valor de *features* pré-treinadas mesmo em um
  domínio (animais) diferente do ImageNet.
- **Bom desempenho global:** pixel accuracy de 0,9010 e 71,2% das imagens com mIoU > 0,7 no teste.

### Limitações do modelo
- **Classe borda** é o gargalo (IoU 0,5071): poucos pixels e ambiguidade intrínseca do contorno. As
  perdas baseadas em Dice ajudaram, mas não resolveram.
- **Resolução 128×128** limita o detalhamento dos contornos; resoluções maiores tendem a beneficiar
  justamente a classe borda, ao custo de tempo.
- **DeepLabV3+ do zero** mostrou *overfitting* (perda de validação crescente) — arquiteturas com
  *backbone* tipo ResNet são mais dependentes de pré-treino e de mais dados.

### Principais dificuldades encontradas
- Sincronizar as transformações geométricas entre **imagem e máscara** (resolvido com a
  `JointTransform`, usando interpolação *nearest* na máscara).
- *Batch Normalization* no ramo de *pooling* global do **ASPP** com batches pequenos (mitigado por
  `drop_last=True` no treino e estatísticas acumuladas na avaliação).
- **Custo computacional** de treinar três arquiteturas por 30 épocas mais nove ablações — exigiu GPU
  (Google Colab).

### Possíveis melhorias futuras
- Combinar as duas melhores descobertas: **U-Net com encoder pré-treinado + perda Dice**, treinada por
  mais épocas e em **256×256**.
- *Backbone* mais profundo (ResNet-50) com atrous, e **perda de Dice generalizada** (Sudre et al.,
  2017) ponderada por classe para reforçar a borda.
- Pós-processamento de contornos (CRF) e augmentation mais agressivo com mais épocas.

---

## 7. Conclusão

### Melhor modelo encontrado
Entre as arquiteturas treinadas do zero, a **U-Net** foi a melhor (teste: **mIoU 0,7408**,
**Dice 0,8395**, **pixel accuracy 0,9010**). Nas ablações, a configuração de maior mIoU de validação
foi a **U-Net com encoder ResNet-34 pré-treinado** (**0,7507**), apontando o caminho da melhor solução
final.

### Principais aprendizados
- Em datasets pequenos, *skip connections* densas (U-Net) superam arquiteturas com mais contexto mas
  mais dependentes de dados/pré-treino (DeepLabV3+).
- *Transfer learning* e *Batch Normalization* trazem ganhos claros; perdas baseadas em Dice ajudam a
  métrica de IoU, sobretudo nas classes raras.
- A classe **borda** é o principal limitador e o alvo natural de trabalhos futuros.

---

## 8. Referências

1. **Long, J., Shelhamer, E., Darrell, T.** (2015). *Fully Convolutional Networks for Semantic Segmentation.* CVPR.
2. **Ronneberger, O., Fischer, P., Brox, T.** (2015). *U-Net: Convolutional Networks for Biomedical Image Segmentation.* MICCAI.
3. **Chen, L.-C., Zhu, Y., Papandreou, G., Schroff, F., Adam, H.** (2018). *Encoder-Decoder with Atrous Separable Convolution for Semantic Image Segmentation (DeepLabV3+).* ECCV.
4. **Milletari, F., Navab, N., Ahmadi, S.-A.** (2016). *V-Net: Fully Convolutional Neural Networks for Volumetric Medical Image Segmentation.* 3DV. (Dice loss)
5. **Sudre, C. H. et al.** (2017). *Generalised Dice Overlap as a Deep Learning Loss Function for Highly Unbalanced Segmentations.* DLMIA.
6. **He, K., Zhang, X., Ren, S., Sun, J.** (2016). *Deep Residual Learning for Image Recognition (ResNet).* CVPR.
7. **Ioffe, S., Szegedy, C.** (2015). *Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift.* ICML.
8. **Kingma, D. P., Ba, J.** (2015). *Adam: A Method for Stochastic Optimization.* ICLR.
9. **Parkhi, O. M., Vedaldi, A., Zisserman, A., Jawahar, C. V.** (2012). *Cats and Dogs (Oxford-IIIT Pet Dataset).* CVPR.
10. **Paszke, A. et al.** (2019). *PyTorch: An Imperative Style, High-Performance Deep Learning Library.* NeurIPS. Documentação: PyTorch e torchvision.
