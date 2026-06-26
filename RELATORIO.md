# Segmentação Semântica de Animais de Estimação com FCN-8s, U-Net e DeepLabV3+

**Disciplina:** Visão Computacional, UFMS

**Alunos:** Jerfferson Jorge e Rafael Tsutomu

---

## 1. Introdução

### Descrição do problema
O objetivo deste trabalho é realizar segmentação semântica de imagens de animais de estimação, ou
seja, atribuir a cada pixel da imagem uma de três classes possíveis: animal, fundo e borda (o
contorno do animal). Diferentemente da classificação de imagens, que foi o foco do Trabalho 2, a
segmentação exige uma predição densa e a preservação da estrutura espacial da cena. Por isso, os
principais desafios da tarefa são recuperar a resolução espacial após a etapa de codificação e
delimitar corretamente os contornos dos objetos.

### Objetivo do trabalho
O trabalho tem três objetivos centrais. O primeiro é implementar três arquiteturas
representativas de segmentação, a FCN-8s, a U-Net e a DeepLabV3+, comparando-as sob um mesmo protocolo
de treino. O segundo é avaliar, por meio de ablações na melhor arquitetura, o efeito de quatro
fatores: a função de perda, o uso de data augmentation, a presença de Batch Normalization e o uso de
transfer learning. O terceiro é reportar métricas adequadas à tarefa de segmentação (mIoU, Dice e
pixel accuracy), acompanhadas de análise por classe, análise por imagem e exemplos visuais das
predições.

---

## 2. Base de imagens

A base utilizada foi a Oxford-IIIT Pet (Parkhi et al., 2012), carregada por meio de
`torchvision.datasets.OxfordIIITPet` com `target_types="segmentation"`, o que faz o download dos
dados automaticamente. O conjunto reúne 7.349 imagens de 37 raças de cães e gatos, e cada imagem
possui uma máscara *trimap* anotada pixel a pixel.

A máscara original do *trimap* usa os valores 1 para o animal, 2 para o fundo e 3 para a borda. Esses
valores foram remapeados para 0 (animal), 1 (fundo) e 2 (borda), de modo a servirem como rótulo para
a `CrossEntropyLoss`. Vale destacar desde já que a classe borda é a mais rara e a mais difícil do
conjunto, pois corresponde a poucos pixels e apresenta ambiguidade natural na transição entre o
animal e o fundo.

Quanto à divisão dos dados, usamos o split oficial do dataset, que separa os conjuntos `trainval` e
`test`. A partir do `trainval`, separamos treino e validação em uma proporção de 80/20 usando um
gerador com semente fixa (`seed=42`), enquanto o conjunto de teste foi reservado exclusivamente para
a avaliação final.

| Conjunto | Imagens |
|---|---|
| Treino | 2.944 |
| Validação | 736 |
| Teste | 3.669 |

A figura a seguir mostra exemplos do dataset, apresentando para cada amostra a imagem, sua máscara
*ground-truth* e a sobreposição das duas.

![Amostras do dataset](imagens/amostras.png)

### Pré-processamento
Todas as imagens foram redimensionadas para 128×128. As máscaras foram redimensionadas com
interpolação *nearest*, evitando que a interpolação gere rótulos inválidos entre as classes. Além
disso, as imagens foram normalizadas com a média e o desvio padrão do ImageNet (média
`[0.485, 0.456, 0.406]` e desvio `[0.229, 0.224, 0.225]`), o que é necessário para o encoder
pré-treinado e foi adotado de forma consistente em todos os modelos para garantir uma comparação
justa.

---

## 3. Metodologia

### Tipo de tarefa
Trata-se de segmentação semântica multiclasse, com três classes, realizando predição densa pixel a
pixel.

### Arquiteturas
A FCN-8s (Long et al., 2015) é o nosso *baseline*. Ela emprega um encoder convolucional no estilo VGG,
faz a classificação por uma convolução 1×1 em baixa resolução e recupera a resolução com
`ConvTranspose2d`, fundindo de forma aditiva os mapas intermediários `pool3` e `pool4` por meio de
*skip connections* grosseiras. No total, possui 20,50 milhões de parâmetros.

A U-Net (Ronneberger et al., 2015) é um *encoder-decoder* simétrico cujo diferencial são as *skip
connections* por concatenação presentes em todos os níveis, ligando o caminho de contração ao de
expansão (blocos `DoubleConv`, `Down` e `Up`). É a maior das três arquiteturas, com 31,04 milhões de
parâmetros.

A DeepLabV3+ (Chen et al., 2018) utiliza um *backbone* ResNet com
*atrous convolutions* (mantendo *output stride* 16), o módulo ASPP para capturar contexto em múltiplas
escalas e um decoder que funde *low-level features*. É a mais compacta das três, com 16,60 milhões de
parâmetros.

Em todas as arquiteturas, a inicialização dos pesos das convoluções segue o esquema de Kaiming (He).

### Data augmentation
O *augmentation* é aplicado apenas no conjunto de treino e de forma sincronizada entre a imagem e a
máscara, por meio da classe `JointTransform`. As transformações usadas são o flip horizontal, a
`RandomAffine` (com rotação de até ±15° e translação de até ±10%) e a `ColorJitter` (variando brilho,
contraste e saturação). A `ColorJitter` é aplicada somente na imagem, já que alterar as cores não faz
sentido para a máscara de rótulos.

### Funções de perda
Foram consideradas três funções de perda. A `CrossEntropyLoss` serve como baseline padrão para
classificação por pixel. A Dice loss otimiza diretamente a sobreposição entre
a predição e o *ground-truth*. Por fim, a combinação CE+Dice soma as
duas perdas, buscando unir a estabilidade da entropia cruzada ao foco em sobreposição do Dice.

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

A seleção do melhor modelo de cada treino é feita por *checkpoint*, guardando os pesos da época com
maior mIoU de validação.

### Ferramentas e bibliotecas
O projeto foi desenvolvido em PyTorch com aceleração CUDA, usando o torchvision para o dataset, as
transformações e os pesos pré-treinados da ResNet-34, além de NumPy e Matplotlib. Implementamos
as três arquiteturas (seus blocos e o método `forward`), as perdas Dice e CE+Dice, as
métricas (IoU, mIoU, Dice e pixel accuracy, calculadas a partir de uma matriz de confusão) e o laço
de treino. Recorremos a bibliotecas apenas para as primitivas `nn.*`, o download do dataset, o
otimizador e o *scheduler*, e os pesos pré-treinados, estes últimos usados somente na ablação de
*transfer learning*. A distinção completa entre o que é próprio e o que é de biblioteca está
documentada no `PLANEJAMENTO.md`.

---

## 4. Experimentos

### 4.1 Comparativo de arquiteturas
As três arquiteturas foram treinadas sob exatamente o mesmo protocolo, com perda CrossEntropy, 30
épocas, sem *augmentation* e com `seed=42`. A comparação considera mIoU, Dice e pixel accuracy, além
do número de parâmetros e do tempo de treino de cada modelo.

### 4.2 Ablações na melhor arquitetura
A partir da melhor arquitetura do comparativo, variamos um fator por vez, sempre com 15 épocas e
`seed=42`. Os fatores avaliados foram a função de perda (CE, Dice e CE+Dice), o uso de data
augmentation (com e sem), a presença de Batch Normalization (com e sem, controlada pelo parâmetro
`use_bn` da U-Net) e o transfer learning (encoder ResNet-34 pré-treinado no ImageNet contra o mesmo
encoder treinado do zero).

---

## 5. Resultados

### 5.1 Comparativo de arquiteturas (validação, 30 épocas)

| Arquitetura | Params (M) | mIoU | Dice | Pixel acc | Tempo (s) |
|---|---|---|---|---|---|
| FCN-8s | 20,50 | 0,6788 | 0,7891 | 0,8731 | 710 |
| **U-Net** | 31,04 | **0,7282** | **0,8295** | **0,8961** | 1460 |
| DeepLabV3+ | 16,60 | 0,6720 | 0,7799 | 0,8640 | 583 |

![Comparativo de arquiteturas](imagens/comparativo_arquiteturas.png)

A U-Net venceu em todas as métricas, atingindo um mIoU de 0,7282, e sua curva de validação ficou
consistentemente acima das demais a partir da oitava época, aproximadamente. A DeepLabV3+ convergiu
rápido no início, mas a sua perda de validação voltou a subir após a décima segunda época, um sinal
claro de *overfitting*, e terminou ligeiramente atrás da FCN-8s em mIoU, ainda que seja o modelo com
menos parâmetros (16,60 milhões) e o mais rápido de treinar (583 segundos). A FCN-8s, por sua vez,
convergiu de forma mais lenta e ruidosa, com uma queda visível de mIoU na décima nona época, mas
acabou estabilizando em 0,6788. O custo de treino acompanhou a profundidade efetiva de cada rede: a
U-Net foi a mais cara, com 1460 segundos, cerca do dobro das outras duas.

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

Em relação à função de perda, a Dice (0,7242) superou tanto a combinação CE+Dice (0,7186) quanto a
entropia cruzada pura (0,7139), o que indica que otimizar diretamente a sobreposição beneficiou a
métrica de IoU. O data augmentation trouxe um ganho pequeno, porém positivo (0,7114 contra 0,7070);
com apenas 15 épocas e um dataset relativamente pequeno, o efeito regularizador ainda é modesto. O
Batch Normalization também ajudou, acelerando e melhorando o treino (0,7040 com BN contra 0,6880 sem
BN).

O fator de maior impacto, no entanto, foi o transfer learning. O encoder ResNet-34 pré-treinado no
ImageNet alcançou 0,7507 em apenas 15 épocas, superando inclusive a U-Net treinada do zero por 30
épocas (0,7282). Como o mesmo encoder ResNet-34 treinado do zero, sem pré-treino, ficou em 0,6875,
fica claro que o ganho vem do pré-treino em si, e não da arquitetura do encoder.

### 5.3 Avaliação final no conjunto de teste

A melhor arquitetura do comparativo, a U-Net, foi avaliada com os seus melhores pesos (selecionados
pelo mIoU de validação) no conjunto de teste oficial, composto por 3.669 imagens nunca usadas em
treino ou validação.

| Métrica | Valor |
|---|---|
| mIoU | 0,7408 |
| Dice | 0,8395 |
| Pixel accuracy | 0,9010 |

Olhando a IoU por classe, fica evidente o contraste entre elas. O fundo é a classe mais fácil
(0,8995) e o animal também é bem segmentado (0,8158), mas a borda é, de longe, a mais difícil
(0,5071), e é justamente ela que puxa o mIoU para baixo.

| Classe | IoU |
|---|---|
| animal | 0,8158 |
| fundo | 0,8995 |
| **borda** | **0,5071** |

Esse diagnóstico é reforçado pela matriz de confusão em nível de pixel, normalizada por classe
verdadeira.

![Matriz de confusão](imagens/matriz_confusao.png)

O padrão é coerente com as IoU por classe. O fundo é acertado em 95% dos seus pixels e o animal em
91%, ambos com pouca dispersão para fora da diagonal. A borda, por sua vez, é acertada em apenas 63%
dos casos, sendo confundida de forma quase equilibrada com animal (18%) e com fundo (19%). Esse
comportamento é esperado, já que a borda é uma faixa estreita de transição entre as outras duas
classes.

Para entender o desempenho além das métricas agregadas, calculamos também a distribuição do mIoU por
imagem sobre as 3.669 imagens do teste.

| Estatística (mIoU por imagem) | Valor |
|---|---|
| Média | 0,7356 |
| Mediana | 0,7625 |
| Desvio padrão | 0,1118 |
| Q25 a Q75 | 0,6859 a 0,8145 |
| Mínimo a Máximo | 0,0083 a 0,9098 |
| Imagens com mIoU > 0,7 | 2.613 (71,2%) |
| Imagens com mIoU > 0,8 | 1.198 (32,7%) |
| Imagens com mIoU < 0,5 | 153 (4,2%) |

A distribuição mostra um desempenho robusto, com a maioria das imagens (71,2%) acima de 0,7 de mIoU e
apenas 4,2% delas abaixo de 0,5, o que indica poucos casos de falha grave. A figura abaixo apresenta
alguns exemplos de predições, com a imagem, o *ground-truth* e a predição lado a lado.

![Predições no teste](imagens/predicoes_teste.png)

As predições reproduzem bem a silhueta do animal e a região de fundo. Os erros se concentram na faixa
de borda e em cenas com camuflagem ou fundo complexo, como na quarta linha da figura, o que é
coerente tanto com a baixa IoU da classe borda quanto com os casos de mIoU mínimo observados na
distribuição por imagem.

---

## 6. Discussão

### Pontos fortes da solução
O comparativo conduzido é justo e reprodutível, pois as três arquiteturas foram treinadas sob o
mesmo protocolo e a mesma semente. Nesse cenário, a U-Net se mostrou a melhor
escolha treinada do zero para um dataset de tamanho reduzido, o que se explica pelas suas *skip
connections* densas, capazes de recuperar bem os detalhes espaciais. O transfer learning foi a
alavanca mais eficaz de todas: o encoder ResNet-34 pré-treinado atingiu o melhor mIoU (0,7507) na
metade das épocas, confirmando o valor das *features* pré-treinadas mesmo em um domínio (animais)
diferente do ImageNet. No conjunto, o desempenho global foi bom, com pixel accuracy de 0,9010 e 71,2%
das imagens do teste acima de 0,7 de mIoU.

### Limitações do modelo
A principal limitação é a classe borda, que se mostrou o gargalo do modelo com IoU de apenas 0,5071,
em razão dos poucos pixels que ocupa e da ambiguidade intrínseca do contorno. As perdas baseadas em
Dice ajudaram, mas não resolveram esse problema. A resolução de 128×128 também limita o detalhamento
dos contornos, e resoluções maiores tendem a beneficiar justamente a classe borda, ainda que ao custo
de mais tempo de treino. Por fim, a DeepLabV3+ treinada do zero apresentou *overfitting*, com perda de
validação crescente, o que reforça que arquiteturas com *backbone* do tipo ResNet são mais
dependentes de pré-treino e de uma quantidade maior de dados.

### Principais dificuldades encontradas
Uma dificuldade prática foi sincronizar as transformações geométricas entre a imagem e a máscara,
resolvida com a classe `JointTransform` e o uso de interpolação *nearest* na máscara. Outra foi lidar
com o Batch Normalization no ramo de *pooling* global do ASPP quando os batches são pequenos, o que
foi mitigado pelo uso de `drop_last=True` no treino e pelas estatísticas acumuladas durante a
avaliação. Houve ainda o custo computacional de treinar três arquiteturas por 30 épocas somado às
nove ablações, o que exigiu o uso de GPU no Google Colab.

### Possíveis melhorias futuras
Um caminho natural é combinar as duas melhores descobertas do trabalho, treinando uma U-Net com
encoder pré-treinado e perda Dice por mais épocas e em resolução 256×256. Também seria interessante
testar um *backbone* mais profundo, como a ResNet-50 com *atrous convolutions*, e adotar uma perda de
Dice generalizada (Sudre et al., 2017) ponderada por classe para reforçar especificamente a borda.
Por fim, um pós-processamento de contornos com CRF e um *augmentation* mais agressivo, acompanhado de
mais épocas de treino, poderiam refinar ainda mais os resultados.

---

## 7. Conclusão

Entre as arquiteturas treinadas do zero, a U-Net foi a melhor, alcançando no teste um mIoU de 0,7408,
Dice de 0,8395 e pixel accuracy de 0,9010. Já entre todas as configurações avaliadas nas ablações, a
de maior mIoU de validação foi a U-Net com encoder ResNet-34 pré-treinado, com 0,7507, o que aponta o
caminho para a melhor solução final.

Os principais aprendizados do trabalho podem ser resumidos em três pontos. Primeiro, em datasets
pequenos, as *skip connections* densas da U-Net superam arquiteturas com mais contexto, porém mais
dependentes de dados e de pré-treino, como a DeepLabV3+. Segundo, tanto o transfer learning quanto o
Batch Normalization trazem ganhos claros, e as perdas baseadas em Dice ajudam a métrica de IoU,
sobretudo nas classes raras. Terceiro, a classe borda é o principal fator limitante do desempenho e
se confirma como o alvo mais promissor para trabalhos futuros.

---

## 8. Referências

1. Long, Jonathan, Evan Shelhamer, and Trevor Darrell. "Fully convolutional networks for semantic segmentation." Proceedings of the IEEE conference on computer vision and pattern recognition. 2015.
2. Ronneberger, Olaf, Philipp Fischer, and Thomas Brox. "U-net: Convolutional networks for biomedical image segmentation." International Conference on Medical image computing and computer-assisted intervention. Cham: Springer international publishing, 2015.
3. Chen, Liang-Chieh, et al. "Encoder-decoder with atrous separable convolution for semantic image segmentation." Proceedings of the European conference on computer vision (ECCV). 2018.
4. Sudre, Carole H., et al. "Generalised dice overlap as a deep learning loss function for highly unbalanced segmentations." International Workshop on Deep Learning in Medical Image Analysis. Cham: Springer International Publishing, 2017.
5. He, Kaiming, et al. "Deep residual learning for image recognition." Proceedings of the IEEE conference on computer vision and pattern recognition. 2016.

### Dataset
[Omkar M Parkhi and Andrea Vedaldi and Andrew Zisserman and C. V. Jawahar: The Oxford-IIIT Pet Dataset](https://www.robots.ox.ac.uk/~vgg/data/pets/)
