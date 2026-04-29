# ArUco Air Hockey

Um jogo de **Air Hockey com visão computacional** usando **OpenCV + ArUco markers**, onde dois jogadores controlam seus rebatedores físicos com marcadores ArUco impressos e interagem com um disco virtual em tempo real.

A proposta deste projeto é transformar um exemplo clássico de detecção com ArUco em algo **visual, interativo e divertido**, tornando o aprendizado de visão computacional muito mais interessante para estudantes, makers e entusiastas.

---

## Visão Geral

Este projeto utiliza a câmera do computador para detectar marcadores **ArUco** em tempo real.  
Cada marcador funciona como um **rebatedor virtual** (stick), e o objetivo é rebater o disco (puck) e marcar pontos no gol adversário — como em uma mesa de Air Hockey.

O jogo foi pensado como uma forma prática e divertida de aprender:

- detecção de marcadores ArUco
- rastreamento em tempo real
- processamento de imagem com OpenCV
- colisão e física básica 2D
- interação entre objetos físicos e elementos virtuais

---

## Objetivo do Projeto

O principal objetivo deste projeto é servir como um **tutorial prático e interativo** para introduzir conceitos de:

- Visão Computacional
- Realidade Aumentada
- Rastreamento com marcadores
- Física de jogos em 2D
- Interação humano-computador

Em vez de aprender ArUco apenas detectando quadrados na tela, a proposta aqui é usar os marcadores em um contexto mais envolvente: **um jogo**.

Isso torna o aprendizado mais intuitivo e ajuda a despertar interesse em temas como:

- robótica
- computação gráfica
- sistemas interativos
- interfaces tangíveis
- prototipagem com visão computacional

---

## Como o jogo funciona

- A câmera captura a cena em tempo real.
- O OpenCV detecta os marcadores ArUco.
- Cada marcador detectado é associado a um jogador.
- A posição e rotação do marcador controlam um rebatedor virtual.
- Um disco virtual se move pelo campo com física simples.
- Quando o disco entra no gol adversário, um ponto é marcado.
- O primeiro jogador a atingir a pontuação de vitória vence.

---

## Tecnologias Utilizadas

Este projeto foi desenvolvido em **Python** e utiliza bibliotecas amplamente adotadas em visão computacional e computação gráfica.

### Linguagem
- Python 3.x

### Bibliotecas
- **OpenCV** (`cv2`)  
  Responsável por:
  - captura da câmera
  - processamento de imagem
  - detecção dos marcadores ArUco
  - renderização gráfica do jogo

- **NumPy** (`numpy`)  
  Utilizada para:
  - manipulação de arrays
  - operações matriciais
  - suporte ao processamento de imagem

- **math**  
  Utilizada para:
  - cálculos trigonométricos
  - vetores
  - ângulos
  - colisões

- **random**  
  Utilizada para:
  - gerar direção aleatória inicial do disco

- **time**  
  Utilizada para:
  - controle de tempo
  - delta time (`dt`)
  - atualização da física do jogo

- **pygame** *(opcional, em versões com áudio)*  
  Utilizada para:
  - reprodução de efeitos sonoros
  - sons de colisão
  - sons de gol
  - efeitos de vitória

---

## Bibliotecas Python utilizadas

```bash
pip install opencv-contrib-python numpy pygame