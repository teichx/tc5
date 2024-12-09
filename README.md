# Instalação

Para executar efetivamente o código do nosso trabalho, é essencial configurar o ambiente Python com as bibliotecas necessárias. Certifique-se de ter o Python instalado e utilize o seguinte comando no terminal para instalar as dependências requeridas:

```bash
pip install numpy Pillow scikit-learn opencv-python matplotlib mediapipe tensor-flow
```

Essas bibliotecas fornecem funcionalidades cruciais para a implementação dos algoritmos, como manipulação eficiente de arrays, processamento de imagens, modelos de aprendizado profundo e visualização de resultados. Após a instalação, você está pronto para executar os scripts principais.

# PROBLEMA

O código aborda o desafio de reconhecer e interpretar gestos em Libras (Língua Brasileira de Sinais) em imagens capturadas em tempo real. A segmentação precisa dos gestos em relação ao plano de fundo é crucial para aplicações que visam melhorar a interação entre pessoas com deficiência auditiva e o ambiente digital.

- Inicialização da Máscara: O algoritmo inicia estabelecendo uma máscara básica para identificar as regiões aproximadas onde os gestos em Libras e o plano de fundo são detectados na imagem.
- Convergência: O processo continua até que a segmentação alcance um estado onde os gestos em Libras estejam claramente separados do plano de fundo na imagem capturada.

Este processo de segmentação é fundamental para garantir a precisão e a confiabilidade dos sistemas de interpretação de gestos, promovendo uma interação mais fluida e inclusiva em ambientes digitais para usuários fluentes em Libras.

# ALGORITMO

O código é dividido em três partes principais, cada uma utilizando algoritmos específicos para resolver diferentes problemas: à captura de iamgens para alimentar o banco, o treinamento de um modelo LSTM para reconhecimento de gestos e a detecção de gestos em tempo real usando MediaPipe.

## Captura de Dados de Gestos com MediaPipe

Essa parte do código é responsável por capturar dados de gestos utilizando a câmera e o MediaPipe para detectar os pontos-chave das mãos. Ele salva os dados de pontos-chave em arquivos .npy para serem usados posteriormente no treinamento do modelo.

Funcionamento do Algoritmo:

- Inicialização do MediaPipe: Configura a solução Holistic do MediaPipe, que inclui a detecção de mãos.
- Captura de Vídeo: Usa a câmera do dispositivo para capturar frames de vídeo.
- Detecção de Pontos-Chave: Cada frame capturado é processado pelo modelo MediaPipe para detectar os pontos-chave das mãos.
- Renderização de Pontos-Chave: Os pontos-chave detectados são desenhados no frame para visualização.
- Extração e Salvamento de Pontos-Chave: Os pontos-chave das mãos são extraídos e salvos em arquivos .npy para serem usados no treinamento do modelo.

## Treinamento de Modelo LSTM para Reconhecimento de Gestos.

Essa código treina um modelo LSTM (Long Short-Term Memory) usando os dados de pontos-chave capturados para reconhecer gestos em Libras.
Funcionamento do Algoritmo:

- Carregamento dos Dados: Os dados de gestos são carregados dos arquivos .npy.
- Pré-processamento dos Dados: Os dados são convertidos para arrays numpy e as labels são codificadas em one-hot encoding.
- Divisão dos Dados: Os dados são divididos em conjuntos de treinamento e teste.
- Construção do Modelo LSTM: O modelo é construído com camadas LSTM e Dropout para prevenir overfitting.
- Treinamento do Modelo: O modelo é treinado utilizando o otimizador Adam e a função de perda categórica.
- Avaliação e Salvamento do Modelo: O modelo é avaliado no conjunto de teste e os resultados são plotados. Finalmente, o modelo é salvo.

## Detecção de Gestos em Tempo Real

Este código utiliza um modelo LSTM previamente treinado para detectar gestos em tempo real usando a câmera e o MediaPipe para processar os pontos-chave das mãos.

Funcionamento do Algoritmo:

- Inicialização do MediaPipe: Configura a solução Holistic do MediaPipe.
- Carregamento do Modelo Treinado: O modelo LSTM treinado é carregado para fazer as predições.
- Captura de Vídeo: Usa a câmera do dispositivo para capturar frames de vídeo.
- Processamento de Imagem: Cada frame capturado é processado pelo MediaPipe para detectar os pontos-chave das mãos.
- Predição do Gesto: A sequência de pontos-chave é alimentada no modelo LSTM para prever o gesto realizado.
- Visualização dos Resultados: O resultado da predição é mostrado na tela.

# SOLUÇÃO

Este projeto visa desenvolver um sistema avançado de reconhecimento de gestos em Libras (Língua Brasileira de Sinais), utilizando uma combinação de Visão Computacional e Aprendizado Profundo. A solução é estruturada em duas etapas fundamentais: o treinamento de um modelo LSTM (Long Short-Term Memory) para classificação precisa dos gestos, e a
implementação de uma aplicação em tempo real capaz de detectar e interpretar esses gestos diretamente de uma câmera.

O modelo LSTM foi configurado para reconhecer um conjunto diversificado de gestos que incluem as 26 letras do alfabeto, além de gestos específicos como "mulher", "homem" e "certo". Essa abordagem não apenas permite a comunicação eficaz em Libras, mas também promove a inclusão digital ao facilitar a interação de pessoas com deficiência auditiva em am-
bientes tecnológicos.

Ao combinar técnicas avançadas de processamento de imagens do MediaPipe com os poderosos recursos de aprendizado profundo oferecidos pela rede LSTM, o sistema é capaz de identificar e interpretar gestos com alta precisão e em tempo real. Isso não apenas melhora a acessibilidade digital, mas também abre portas para aplicações inovadoras em áreas como assistência tecnológica e interfaces de usuário baseadas em gestos.

Este projeto representa um passo significativo em direção a um futuro mais inclusivo e tecnologicamente avançado, onde a comunicação através de gestos em Libras pode ser realizada
de forma eficiente e acessível.

O código implementa um sistema para captura de gestos em Libras usando a biblioteca MediaPipe para processamento de imagens em tempo real. Ele é relevante para o domínio espacial da visão computacional, envolvendo a detecção e a manipulação de pontos-chave das mãos em imagens.

## Manipulação de Pixels

O código utiliza a biblioteca OpenCV para manipulação de pixels. Cada frame de vídeo capturado pela câmera é processado para detectar os pontos-chave das mãos usando o Media-Pipe. Esses pontos são então utilizados para reconhecer e salvar os gestos em arquivos .npy, que representam as coordenadas espaciais dos pontos-chave das mãos.

## Operações no Espaço da Imagem

As operações no espaço da imagem são fundamentais para este código. Ele captura os frames de vídeo, processa cada frame para detectar os pontos-chave das mãos e, em seguida, utiliza essas informações para salvar os gestos em arquivos .npy. Esse processo todo ocorre no espaço da imagem, manipulando coordenadas e pixels para reconhecer e armazenar os gestos
capturados.

Este sistema de captura de gestos em Libras demonstra como a manipulação espacial de pixels e pontos-chave das mãos é essencial para a aplicação prática de reconhecimento de gestos em tempo real. Embora não diretamente relacionado ao algoritmo GrabCut mencionado anteriormente, ele compartilha o princípio de manipulação de dados no espaço da imagem para
atingir seus objetivos computacionais.

# RESULTADOS OBTIDOS

Com base nos valores obtidos após o treinamento do modelo:

- **Loss (Perda):** A perda média calculada durante o treinamento foi de aproximadamente
  0.69. Isso indica a média dos erros do modelo ao fazer previsões em relação às classes
  reais dos dados de teste.
- **Acurácia:** A acurácia do modelo foi de aproximadamente 80%. Isso significa que, em
  média, o modelo classificou corretamente 80% das amostras do conjunto de dados de
  teste.

## Interpretação dos Resultados:

Uma perda de 0.69 sugere que ainda há espaço para melhorar o modelo, visando uma correspondência mais precisa entre as previsões do modelo e os rótulos reais dos dados de teste.

Uma acurácia de 80% é considerada razoavelmente boa, mas pode variar dependendo do contexto da aplicação e da complexidade dos dados de entrada.

Estes resultados destacam a importância de continuar refinando o modelo para otimizar ainda mais seu desempenho.
