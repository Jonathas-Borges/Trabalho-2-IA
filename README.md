# Trabalho-IA

Trabalho da disciplina de Inteligência Artificial da Universidade Federal de Mato Grosso do Sul

Este projeto consiste no desenvolvimento e treinamento de um modelo baseado na arquitetura DistilBERT para a classificação de textos pertencentes às classes "Poema" e "Crítica" (Review). O modelo foi integrado a um chatbot no Telegram, que permite ao usuário enviar um texto e receber a classificação correspondente em tempo real.

Além disso, o processo de treinamento foi realizado utilizando técnicas de divisão de dados em conjuntos de treino, validação e teste, garantindo a avaliação consistente do modelo. O chatbot aproveita a robustez da arquitetura pré-treinada DistilBERT para processar os textos enviados e gerar predições rápidas e precisas.

Funcionalidades do Projeto ->

## Classificação de Texto:

- Permite classificar textos como "Poema" ou "Crítica".

## Chatbot no Telegram:

- Interface acessível para interação com o modelo treinado.
- Usuários podem enviar textos diretamente pelo Telegram para obter a classificação.

## Treinamento do Modelo:

- Modelo treinado com um dataset específico dividido em 70% para treino, 15% para validação e 15% para teste.
- Utiliza a arquitetura pré-treinada DistilBERT adaptada para a tarefa de classificação binária.

## Resultados:

- Alta precisão no conjunto de teste, com métricas de avaliação que comprovam a eficiência do modelo.

## Requisitos:
- Python 3.10 ou superior
- Bibliotecas: torch, transformers, sklearn, telegram

Este trabalho demonstra a aplicação prática de técnicas de processamento de linguagem natural (NLP) e aprendizado profundo em um contexto acessível e interativo.
