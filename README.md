# Segmentação Automática de Piadas em Setup e Punch via Classificação de Tokens

## Objetivo

Desenvolver um modelo de inteligência artificial capaz de identificar automaticamente as partes **setup** e **punch** em piadas textuais em português brasileiro.

## Dataset

O conjunto de dados contém **431 piadas** previamente separadas em setup e punch, além da versão completa de cada piada. Os dados foram integrados a partir de dois arquivos originais e consolidados em um único dataset.

## Análise Exploratória

A análise revelou padrões claros na estrutura das piadas:

- O **setup** tem em média **120 caracteres**.
- O **punch** tem em média **46 caracteres**.
- A piada completa possui cerca de **168 caracteres**.
- A proporção média entre setup e punch é de **2,6:1**.
- **88,9% das piadas** começam com uma pergunta.

## Pré-processamento

O texto foi normalizado com:
- Conversão para letras minúsculas
- Remoção de acentos
- Eliminação de caracteres especiais
- Padronização de espaçamento

## Engenharia de Features

Foram extraídas características linguísticas como:
- Número de palavras
- Presença de pontuação específica
- Diversidade lexical
- Indicadores de diálogo e interrogação

## Modelo de Classificação

Foi treinado um modelo de **Regressão Logística** com vetores **TF-IDF** para distinguir entre setup e punch.

### Resultados

- **Acurácia:** 89,6%
- **F1-score médio:** 90%
- Alta precisão na identificação de punchlines (94%)

## Arquivos Gerados

- Dataset consolidado com piadas completas
- Versão final com todas as features extraídas

## Conclusão

O modelo demonstra alta capacidade de segmentar corretamente as partes estruturais de piadas brasileiras, com desempenho robusto mesmo em um modelo simples. A abordagem pode ser estendida com técnicas avançadas de NLP.

## Próximos Passos

- Aplicação de modelos baseados em transformers
- Segmentação automática em piadas não rotuladas
- Desenvolvimento de ferramenta interativa

---

**Autor:** Cleon  
**Área:** Inteligência Artificial  
**Licença:** MIT
