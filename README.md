# 🌱 Plant Disease Detection API

**Projeto de TCC - Detecção de Doenças em Plantas**

---

## 👥 Autores

- **Gustavo Gomes Balbino - RA: 52400106**
- **Gustavo Lopes Urio Fonseca - RA: 52400113**

**Orientador**: Fábio Oliveira Guimarães 
**Instituição**: Centro Universitário de Brasília (CEUB)  
**Curso**: Ciência de Dados e Machine Learning  
**Ano**: 2025

---

## 📋 Resumo do Projeto

Este projeto desenvolve uma **API inteligente para detecção de doenças em plantas** utilizando técnicas avançadas de Machine Learning e Deep Learning. O sistema implementa um **pipeline hierárquico** que combina classificação de espécies e análise de saúde das plantas, alcançando alta precisão na identificação de doenças.

### 🎯 Objetivos Principais

1. **Desenvolver um sistema de detecção automática** de doenças em plantas
2. **Implementar pipeline hierárquico** com modelos especializados
3. **Otimizar thresholds científicos** para máxima precisão
4. **Criar API robusta** para uso em produção
5. **Alcançar acurácia superior a 90%** na classificação

### 🌿 Espécies Suportadas

- **🍅 Tomate** (Tomato) - 10 doenças detectadas
- **🥔 Batata** (Potato) - 2 doenças detectadas  
- **🌶️ Pimentão** (Pepper) - 1 doença detectada

---

## 🚀 Tecnologias Utilizadas

### 🤖 Machine Learning
- **TensorFlow/Keras** - Framework principal
- **ResNet50** - Arquitetura base com Transfer Learning
- **Pipeline Hierárquico** - Classificação em duas etapas

### 🛠️ Desenvolvimento
- **FastAPI** - API REST moderna e rápida
- **Python 3.9+** - Linguagem principal
- **Jupyter Notebooks** - Desenvolvimento e experimentação

### 📊 Dados
- **PlantVillage Dataset** - 54,305 imagens
- **Data Augmentation** - Técnicas de balanceamento
- **Cross-Validation** - Validação robusta

---

## 🏗️ Arquitetura do Sistema

### 📈 Pipeline Hierárquico

```
1. Classificação de Espécie
   ↓
2. Modelo Especialista (Saúde)
   ↓  
3. Threshold Científico
   ↓
4. Resultado Final
```

### 🧠 Modelos Implementados

1. **Modelo de Espécies** - Classifica entre Tomato, Potato, Pepper
2. **Modelos Especialistas** - Detectam healthy/unhealthy para cada espécie
3. **Thresholds Científicos** - Otimizados para máxima precisão

---

## 📊 Resultados Alcançados

### 🎯 Performance Final
- **Acurácia Geral**: 93.8% (melhoria de +20.5 pontos)
- **Detecção de Doenças**: 98.3% (unhealthy plants)
- **Classificação de Espécies**: 85.6%
- **Thresholds Otimizados**:
  - 🍅 Tomato: 0.75 (F1=100%)
  - 🥔 Potato: 0.65 (F1=95.2%)
  - 🌶️ Pepper: 0.15 (F1=95.2%)

### 🔬 Inovações Científicas
- **Thresholds Científicos** - Baseados em análise de dados reais
- **Balanceamento Inteligente** - Técnicas avançadas de data augmentation
- **Pipeline Otimizado** - Sistema hierárquico eficiente

---

## 📁 Estrutura do Projeto

```
leaf-disease-detector/
├── 01_Preparacao_Dados.ipynb   # Preparação e organização dos dados
├── 02_Preprocessamento.ipynb   # Preprocessamento das imagens
├── 03_Modelo_Classificador_Especies.ipynb  # Modelo de espécies
├── 04_Modelos_Especialistas_Saude.ipynb    # Modelos especialistas
├── 05_Pipeline_Hierarquico_e_Avaliacao.ipynb # Avaliação final
├── 🚀 api.py                       # API principal
├── 🧪 test_api.py                  # Testes da API
├── 🛠️ utils.py                     # Utilitários e configurações
├── 📁 PlantVillage/                # Dataset original (54,305 imagens)
├── 📁 modelos_salvos/              # Modelos treinados
│   ├── melhor_modelo_especies_final_otimizado.h5
│   ├── especialistas/
│   │   ├── especialista_tomato_balanceado_final.h5
│   │   ├── especialista_potato_balanceado_final.h5
│   │   └── especialista_pepper_balanceado_final.h5
│   └── resumo_modelo_especies.json
├── 📁 datasets_processados/         # Dados processados
│   ├── dataset_especies.csv        # Dataset principal
│   ├── dataset_tomato.csv          # Dados do tomate
│   ├── dataset_potato.csv          # Dados da batata
│   ├── dataset_pepper_bell.csv     # Dados do pimentão
│   └── configuracoes.json          # Configurações do sistema
└── 📚 README.md                    # Documentação principal
```

---

## 🚀 Como Usar

### 1. Preparação dos Dados
```bash
# Carregue a pasta PlantVillage com todas as imagens
# Execute os notebooks na ordem:
# 1. 01_Preparacao_Dados.ipynb - Organização dos dados
# 2. 02_Preprocessamento.ipynb - Preprocessamento das imagens
```

### 2. Desenvolvimento dos Modelos
```bash
# Execute os notebooks de desenvolvimento:
# 3. 03_Modelo_Classificador_Especies.ipynb - Modelo de espécies
# 4. 04_Modelos_Especialistas_Saude.ipynb - Modelos especialistas
# 5. 05_Pipeline_Hierarquico_e_Avaliacao.ipynb - Avaliação final
```

### 3. Executar a API
```bash
pip install -r requirements.txt
python api.py
```

### 4. Testar o Sistema
```bash
python test_api.py
```

---

## 📚 Notebooks de Desenvolvimento

### 🔧 **Preparação e Processamento**
1. **`01_Preparacao_Dados.ipynb`** - Preparação e organização dos dados do PlantVillage
2. **`02_Preprocessamento.ipynb`** - Preprocessamento das imagens e balanceamento

### 🤖 **Desenvolvimento dos Modelos**
3. **`03_Modelo_Classificador_Especies.ipynb`** - Modelo de classificação de espécies
4. **`04_Modelos_Especialistas_Saude.ipynb`** - Modelos especialistas para saúde
5. **`05_Pipeline_Hierarquico_e_Avaliacao.ipynb`** - Avaliação completa do sistema

---

## 🌐 API Endpoints

- **`GET /`** - Informações da API
- **`GET /status`** - Status dos modelos
- **`POST /predict`** - Classificação de imagens
- **`GET /docs`** - Documentação interativa

---

## 🎓 Contribuições Acadêmicas

### 📖 Metodologia
- Desenvolvimento de pipeline hierárquico para classificação de plantas
- Otimização de thresholds baseada em análise científica
- Implementação de técnicas avançadas de balanceamento de dados

### 🔬 Inovações
- Sistema de thresholds científicos otimizados
- Pipeline de dois estágios para máxima precisão
- API robusta para uso em produção

---

## 📄 Licença

Este projeto foi desenvolvido como trabalho de conclusão de curso no Centro Universitário de Brasília (CEUB).

---
