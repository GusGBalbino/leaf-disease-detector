# 🌱 Plant Disease Detection API

API simples em FastAPI para detecção de doenças em plantas usando pipeline hierárquico.

## 🚀 Como usar

### 1. Instalar dependências
```bash
pip install -r requirements_api.txt
```

### 2. Executar a API
```bash
python api.py
```

A API estará disponível em: `http://localhost:8000`

### 3. Testar a API
```bash
python test_api.py
```

### `POST /predict`
Endpoint principal para classificação de doenças

**Input:** Imagem (JPEG, PNG, etc.)

**Output:**
```json
{
  "especie": {
    "nome": "Tomato",
    "confianca": 0.968
  },
  "saude": {
    "status": "unhealthy",
    "confianca": 0.924
  },
  "resultado_final": {
    "classificacao": "Tomato_unhealthy",
    "confianca": 0.894
  },
  "pipeline_sucesso": true,
  "debug_info": {
    "threshold_usado": 0.75,
    "probabilidade_bruta": 0.924,
    "logica_aplicada": "Threshold científico fixo para tomato",
    "decisao": "pred_saude (0.924) > threshold (0.750) = true",
    "sistema": "threshold_cientifico_fixo"
  }
}
```
