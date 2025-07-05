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
  "detalhes": {
    "modelo_especie_usado": true,
    "modelo_especialista_usado": true,
    "especialista_disponivel": true
  },
  "arquivo": {
    "nome": "imagem.jpg",
    "tamanho": 45678,
    "tipo": "image/jpeg"
  }
}
```
