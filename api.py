from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import pickle
import os
from PIL import Image
import io
from typing import Dict, Any
from contextlib import asynccontextmanager

# Variáveis globais para os modelos
modelo_especies = None
encoder_especies = None
modelos_especialistas = {}

def carregar_modelos():
    """Carrega todos os modelos necessários"""
    global modelo_especies, encoder_especies, modelos_especialistas
    
    try:
        # Carregar modelo de espécies
        print("📂 Carregando modelo de espécies...")
        modelo_especies = load_model('modelos_salvos/melhor_modelo_especies_final_otimizado.h5')
        
        with open('datasets_processados/label_encoder_especies_modelo.pkl', 'rb') as f:
            encoder_especies = pickle.load(f)
        
        print(f"✅ Modelo de espécies carregado: {encoder_especies.classes_}")
        
        # Carregar modelos especialistas
        print("📂 Carregando modelos especialistas...")
        for especie in ['tomato', 'potato', 'pepper']:
            modelo_path = f'modelos_salvos/especialistas/especialista_{especie}_binario_final.h5'
            if os.path.exists(modelo_path):
                modelos_especialistas[especie] = load_model(modelo_path)
                print(f"✅ Modelo {especie} carregado")
            else:
                print(f"⚠️ Modelo {especie} não encontrado")
        
        print("🎯 Todos os modelos carregados com sucesso!")
        
    except Exception as e:
        print(f"❌ Erro ao carregar modelos: {e}")
        raise e

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Gerencia o ciclo de vida da aplicação"""
    carregar_modelos()
    yield


app = FastAPI(
    title="Plant Disease Detection API",
    description="API para detecção de doenças em plantas usando pipeline hierárquico",
    version="1.0.0",
    lifespan=lifespan
)

def preprocessar_imagem(img_bytes: bytes, target_size=(224, 224)):
    """Preprocessa imagem para os modelos"""
    try:
        # Converter bytes para PIL Image
        img = Image.open(io.BytesIO(img_bytes))
        
        # Converter para RGB se necessário
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        # Redimensionar
        img = img.resize(target_size)
        
        # Converter para array numpy
        img_array = np.array(img)
        
        # Normalizar
        img_array = img_array / 255.0
        
        # Adicionar dimensão batch
        img_array = np.expand_dims(img_array, axis=0)
        
        return img_array
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Erro ao processar imagem: {str(e)}")

def pipeline_hierarquico(img_array: np.ndarray) -> Dict[str, Any]:
    """
    Pipeline completo: Espécie → Saúde → Resultado Final
    """
    try:
        # PASSO 1: Classificar espécie
        pred_especies = modelo_especies.predict(img_array, verbose=0)
        indice_especie = np.argmax(pred_especies)
        especie_predita = encoder_especies.inverse_transform([indice_especie])[0]
        confianca_especie = float(np.max(pred_especies))
        
        # Mapear nome da espécie para o modelo especialista
        mapeamento_especies = {
            'Tomato': 'tomato',
            'Potato': 'potato', 
            'Pepper_bell': 'pepper'
        }
        
        especie_modelo = mapeamento_especies.get(especie_predita)
        
        # PASSO 2: Classificar saúde
        if especie_modelo and especie_modelo in modelos_especialistas:
            modelo_especialista = modelos_especialistas[especie_modelo]
            pred_saude = modelo_especialista.predict(img_array, verbose=0)[0][0]
            
            # Conversão binária: >0.5 = unhealthy, <=0.5 = healthy
            if pred_saude > 0.5:
                saude_predita = 'unhealthy'
                confianca_saude = float(pred_saude)
            else:
                saude_predita = 'healthy'
                confianca_saude = float(1 - pred_saude)
                
            # Resultado final combinado
            resultado_final = f"{especie_predita}_{saude_predita}"
            confianca_final = confianca_especie * confianca_saude
            pipeline_sucesso = True
            
        else:
            # Modelo especialista não disponível
            saude_predita = 'unknown'
            confianca_saude = 0.0
            resultado_final = f"{especie_predita}_unknown"
            confianca_final = confianca_especie
            pipeline_sucesso = False
        
        return {
            'especie': {
                'nome': especie_predita,
                'confianca': confianca_especie
            },
            'saude': {
                'status': saude_predita,
                'confianca': confianca_saude
            },
            'resultado_final': {
                'classificacao': resultado_final,
                'confianca': confianca_final
            },
            'pipeline_sucesso': pipeline_sucesso
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro no pipeline: {str(e)}")


# Endpoint principal de predição
@app.post("/predict")
async def predict_plant_disease(file: UploadFile = File(...)):
    """
    Endpoint principal para classificação de doenças em plantas
    
    Recebe uma imagem e retorna:
    - Espécie da planta
    - Status de saúde (healthy/unhealthy)
    - Confiança das predições
    """
    
    # Validar tipo de arquivo
    if not file.content_type.startswith('image/'):
        raise HTTPException(
            status_code=400, 
            detail="Arquivo deve ser uma imagem (JPEG, PNG, etc.)"
        )
    
    try:
        # Ler bytes da imagem
        img_bytes = await file.read()
        
        # Preprocessar imagem
        img_array = preprocessar_imagem(img_bytes)
        
        # Executar pipeline hierárquico
        resultado = pipeline_hierarquico(img_array)
        
        return JSONResponse(content=resultado)
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro interno: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)