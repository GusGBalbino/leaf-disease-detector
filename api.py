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
sistema_threshold_inteligente = {}

def threshold_inteligente(probabilidade, especie):
    """
    Threshold inteligente baseado na confiança da predição
    Prioriza detecção de plantas doentes (recall Unhealthy)
    """
    
    # Configurações base por espécie (mais baixos = mais sensível a unhealthy)
    thresholds_base = {
        'tomato': 0.55,    # Moderadamente sensível
        'potato': 0.45,    # Mais sensível (problema histórico)
        'pepper': 0.50     # Equilibrado
    }
    
    threshold_base = thresholds_base.get(especie.lower(), 0.5)
    
    # Ajuste dinâmico baseado na confiança
    if probabilidade >= 0.8:
        # Alta confiança - usar threshold mais baixo para capturar unhealthy
        threshold_ajustado = threshold_base * 0.7
    elif probabilidade >= 0.6:
        # Confiança média-alta - leve redução
        threshold_ajustado = threshold_base * 0.85
    elif probabilidade >= 0.4:
        # Confiança média - threshold base
        threshold_ajustado = threshold_base
    else:
        # Baixa confiança - ser mais conservador
        threshold_ajustado = threshold_base * 1.15
    
    # Garantir limites válidos
    threshold_ajustado = max(0.2, min(0.8, threshold_ajustado))
    
    return threshold_ajustado

def carregar_modelos():
    """Carrega todos os modelos necessários"""
    global modelo_especies, encoder_especies, modelos_especialistas, sistema_threshold_inteligente
    
    try:
        # Carregar modelo de espécies
        print("📂 Carregando modelo de espécies...")
        modelo_especies = load_model('modelos_salvos/melhor_modelo_especies_final_otimizado.h5')
        
        with open('datasets_processados/label_encoder_especies_modelo.pkl', 'rb') as f:
            encoder_especies = pickle.load(f)
        
        print(f"✅ Modelo de espécies carregado: {encoder_especies.classes_}")
        
        # Carregar modelos especialistas balanceados
        print("📂 Carregando modelos especialistas balanceados...")
        for especie in ['tomato', 'potato', 'pepper']:
            modelo_path = f'modelos_salvos/especialistas/especialista_{especie}_balanceado_final.h5'
            if os.path.exists(modelo_path):
                modelos_especialistas[especie] = load_model(modelo_path)
                print(f"✅ Modelo {especie} balanceado carregado")
            else:
                print(f"⚠️ Modelo {especie} não encontrado em {modelo_path}")
        
        # Carregar sistema de threshold inteligente
        print("📂 Carregando sistema de threshold inteligente...")
        sistema_path = 'modelos_salvos/especialistas/sistema_threshold_inteligente.pkl'
        if os.path.exists(sistema_path):
            with open(sistema_path, 'rb') as f:
                sistema_threshold_inteligente = pickle.load(f)
            print(f"✅ Sistema threshold inteligente carregado: {list(sistema_threshold_inteligente.keys())}")
        else:
            print("⚠️ Sistema threshold inteligente não encontrado, usando sistema padrão")
            sistema_threshold_inteligente = {}
        
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
    description="API para detecção de doenças em plantas usando pipeline hierárquico com sistema de threshold inteligente",
    version="3.0.0",
    lifespan=lifespan
)

# Endpoint de status
@app.get("/")
async def root():
    """Endpoint raiz com informações da API"""
    return {
        "message": "Plant Disease Detection API v3.0.0",
        "description": "API para detecção de doenças em plantas usando pipeline hierárquico com sistema de threshold inteligente",
        "features": [
            "🧠 Threshold inteligente baseado na confiança",
            "🎯 Foco na detecção de plantas doentes",
            "📈 Thresholds dinâmicos adaptativos",
            "🌱 Modelos especialistas balanceados"
        ],
        "endpoints": {
            "/predict": "POST - Classificar imagem de planta",
            "/status": "GET - Verificar status dos modelos",
            "/docs": "GET - Documentação interativa"
        }
    }

@app.get("/status")
async def check_status():
    """Verifica o status dos modelos carregados"""
    return {
        "modelo_especies": {
            "carregado": modelo_especies is not None,
            "classes": encoder_especies.classes_.tolist() if encoder_especies else []
        },
        "modelos_especialistas": {
            "carregados": list(modelos_especialistas.keys()),
            "total": len(modelos_especialistas)
        },
        "sistema_threshold_inteligente": {
            "carregado": len(sistema_threshold_inteligente) > 0,
            "especies": list(sistema_threshold_inteligente.keys()),
            "ranges": {k: v.get('thresholds_range', 'N/A') for k, v in sistema_threshold_inteligente.items()}
        }
    }

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
    Usa threshold inteligente baseado na confiança para maximizar detecção de doenças
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
        
        # PASSO 2: Classificar saúde com threshold inteligente
        if especie_modelo and especie_modelo in modelos_especialistas:
            modelo_especialista = modelos_especialistas[especie_modelo]
            pred_saude = modelo_especialista.predict(img_array, verbose=0)[0][0]
            
            # Aplicar threshold inteligente baseado na confiança
            threshold_dinamico = threshold_inteligente(pred_saude, especie_modelo)
            
            # Aplicar threshold inteligente
            if pred_saude > threshold_dinamico:
                saude_predita = 'unhealthy'
                confianca_saude = float(pred_saude)
            else:
                saude_predita = 'healthy'
                confianca_saude = float(1 - pred_saude)
                
            # Resultado final combinado
            resultado_final = f"{especie_predita}_{saude_predita}"
            confianca_final = confianca_especie * confianca_saude
            pipeline_sucesso = True
            
            # Informações adicionais para debug
            info_threshold = {
                'threshold_usado': threshold_dinamico,
                'probabilidade_bruta': float(pred_saude),
                'logica_aplicada': f"Confiança {pred_saude:.3f} → Threshold {threshold_dinamico:.3f}",
                'decisao': f"pred_saude ({pred_saude:.3f}) > threshold ({threshold_dinamico:.3f}) = {pred_saude > threshold_dinamico}",
                'sistema': 'threshold_inteligente'
            }
            
        else:
            # Modelo especialista não disponível
            saude_predita = 'unknown'
            confianca_saude = 0.0
            resultado_final = f"{especie_predita}_unknown"
            confianca_final = confianca_especie
            pipeline_sucesso = False
            info_threshold = {'erro': 'Modelo especialista não disponível'}
        
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
            'pipeline_sucesso': pipeline_sucesso,
            'debug_info': info_threshold
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro no pipeline: {str(e)}")


# Endpoint principal de predição
@app.post("/predict")
async def predict_plant_disease(file: UploadFile = File(...)):
    """
    Endpoint principal para classificação de doenças em plantas
    
    🧠 **Sistema de Threshold Inteligente v3.0**
    
    Recebe uma imagem e retorna:
    - Espécie da planta
    - Status de saúde (healthy/unhealthy) usando threshold inteligente
    - Confiança das predições
    - Informações de debug sobre o threshold aplicado
    
    **Sistema Threshold Inteligente**:
    - 🎯 **Foco**: Maximizar detecção de plantas doentes
    - 🧠 **Lógica**: Thresholds dinâmicos baseados na confiança
    - 📊 **Tomato**: Base 0.55 → Dinâmico 0.39-0.63
    - 📊 **Potato**: Base 0.45 → Dinâmico 0.32-0.52
    - 📊 **Pepper**: Base 0.50 → Dinâmico 0.35-0.58
    
    **Funcionamento**:
    - Confiança ≥ 0.8: Threshold reduzido (mais sensível a doenças)
    - Confiança 0.6-0.8: Leve redução do threshold
    - Confiança 0.4-0.6: Threshold base
    - Confiança < 0.4: Threshold aumentado (mais conservador)
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
        
        # Validar tamanho da imagem
        if len(img_bytes) == 0:
            raise HTTPException(status_code=400, detail="Arquivo de imagem vazio")
        
        if len(img_bytes) > 10 * 1024 * 1024:  # 10MB
            raise HTTPException(status_code=400, detail="Arquivo muito grande. Máximo: 10MB")
        
        # Preprocessar imagem
        img_array = preprocessar_imagem(img_bytes)
        
        # Executar pipeline hierárquico
        resultado = pipeline_hierarquico(img_array)
        
        # Log do resultado para monitoramento
        print(f"🔍 Predição: {resultado['especie']['nome']} - {resultado['saude']['status']} "
              f"(Confiança: {resultado['resultado_final']['confianca']:.3f})")
        
        return JSONResponse(content=resultado)
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Erro não tratado: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Erro interno do servidor: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    print("🚀 Iniciando Plant Disease Detection API v2.0.0")
    print("📋 Recursos:")
    print("   - Modelos especialistas com balanceamento otimizado")
    print("   - Thresholds calibrados (Tomato: 0.70, Potato: 0.60, Pepper: 0.65)")
    print("   - Pipeline hierárquico aprimorado")
    print("   - Endpoints: / | /status | /predict | /docs")
    print("🌐 Acesse: http://localhost:8000/docs para documentação interativa")
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")