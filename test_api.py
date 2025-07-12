import requests
import os
import random
from collections import defaultdict
import time

API_URL = "http://localhost:8000"

def test_prediction(image_path, expected_species=None, expected_health=None):
    """Testa predição com uma imagem"""
    if not os.path.exists(image_path):
        print(f"❌ Imagem não encontrada: {image_path}")
        return None
    
    try:
        with open(image_path, 'rb') as f:
            files = {'file': (os.path.basename(image_path), f, 'image/jpeg')}
            response = requests.post(f"{API_URL}/predict", files=files)
        
        if response.status_code == 200:
            data = response.json()
            result = {
                'arquivo': os.path.basename(image_path),
                'especie_pred': data['especie']['nome'],
                'especie_conf': data['especie']['confianca'],
                'saude_pred': data['saude']['status'],
                'saude_conf': data['saude']['confianca'],
                'resultado_final': data['resultado_final']['classificacao'],
                'confianca_final': data['resultado_final']['confianca'],
                'pipeline_sucesso': data['pipeline_sucesso'],
                'expected_species': expected_species,
                'expected_health': expected_health,
                'debug_info': data.get('debug_info', {})
            }
            
            # Verificar acertos
            if expected_species:
                result['acerto_especie'] = expected_species.lower() in data['especie']['nome'].lower()
            if expected_health:
                result['acerto_saude'] = expected_health == data['saude']['status']
                
            return result
        else:
            print(f"   ❌ Erro na API: {response.status_code} - {response.text}")
            return None
            
    except Exception as e:
        print(f"❌ Erro na predição: {e}")
        return None

def coletar_amostras_grandes():
    """Coleta amostras maiores de cada categoria"""
    
    # Configuração das amostras
    amostras_por_categoria = 20  # Aumentar para amostra mais robusta
    
    categorias = {
        # Tomato
        'Tomato_healthy': ('tomato', 'healthy'),
        'Tomato_Early_blight': ('tomato', 'unhealthy'),
        'Tomato_Late_blight': ('tomato', 'unhealthy'),
        'Tomato_Leaf_Mold': ('tomato', 'unhealthy'),
        'Tomato_Septoria_leaf_spot': ('tomato', 'unhealthy'),
        'Tomato_Spider_mites_Two_spotted_spider_mite': ('tomato', 'unhealthy'),
        'Tomato__Target_Spot': ('tomato', 'unhealthy'),
        'Tomato__Tomato_mosaic_virus': ('tomato', 'unhealthy'),
        'Tomato__Tomato_YellowLeaf__Curl_Virus': ('tomato', 'unhealthy'),
        'Tomato_Bacterial_spot': ('tomato', 'unhealthy'),
        
        # Potato
        'Potato___healthy': ('potato', 'healthy'),
        'Potato___Early_blight': ('potato', 'unhealthy'),
        'Potato___Late_blight': ('potato', 'unhealthy'),
        
        # Pepper
        'Pepper__bell___healthy': ('pepper', 'healthy'),
        'Pepper__bell___Bacterial_spot': ('pepper', 'unhealthy'),
    }
    
    amostras = []
    base_path = "./PlantVillage"
    
    for categoria, (especie, saude) in categorias.items():
        categoria_path = os.path.join(base_path, categoria)
        
        if os.path.exists(categoria_path):
            # Listar todas as imagens da categoria
            imagens = [f for f in os.listdir(categoria_path) 
                            if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            
            if imagens:
                # Selecionar amostra aleatória
                amostra_categoria = random.sample(
                    imagens, 
                    min(amostras_por_categoria, len(imagens))
                )
                
                for img in amostra_categoria:
                    amostras.append({
                        'path': os.path.join(categoria_path, img),
                        'especie_esperada': especie,
                        'saude_esperada': saude,
                        'categoria': categoria
                    })
                    
                print(f"✅ {categoria}: {len(amostra_categoria)} imagens coletadas")
            else:
                print(f"❌ {categoria}: Nenhuma imagem encontrada")
        else:
            print(f"❌ {categoria}: Diretório não existe")
    
    return amostras

def verificar_api_status():
    """Verifica se a API está rodando e mostra informações"""
    try:
        response = requests.get(f'{API_URL}/status')
        if response.status_code == 200:
            status_data = response.json()
            print(f"📊 API Status: {status_data.get('versao', 'N/A')}")
            return True
        else:
            print("❌ API não está respondendo corretamente")
            return False
    except Exception as e:
        print(f"❌ Erro ao conectar com API: {e}")
        return False

def run_comprehensive_test():
    """Executa teste abrangente com muitas imagens de cada categoria"""

    print("🔍 TESTE ABRANGENTE - API DE DETECÇÃO DE DOENÇAS")
    print("=" * 60)
    
    # Verificar status da API
    if not verificar_api_status():
        print("❌ API não está rodando. Inicie com: python api.py")
        return
    
    print()
    
    # Coletar amostras
    print("📋 Coletando amostras...")
    amostras = coletar_amostras_grandes()
    
    if not amostras:
        print("❌ Nenhuma amostra coletada")
        return
    
    print(f"📊 Total de amostras coletadas: {len(amostras)}")
    print()
    
    # Inicializar contadores
    resultados = {
        'total': 0,
        'corretos_especie': 0,
        'corretos_saude': 0,
        'corretos_ambos': 0,
        'por_especie': defaultdict(lambda: {'total': 0, 'corretos_especie': 0, 'corretos_saude': 0, 'corretos_ambos': 0}),
        'por_saude': defaultdict(lambda: {'total': 0, 'corretos': 0}),
        'detalhes': []
    }
    
    # Testar cada amostra
    print("🧪 Testando amostras...")
    for i, amostra in enumerate(amostras):
        print(f"Testando {i+1}/{len(amostras)}: {os.path.basename(amostra['path'])}", end="... ")
            
        result = test_prediction(
            amostra['path'], 
            amostra['especie_esperada'], 
            amostra['saude_esperada']
            )
            
        if result:
            # Verificar acertos
            especie_correta = result.get('acerto_especie', False)
            saude_correta = result.get('acerto_saude', False)
            ambos_corretos = especie_correta and saude_correta
            
            # Atualizar contadores
            resultados['total'] += 1
            if especie_correta:
                resultados['corretos_especie'] += 1
            if saude_correta:
                resultados['corretos_saude'] += 1
            if ambos_corretos:
                resultados['corretos_ambos'] += 1
            
            # Por espécie
            esp = amostra['especie_esperada']
            resultados['por_especie'][esp]['total'] += 1
            if especie_correta:
                resultados['por_especie'][esp]['corretos_especie'] += 1
            if saude_correta:
                resultados['por_especie'][esp]['corretos_saude'] += 1
            if ambos_corretos:
                resultados['por_especie'][esp]['corretos_ambos'] += 1
            
            # Por saúde
            saude = amostra['saude_esperada']
            resultados['por_saude'][saude]['total'] += 1
            if saude_correta:
                resultados['por_saude'][saude]['corretos'] += 1
            
            # Detalhes
            resultados['detalhes'].append({
                'arquivo': os.path.basename(amostra['path']),
                'categoria': amostra['categoria'],
                'especie_esperada': amostra['especie_esperada'],
                'especie_pred': result['especie_pred'],
                'saude_esperada': amostra['saude_esperada'],
                'saude_pred': result['saude_pred'],
                'especie_correta': especie_correta,
                'saude_correta': saude_correta,
                'ambos_corretos': ambos_corretos,
                'probabilidade_bruta': result.get('debug_info', {}).get('probabilidade_bruta', 0)
            })
            
            status = "✅" if ambos_corretos else "❌"
            print(f"{status} E:{result['especie_pred']} S:{result['saude_pred']}")
        else:
            print("❌ ERRO")
    
        # Pequena pausa para não sobrecarregar a API
        time.sleep(0.1)
    
    # Calcular e exibir resultados
    exibir_resultados(resultados)

def exibir_resultados(resultados):
    """Exibe resultados detalhados"""
    
    print("\n" + "="*60)
    print("📊 RESULTADOS FINAIS")
    print("="*60)
    
    total = resultados['total']
    if total == 0:
        print("❌ Nenhum teste realizado")
        return
    
    # Resultados gerais
    acc_especie = (resultados['corretos_especie'] / total) * 100
    acc_saude = (resultados['corretos_saude'] / total) * 100
    acc_ambos = (resultados['corretos_ambos'] / total) * 100
        
    print(f"📈 MÉTRICAS GERAIS:")
    print(f"   Total de testes: {total}")
    print(f"   Acurácia Espécie: {acc_especie:.1f}% ({resultados['corretos_especie']}/{total})")
    print(f"   Acurácia Saúde: {acc_saude:.1f}% ({resultados['corretos_saude']}/{total})")
    print(f"   Acurácia Completa: {acc_ambos:.1f}% ({resultados['corretos_ambos']}/{total})")
    print()
        
    # Por espécie
    print("🌱 MÉTRICAS POR ESPÉCIE:")
    for especie, stats in resultados['por_especie'].items():
        if stats['total'] > 0:
            acc_esp = (stats['corretos_especie'] / stats['total']) * 100
            acc_sau = (stats['corretos_saude'] / stats['total']) * 100
            acc_amb = (stats['corretos_ambos'] / stats['total']) * 100
            
            print(f"   {especie.upper()}:")
            print(f"     Espécie: {acc_esp:.1f}% ({stats['corretos_especie']}/{stats['total']})")
            print(f"     Saúde: {acc_sau:.1f}% ({stats['corretos_saude']}/{stats['total']})")
            print(f"     Completa: {acc_amb:.1f}% ({stats['corretos_ambos']}/{stats['total']})")
    print()
    
    # Por saúde
    print("🏥 MÉTRICAS POR SAÚDE:")
    for saude, stats in resultados['por_saude'].items():
        if stats['total'] > 0:
            acc = (stats['corretos'] / stats['total']) * 100
            print(f"   {saude.upper()}: {acc:.1f}% ({stats['corretos']}/{stats['total']})")
    print()
    
    # Análise de erros
    print("🔍 ANÁLISE DE ERROS:")
    erros_especie = [d for d in resultados['detalhes'] if not d['especie_correta']]
    erros_saude = [d for d in resultados['detalhes'] if not d['saude_correta']]
    
    if erros_especie:
        print(f"   Erros de Espécie ({len(erros_especie)}):")
        for erro in erros_especie[:5]:  # Mostrar apenas os primeiros 5
            print(f"     • {erro['arquivo']}: {erro['especie_esperada']} → {erro['especie_pred']}")
        if len(erros_especie) > 5:
            print(f"     ... e mais {len(erros_especie) - 5} erros")
    
    if erros_saude:
        print(f"   Erros de Saúde ({len(erros_saude)}):")
        for erro in erros_saude[:5]:  # Mostrar apenas os primeiros 5
            prob = erro['probabilidade_bruta']
            print(f"     • {erro['arquivo']}: {erro['saude_esperada']} → {erro['saude_pred']} (prob: {prob:.3f})")
        if len(erros_saude) > 5:
            print(f"     ... e mais {len(erros_saude) - 5} erros")
    


def main():
    """Função principal"""
    # Definir seed para reprodutibilidade
    random.seed(42)
    
    run_comprehensive_test()

if __name__ == "__main__":
    main() 