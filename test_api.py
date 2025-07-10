import requests
import os
import random
from collections import defaultdict

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
                'expected_health': expected_health
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

def get_test_images_by_species(n_images=5):
    """Coleta imagens de teste organizadas por espécie"""
    especies_config = {
        'Tomato': {
            'healthy': 'PlantVillage/Tomato_healthy',
            'unhealthy': [
                'PlantVillage/Tomato_Early_blight',
                'PlantVillage/Tomato_Late_blight',
                'PlantVillage/Tomato_Bacterial_spot',
                'PlantVillage/Tomato_Leaf_Mold'
            ]
        },
        'Potato': {
            'healthy': 'PlantVillage/Potato___healthy',
            'unhealthy': [
                'PlantVillage/Potato___Early_blight',
                'PlantVillage/Potato___Late_blight'
            ]
        },
        'Pepper': {
            'healthy': 'PlantVillage/Pepper__bell___healthy',
            'unhealthy': [
                'PlantVillage/Pepper__bell___Bacterial_spot'
            ]
        }
    }
    
    test_images = defaultdict(list)
    
    for especie, paths in especies_config.items():
        print(f"📂 Coletando imagens de {especie}...")
        
        # Imagens saudáveis
        if os.path.exists(paths['healthy']):
            healthy_images = [f for f in os.listdir(paths['healthy']) 
                            if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            if healthy_images:
                selected = random.sample(healthy_images, min(n_images, len(healthy_images)))
                for img in selected:
                    test_images[especie].append({
                        'path': os.path.join(paths['healthy'], img),
                        'expected_species': especie,
                        'expected_health': 'healthy'
                    })
        
        # Imagens doentes
        unhealthy_collected = 0
        for unhealthy_path in paths['unhealthy']:
            if unhealthy_collected >= n_images:
                break
            if os.path.exists(unhealthy_path):
                unhealthy_images = [f for f in os.listdir(unhealthy_path) 
                                  if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
                if unhealthy_images:
                    needed = min(n_images - unhealthy_collected, len(unhealthy_images))
                    selected = random.sample(unhealthy_images, needed)
                    for img in selected:
                        test_images[especie].append({
                            'path': os.path.join(unhealthy_path, img),
                            'expected_species': especie,
                            'expected_health': 'unhealthy'
                        })
                        unhealthy_collected += 1
        
        print(f"   ✅ {len(test_images[especie])} imagens coletadas para {especie}")
    
    return test_images

def run_comprehensive_test():
    """Executa teste abrangente com 5 imagens de cada espécie"""

    print(f"\n📋 Coletando 10 imagens de cada espécie...")
    random.seed(42)
    test_images = get_test_images_by_species(n_images=5)
    
    if not test_images:
        print("❌ Nenhuma imagem de teste encontrada")
        return
    
    print(f"\n🔬 EXECUTANDO TESTES...")
    all_results = []
    
    for especie, images in test_images.items():
        print(f"\n🌱 Testando {especie}:")
        species_results = []
        
        for i, img_info in enumerate(images, 1):
            print(f"   {i}/10 - {os.path.basename(img_info['path'])}...", end=" ")
            
            result = test_prediction(
                img_info['path'], 
                img_info['expected_species'], 
                img_info['expected_health']
            )
            
            if result:
                species_results.append(result)
                all_results.append(result)
                
                # Mostrar resultado resumido
                especie_ok = "✅" if result.get('acerto_especie', False) else "❌"
                saude_ok = "✅" if result.get('acerto_saude', False) else "❌"
                print(f"{especie_ok} {result['especie_pred']} | {saude_ok} {result['saude_pred']} (conf: {result['confianca_final']:.3f})")
            else:
                print("❌ ERRO")
    
    print(f"\n📊 ANÁLISE DOS RESULTADOS:")
    
    if all_results:
        # Métricas gerais
        total_tests = len(all_results)
        acertos_especie = sum(1 for r in all_results if r.get('acerto_especie', False))
        acertos_saude = sum(1 for r in all_results if r.get('acerto_saude', False))
        pipeline_sucessos = sum(1 for r in all_results if r.get('pipeline_sucesso', False))
        
        print(f"📈 MÉTRICAS GERAIS:")
        print(f"   Total de testes: {total_tests}")
        print(f"   Acurácia espécie: {acertos_especie}/{total_tests} ({acertos_especie/total_tests*100:.1f}%)")
        print(f"   Acurácia saúde: {acertos_saude}/{total_tests} ({acertos_saude/total_tests*100:.1f}%)")
        print(f"   Pipeline sucesso: {pipeline_sucessos}/{total_tests} ({pipeline_sucessos/total_tests*100:.1f}%)")
        
        # Métricas por espécie
        print(f"\n🔬 MÉTRICAS POR ESPÉCIE:")
        especies_stats = defaultdict(lambda: {'total': 0, 'especie_ok': 0, 'saude_ok': 0})
        
        for result in all_results:
            esp = result['expected_species']
            especies_stats[esp]['total'] += 1
            if result.get('acerto_especie', False):
                especies_stats[esp]['especie_ok'] += 1
            if result.get('acerto_saude', False):
                especies_stats[esp]['saude_ok'] += 1
        
        for especie, stats in especies_stats.items():
            esp_acc = stats['especie_ok'] / stats['total'] * 100
            sau_acc = stats['saude_ok'] / stats['total'] * 100
            print(f"   {especie}: Espécie {esp_acc:.1f}% | Saúde {sau_acc:.1f}%")
        
        # Confiança média
        conf_media = sum(r['confianca_final'] for r in all_results) / len(all_results)
        print(f"\n🎯 Confiança média do sistema: {conf_media:.3f}")

def main():
    """Função principal"""
    run_comprehensive_test()

if __name__ == "__main__":
    main() 