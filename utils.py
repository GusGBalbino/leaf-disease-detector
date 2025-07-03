import pandas as pd
import numpy as np
import json
import pickle
import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image


def carregar_configuracoes(caminho='datasets_processados/configuracoes.json'):
    """Carrega configurações do projeto"""
    with open(caminho, 'r') as f:
        return json.load(f)

def carregar_dataset_especies(caminho='datasets_processados/dataset_especies.csv'):
    """Carrega dataset de espécies dividido"""
    df = pd.read_csv(caminho)
    
    train_data = df[df['split'] == 'train']
    val_data = df[df['split'] == 'val']
    test_data = df[df['split'] == 'test']
    
    return {
        'train': {'X': train_data['caminho'].values, 'y': train_data['especie'].values},
        'val': {'X': val_data['caminho'].values, 'y': val_data['especie'].values},
        'test': {'X': test_data['caminho'].values, 'y': test_data['especie'].values}
    }

def carregar_dataset_especialista(especie, caminho_base='datasets_processados'):
    """Carrega dataset de um especialista específico"""
    caminho = f'{caminho_base}/dataset_{especie.lower()}.csv'
    df = pd.read_csv(caminho)
    
    train_data = df[df['split'] == 'train']
    val_data = df[df['split'] == 'val']
    test_data = df[df['split'] == 'test']
    
    return {
        'train': {'X': train_data['caminho'].values, 'y': train_data['classe'].values},
        'val': {'X': val_data['caminho'].values, 'y': val_data['classe'].values},
        'test': {'X': test_data['caminho'].values, 'y': test_data['classe'].values}
    }

def carregar_label_encoder(tipo, especie=None, caminho_base='datasets_processados'):
    """Carrega label encoder"""
    if tipo == 'especies':
        caminho = f'{caminho_base}/label_encoder_especies.pkl'
    else:
        caminho = f'{caminho_base}/label_encoder_{especie.lower()}.pkl'
    
    with open(caminho, 'rb') as f:
        return pickle.load(f)

def criar_geradores(dataset, config, augment_train=True):
    """Cria geradores de dados a partir do dataset carregado"""
    
    # Configurações de augmentation
    if augment_train:
        train_datagen = ImageDataGenerator(
            rescale=1.0/255.0,
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            fill_mode='nearest',
            brightness_range=[0.8, 1.2]
        )
    else:
        train_datagen = ImageDataGenerator(rescale=1.0/255.0)
    
    val_test_datagen = ImageDataGenerator(rescale=1.0/255.0)
    
    # Criar DataFrames
    train_df = pd.DataFrame({'filename': dataset['train']['X'], 'class': dataset['train']['y']})
    val_df = pd.DataFrame({'filename': dataset['val']['X'], 'class': dataset['val']['y']})
    test_df = pd.DataFrame({'filename': dataset['test']['X'], 'class': dataset['test']['y']})
    
    # Criar geradores
    train_gen = train_datagen.flow_from_dataframe(
        train_df, x_col='filename', y_col='class',
        target_size=(config['img_height'], config['img_width']),
        batch_size=config['batch_size'],
        class_mode='categorical', shuffle=True
    )
    
    val_gen = val_test_datagen.flow_from_dataframe(
        val_df, x_col='filename', y_col='class',
        target_size=(config['img_height'], config['img_width']),
        batch_size=config['batch_size'],
        class_mode='categorical', shuffle=False
    )
    
    test_gen = val_test_datagen.flow_from_dataframe(
        test_df, x_col='filename', y_col='class',
        target_size=(config['img_height'], config['img_width']),
        batch_size=config['batch_size'],
        class_mode='categorical', shuffle=False
    )
    
    return train_gen, val_gen, test_gen

def carregar_modelo_especies(caminho_modelo='modelos_salvos/melhor_modelo_especies_final.h5',
                           caminho_encoder='datasets_processados/label_encoder_especies_modelo.pkl'):
    """Carrega modelo de classificação de espécies e seu encoder"""
    
    modelo = load_model(caminho_modelo)
    
    with open(caminho_encoder, 'rb') as f:
        encoder = pickle.load(f)
    
    return modelo, encoder

def classificar_especie(modelo, encoder, imagem_preprocessada):
    """
    Classifica a espécie de uma imagem preprocessada
    
    Args:
        modelo: Modelo carregado
        encoder: Label encoder das espécies
        imagem_preprocessada: Imagem já preprocessada (224x224x3, normalizada)
    
    Returns:
        tuple: (especie_predita, probabilidades, confianca)
    """
    
    
    # Fazer predição
    predicoes = modelo.predict(imagem_preprocessada)
    
    # Obter índice da classe com maior probabilidade
    indice_predito = np.argmax(predicoes, axis=1)[0]
    
    # Converter índice para nome da classe
    especie_predita = encoder.inverse_transform([indice_predito])[0]
    
    # Obter probabilidades e confiança
    probabilidades = predicoes[0]
    confianca = np.max(probabilidades)
    
    return especie_predita, probabilidades, confianca

def preprocessar_imagem_para_especies(caminho_imagem, target_size=(224, 224)):
    """
    Preprocessa uma imagem para classificação de espécies
    
    Args:
        caminho_imagem: Caminho para a imagem
        target_size: Tamanho desejado (altura, largura)
    
    Returns:
        numpy.array: Imagem preprocessada pronta para predição
    """
    
    # Carregar e redimensionar imagem
    img = image.load_img(caminho_imagem, target_size=target_size)
    
    # Converter para array
    img_array = image.img_to_array(img)
    
    # Normalizar (0-255 → 0-1)
    img_array = img_array / 255.0
    
    # Adicionar dimensão batch
    img_array = np.expand_dims(img_array, axis=0)
    
    return img_array

def pipeline_classificacao_especies(caminho_imagem, 
                                  caminho_modelo='modelos_salvos/melhor_modelo_especies_final.h5',
                                  caminho_encoder='datasets_processados/label_encoder_especies_modelo.pkl'):
    """
    Pipeline completo para classificar espécie de uma imagem
    
    Args:
        caminho_imagem: Caminho para a imagem a ser classificada
        caminho_modelo: Caminho para o modelo treinado
        caminho_encoder: Caminho para o label encoder
    
    Returns:
        dict: Resultado da classificação com espécie, probabilidades e confiança
    """
    # Carregar modelo e encoder
    modelo, encoder = carregar_modelo_especies(caminho_modelo, caminho_encoder)
    
    # Preprocessar imagem
    imagem_prep = preprocessar_imagem_para_especies(caminho_imagem)
    
    # Classificar
    especie, probs, confianca = classificar_especie(modelo, encoder, imagem_prep)
    
    # Criar dicionário de probabilidades por classe
    classes = encoder.classes_
    prob_dict = {classe: float(prob) for classe, prob in zip(classes, probs)}
    
    return {
        'especie_predita': especie,
        'confianca': float(confianca),
        'probabilidades': prob_dict,
        'todas_especies': list(classes)
    }