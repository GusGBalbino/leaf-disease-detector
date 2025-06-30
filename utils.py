import pandas as pd
import numpy as np
import json
import pickle
import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator

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