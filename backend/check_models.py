#!/usr/bin/env python3
"""Check which models are loaded and their configurations."""

from services.model_registry import get_registry
from config import settings
import os

print('='*80)
print('MODEL CONFIGURATION CHECK')
print('='*80)

print('\n1. MODELS CONFIGURED IN config.py:')
print('-'*80)
for name, cfg in settings.MODEL_REGISTRY.items():
    model_type = cfg.get('type', 'unknown')
    enabled = cfg.get('enabled', False)
    conf = cfg.get('conf_threshold', 0.5)
    zones = cfg.get('zones', [])
    path = cfg.get('path', '')
    classes = cfg.get('classes', {})
    
    status = '✅' if enabled else '❌'
    print(f'\n{status} {name}')
    print(f'   Type: {model_type}')
    print(f'   Enabled: {enabled}')
    print(f'   Confidence: {conf}')
    print(f'   Zones: {zones}')
    print(f'   Path exists: {os.path.exists(path) if path else "N/A"}')
    if classes:
        print(f'   Classes ({len(classes)}): {list(classes.values())[:5]}...')

print('\n' + '='*80)
print('2. LOADED MODELS IN RUNTIME:')
print('-'*80)

try:
    registry = get_registry()
    all_models = registry.get_all_models()
    
    print(f'\nTotal models in registry: {len(all_models)}')
    
    for model_info in all_models:
        status = '✅' if model_info.enabled else '❌'
        loaded = '🟢' if model_info.model is not None else '🔴'
        
        print(f'\n{status} {loaded} {model_info.model_key}')
        print(f'   Type: {model_info.model_type}')
        print(f'   Conf Threshold: {model_info.conf_threshold}')
        print(f'   Zones: {model_info.zones}')
        print(f'   Model object: {type(model_info.model).__name__ if model_info.model else "NOT LOADED"}')
        
except Exception as e:
    print(f'\n❌ Error loading model registry: {e}')
    import traceback
    traceback.print_exc()

print('\n' + '='*80)
print('3. CLASSES DETECTION CHECK:')
print('-'*80)

# Check if person detection works
print('\n🔍 Checking YOLO primary model for person detection...')
try:
    from services import yolo_service
    
    if hasattr(yolo_service, 'model') and yolo_service.model is not None:
        print('✅ Primary YOLO model is loaded')
        print(f'   Model type: {type(yolo_service.model).__name__}')
        print(f'   Model has names: {hasattr(yolo_service.model, "names")}')
        
        if hasattr(yolo_service.model, 'names'):
            names = yolo_service.model.names
            print(f'   Total classes: {len(names)}')
            print(f'   Class 0 (person): {names.get(0, "NOT FOUND")}')
            print(f'   Class 67 (cell phone): {names.get(67, "NOT FOUND")}')
    else:
        print('❌ Primary YOLO model NOT loaded')
        
except Exception as e:
    print(f'❌ Error checking YOLO model: {e}')

print('\n' + '='*80)
print('SUMMARY')
print('='*80)

# Count enabled vs disabled
config_enabled = sum(1 for cfg in settings.MODEL_REGISTRY.values() if cfg.get('enabled'))
config_total = len(settings.MODEL_REGISTRY)

print(f'\n📊 Configured models: {config_enabled}/{config_total} enabled')
print(f'📊 Runtime models: {len(all_models) if "all_models" in locals() else 0} loaded')

print('\n✅ Check complete!')
print('='*80)
