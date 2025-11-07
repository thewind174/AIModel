"""
Detección de Excavadoras usando Faster R-CNN con Detectron2
Incluye: Preparación de datos, entrenamiento y detección
"""

import torch
import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

import numpy as np
import os, json, cv2, random
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor, DefaultTrainer
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.structures import BoxMode
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data import build_detection_test_loader

# ================== PASO 1: PREPARAR DATASET ==================
def get_excavator_dicts(img_dir, annotation_file):
    """
    Carga el dataset de excavadoras en formato COCO
    
    Estructura esperada:
    - img_dir: carpeta con imágenes
    - annotation_file: archivo JSON con anotaciones en formato COCO
    
    Formato COCO simplificado:
    {
        "images": [{"id": 1, "file_name": "img1.jpg", "height": 480, "width": 640}],
        "annotations": [{"id": 1, "image_id": 1, "category_id": 1, 
                        "bbox": [x, y, width, height], "area": area}],
        "categories": [{"id": 1, "name": "excavadora"}]
    }
    """
    with open(annotation_file) as f:
        imgs_anns = json.load(f)
    
    dataset_dicts = []
    for idx, img_info in enumerate(imgs_anns["images"]):
        record = {}
        
        filename = os.path.join(img_dir, img_info["file_name"])
        record["file_name"] = filename
        record["image_id"] = img_info["id"]
        record["height"] = img_info["height"]
        record["width"] = img_info["width"]
        
        # Obtener anotaciones para esta imagen
        annos = [ann for ann in imgs_anns["annotations"] 
                if ann["image_id"] == img_info["id"]]
        
        objs = []
        for anno in annos:
            obj = {
                "bbox": anno["bbox"],
                "bbox_mode": BoxMode.XYWH_ABS,  # formato COCO (x,y,w,h)
                "category_id": 0,  # Solo tenemos una clase: excavadora
            }
            objs.append(obj)
        record["annotations"] = objs
        dataset_dicts.append(record)
    
    return dataset_dicts

# Registrar el dataset
def register_excavator_dataset(train_dir, train_json, val_dir, val_json):
    """Registra datasets de entrenamiento y validación"""
    for d in ["train", "val"]:
        if d == "train":
            DatasetCatalog.register(
                "excavator_" + d, 
                lambda: get_excavator_dicts(train_dir, train_json)
            )
        else:
            DatasetCatalog.register(
                "excavator_" + d, 
                lambda: get_excavator_dicts(val_dir, val_json)
            )
        MetadataCatalog.get("excavator_" + d).set(thing_classes=["excavadora"])
    
    return MetadataCatalog.get("excavator_train")

# ================== PASO 2: CONFIGURAR MODELO ==================
def setup_cfg(output_dir="./output", num_classes=1, max_iter=3000):
    """
    Configura Faster R-CNN con ResNet-50 como backbone
    """
    cfg = get_cfg()
    
    # Modelo base pre-entrenado en COCO
    cfg.merge_from_file(model_zoo.get_config_file(
        "COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"
    ))
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(
        "COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"
    )
    
    # Dataset
    cfg.DATASETS.TRAIN = ("excavator_train",)
    cfg.DATASETS.TEST = ("excavator_val",)
    cfg.DATALOADER.NUM_WORKERS = 2
    
    # Hiperparámetros de entrenamiento
    cfg.SOLVER.IMS_PER_BATCH = 2
    cfg.SOLVER.BASE_LR = 0.00025
    cfg.SOLVER.MAX_ITER = max_iter
    cfg.SOLVER.STEPS = (1000, 2000)  # Learning rate decay
    cfg.SOLVER.GAMMA = 0.1
    cfg.SOLVER.CHECKPOINT_PERIOD = 500
    
    # Modelo
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = num_classes
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # Umbral de confianza
    
    # Output
    cfg.OUTPUT_DIR = output_dir
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    
    return cfg

# ================== PASO 3: ENTRENAR ==================
def train_excavator_detector(cfg):
    """Entrena el modelo Faster R-CNN"""
    trainer = DefaultTrainer(cfg)
    trainer.resume_or_load(resume=False)
    trainer.train()
    return trainer

# ================== PASO 4: EVALUAR ==================
def evaluate_model(cfg, model_weights):
    """Evalúa el modelo en el conjunto de validación"""
    cfg.MODEL.WEIGHTS = model_weights
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
    predictor = DefaultPredictor(cfg)
    
    evaluator = COCOEvaluator("excavator_val", cfg, False, output_dir=cfg.OUTPUT_DIR)
    val_loader = build_detection_test_loader(cfg, "excavator_val")
    results = inference_on_dataset(predictor.model, val_loader, evaluator)
    print(results)
    return results

# ================== PASO 5: DETECCIÓN EN NUEVAS IMÁGENES ==================
def detect_excavators(image_path, cfg, model_weights, output_path=None):
    """
    Detecta excavadoras en una imagen
    
    Args:
        image_path: ruta a la imagen
        cfg: configuración del modelo
        model_weights: ruta a los pesos entrenados
        output_path: donde guardar la imagen con detecciones (opcional)
    """
    cfg.MODEL.WEIGHTS = model_weights
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
    predictor = DefaultPredictor(cfg)
    
    # Cargar imagen
    im = cv2.imread(image_path)
    
    # Realizar detección
    outputs = predictor(im)
    
    # Visualizar resultados
    v = Visualizer(im[:, :, ::-1],
                   metadata=MetadataCatalog.get("excavator_train"),
                   scale=1.0)
    out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    result_image = out.get_image()[:, :, ::-1]
    
    # Guardar o mostrar
    if output_path:
        cv2.imwrite(output_path, result_image)
    
    # Imprimir detecciones
    instances = outputs["instances"].to("cpu")
    boxes = instances.pred_boxes.tensor.numpy()
    scores = instances.scores.numpy()
    classes = instances.pred_classes.numpy()
    
    print(f"\n=== Detecciones en {image_path} ===")
    for i, (box, score, cls) in enumerate(zip(boxes, scores, classes)):
        print(f"Excavadora {i+1}: Confianza={score:.2f}, BBox={box}")
    
    return outputs, result_image

# ================== PASO 6: DETECCIÓN EN VIDEO ==================
def detect_excavators_video(video_path, cfg, model_weights, output_path="output_video.mp4"):
    """Detecta excavadoras en un video"""
    cfg.MODEL.WEIGHTS = model_weights
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
    predictor = DefaultPredictor(cfg)
    
    cap = cv2.VideoCapture(video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        outputs = predictor(frame)
        v = Visualizer(frame[:, :, ::-1],
                      metadata=MetadataCatalog.get("excavator_train"),
                      scale=1.0)
        result = v.draw_instance_predictions(outputs["instances"].to("cpu"))
        result_frame = result.get_image()[:, :, ::-1]
        
        out.write(result_frame)
        frame_count += 1
        
        if frame_count % 30 == 0:
            print(f"Procesados {frame_count} frames...")
    
    cap.release()
    out.release()
    print(f"Video procesado guardado en: {output_path}")

# ================== EJEMPLO DE USO COMPLETO ==================
if __name__ == "__main__":
    """
    Flujo completo de entrenamiento y detección
    """
    
    # 1. Configurar rutas de tu dataset
    TRAIN_DIR = "data/excavators/train/images"
    TRAIN_JSON = "data/excavators/train/annotations.json"
    VAL_DIR = "data/excavators/val/images"
    VAL_JSON = "data/excavators/val/annotations.json"
    
    # 2. Registrar dataset
    excavator_metadata = register_excavator_dataset(
        TRAIN_DIR, TRAIN_JSON, VAL_DIR, VAL_JSON
    )
    
    # 3. Configurar modelo
    cfg = setup_cfg(output_dir="./output_excavator", num_classes=1, max_iter=3000)
    
    # 4. Entrenar (comentar si ya tienes modelo entrenado)
    print("Iniciando entrenamiento...")
    trainer = train_excavator_detector(cfg)
    
    # 5. Evaluar
    print("\nEvaluando modelo...")
    model_weights = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
    evaluate_model(cfg, model_weights)
    
    # 6. Detectar en imagen individual
    print("\nDetectando en imagen de prueba...")
    test_image = "test_excavator.jpg"
    outputs, result = detect_excavators(
        test_image, 
        cfg, 
        model_weights, 
        output_path="detected_excavator.jpg"
    )
    
    # 7. Detectar en video (opcional)
    # detect_excavators_video("excavator_video.mp4", cfg, model_weights)
    
    print("\n¡Proceso completado!")