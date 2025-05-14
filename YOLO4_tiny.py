import cv2
import numpy as np
import os

def setup_model():
    # Configurar el modelo
    model_dir = "dnn_model"
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
        
    # Rutas de los archivos del modelo
    model_cfg = os.path.join(model_dir, "yolov4-tiny.cfg")
    model_weights = os.path.join(model_dir, "yolov4-tiny.weights")
    classes_file = os.path.join(model_dir, "classes.txt")

    # Cargar la red YOLO
    net = cv2.dnn.readNet(model_weights, model_cfg)
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

    # Cargar clases
    with open(classes_file, "r") as f:
        class_names = [line.strip() for line in f.readlines()]
        
    return net, class_names

def detect_objects(frame, net, class_names):
    # Preprocesar imagen
    blob = cv2.dnn.blobFromImage(frame, 1/255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    
    # Obtener capas de salida (versión corregida)
    layer_names = net.getLayerNames()
    output_layers = []
    for i in net.getUnconnectedOutLayers():
        if isinstance(i, (list, np.ndarray)):
            output_layers.append(layer_names[i[0] - 1])
        else:
            output_layers.append(layer_names[i - 1])
    
    # Detección
    outs = net.forward(output_layers)
    
    # Inicializar listas
    boxes = []
    confidences = []
    class_ids = []
    
    # Dimensiones del frame
    height, width = frame.shape[:2]
    
    # Procesar detecciones
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            
            if confidence > 0.5:
                # Coordenadas del objeto
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                
                # Puntos del rectángulo
                x = int(center_x - w/2)
                y = int(center_y - h/2)
                
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)
    
    return boxes, confidences, class_ids

def main():
    # Configurar el modelo
    net, class_names = setup_model()
    
    # Configurar entrada de video
    video_path = "av_revolucion.mp4"  # Cambia esto por tu archivo de video
    cap = cv2.VideoCapture(video_path)
    
    # Configurar parámetros del video
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    
    # Configurar video writer
    out = cv2.VideoWriter('output.mp4', 
                         cv2.VideoWriter_fourcc(*'mp4v'), 
                         fps, 
                         (frame_width, frame_height))
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        # Detectar objetos
        boxes, confidences, class_ids = detect_objects(frame, net, class_names)
        
        # Aplicar non-maximum suppression
        indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
        
        # Dibujar detecciones
        if len(indices) > 0:
            for i in indices.flatten():
                x, y, w, h = boxes[i]
                label = f"{class_names[class_ids[i]]}: {confidences[i]:.2f}"
                
                # Dibujar bbox y etiqueta
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(frame, label, (x, y - 10),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # Mostrar y guardar frame
        cv2.imshow("YOLOv4-tiny Detection", frame)
        out.write(frame)
        
        # Salir con 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Liberar recursos
    cap.release()
    out.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()