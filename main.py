from ultralytics import YOLO
import cv2
import matplotlib.pyplot as plt

def detect_objects(image_path):
    # 1. YOLO-Modell laden (vortrainiertes Modell)
    model = YOLO('yolov8n.pt')  # 'yolov8n.pt' ist das Nano-Modell (schnell und leicht)

    # 2. Bild laden
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Bild '{image_path}' konnte nicht geladen werden.")
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Konvertiere zu RGB für die Anzeige

    # 3. Objekterkennung durchführen
    results = model(image_rgb)

    # 4. Ergebnisse anzeigen
    annotated_image = results[0].plot()  # Ergebnisse direkt auf das Bild zeichnen
    plt.imshow(annotated_image)
    plt.axis('off')
    plt.show()

    # 5. Zusätzliche Informationen ausgeben
    detected_objects = []
    for result in results[0].boxes:
        detected_objects.append({
            "class": result.cls,
            "confidence": result.conf,
            "box": result.xyxy.tolist()
        })
    return detected_objects

if __name__ == "__main__":
    image_path = "fu.jpg"  # Beispielbild
    objects = detect_objects(image_path)
    print("Erkannte Objekte:", objects)
