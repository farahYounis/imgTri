
from ultralytics import YOLO
from pathlib import Path
from PIL import Image

# 1. Paths
image_path = Path("data/shirt.jpg")
model_path = Path("models/best.pt")
output_dir = Path("crops_fashion")

# 2. Create the output folder
output_dir.mkdir(parents=True, exist_ok=True)

# 3. Load the model and original image
model = YOLO(str(model_path))
image = Image.open(image_path).convert("RGB")

# 4. Detect objects
results = model.predict(
    source=str(image_path),
    conf=0.25
)

result = results[0]
saved_count = 0

# 5. Crop every detected object
for i, box in enumerate(result.boxes):

    class_id = int(box.cls[0])
    class_name = result.names[class_id]
    confidence = float(box.conf[0])
        # Filter low-confidence detections
    if confidence < 0.50:
        print(f"Skipped {class_name}: low confidence ({confidence:.2f})")
        continue

    # Get bounding-box coordinates
    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())

    width = x2 - x1
    height = y2 - y1

    # Filter very small crops
    if width < 64 or height < 64:
        print(f"Skipped {class_name}: crop too small ({width}x{height})")
        continue

   
    # Get the bounding box coordinates
    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())

    # Crop the object from the original image
    crop = image.crop((x1, y1, x2, y2))

    # Build the output filename
    filename = f"{i}_{class_name}_{confidence:.2f}.jpg"
    save_path = output_dir / filename

    # Save the crop
    crop.save(save_path)

    print(f"Saved: {save_path}")
    print(f"Class: {class_name}, Confidence: {confidence:.2f}")
    print(f"Crop size: {crop.size}")
    print("-" * 30)

    saved_count += 1

print(f"Finished! Saved {saved_count} crops.")
