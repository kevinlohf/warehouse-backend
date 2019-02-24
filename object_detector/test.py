from qr_detector import QRModel
from PIL import Image

model = QRModel("model.onnx")
img = Image.open("test.jpg")

print(model.predict(img))
