import onnxruntime
import numpy as np
import onnx
from PIL import Image
from dataset import transform

def preprocess_image(image_path):
    image = Image.open(image_path).convert('RGB')
    image = transform(image).numpy()
    image = np.expand_dims(image, axis=0)
    return image


if __name__ == '__main__':
    input_image = preprocess_image('./data/test/101/0_003.jpg')
    ort_session = onnxruntime.InferenceSession("model.onnx")

    

    outputs = ort_session.run(
        None,
        {'modelInput': input_image.astype(np.float32)}
    )

    print(outputs)