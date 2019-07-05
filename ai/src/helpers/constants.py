import os

class Path:
    root = os.getcwd()
    root = os.path.join(root, "ai") 
    data = os.path.join(root, "data") 
    img = os.path.join(data, "img")
    img_original = os.path.join(img, "original")
    img_synthetic = os.path.join(img, "synthetic")
    test = os.path.join(data, "test")
    train = os.path.join(data, "train")
    validate = os.path.join(data, "validate")
    models = os.path.join(root, "saved_models")

