import torch
from CNN_model import FirstCNN
import albumentations as A
from PIL import Image
import numpy as np
import io


class AlbumentationsTransform:
    def __init__(self, aug):
        self.aug = aug

    def __call__(self, image):
        # image: PIL Image
        image = np.array(image)
        augmented = self.aug(image=image)
        return augmented['image']


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = FirstCNN()
# model = torch.load(torch.load("CNN_model.pth"), map_location=torch.device('cpu'))
model.load_state_dict(torch.load("CNN_model.pth", map_location=torch.device('cpu')))
model.eval()


def preprocess_image(image_bytes):
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    transforms = A.Compose([
        A.Resize(64, 64),
        A.CenterCrop(56, 56),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        A.ToTensorV2()
    ])
    out = transforms(image=np.array(image))
    img_tensor = out['image']  # This is now a tensor
    return img_tensor.unsqueeze(0)


async def predict_image(img_tensor):
    with torch.no_grad():
        output = model(img_tensor)
    prob = torch.sigmoid(output)
    prediction = (prob > 0.5).long()  # 0 or 1
    labels = ["cat", "dog"]  # Or whatever your actual class names are
    pred = labels[prediction.item()]
    return pred
