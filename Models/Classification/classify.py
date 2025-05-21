from torchvision import transforms
import torch
import os
from .CNN import CNN

def preprocessing(image):
    """Preprocesses an image for classification."""
    transformation = transforms.Compose([
        transforms.Resize((60,60)),
        transforms.ToTensor()
    ])
    image = transformation(image)
    return image


def classifier(image, weights, class_names):

    print("\n\n Model \n\n")
    
    model = CNN()
    # Load model weights  
    
    print("\n\n Model A \n\n")

    weights_path = os.path.join(os.path.dirname(__file__),"..","Weights",f"{weights}")
    
    print("\n\n Model B \n\n")

    state_dict = torch.load(weights_path, map_location=torch.device('cpu'))
    
    print("\n\n Model C \n\n")

    model.load_state_dict(state_dict)
    
    print("\n\n Model D \n\n")

    # model = torch.load(weights_path)
    
    # Set the model to evaluation mode
    model.eval()  
    
    preprocessed_img = preprocessing(image)    
    preprocessed_img = preprocessed_img.unsqueeze(0)

    output = model(preprocessed_img)
    pred = torch.argmax(output).item()
    label = class_names[pred]
    score = output[0][pred].item()
    score = round(score * 100, 2)
    score = int(score)
    
    return label,score