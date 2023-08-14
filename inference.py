import json
import logging
import sys
import os
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import io
import requests

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler(sys.stdout))
JSON_CONTENT_TYPE = 'application/json'
JPEG_CONTENT_TYPE = 'image/jpeg'

hyperparameters={'layers': [512], 'model_name': 'resnet50', 'num_classes': 133}

def net(model_name, num_classes, layers):
    
    logger.info("Model creation for fine-tuning started.")
    
    model = eval("models."+model_name)(pretrained=True, progress=True)
    
    for param in model.parameters():
        param.requires_grad = False

    num_features = model.fc.in_features
    
    full = [num_features,]+layers+[num_classes,]
    
    seq = list()
    
    for i in range(len(full)-2):
        seq.append(nn.Linear(full[i], full[i+1]))
        seq.append(nn.ReLU())
    
    seq.append(nn.Linear(full[-2], full[-1]))
    
    model.fc = nn.Sequential(*seq)

    logger.info("Model creation completed.")

    return model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



def model_fn(model_dir):
    logger.info("In model_fn. Model directory is -")
    logger.info(model_dir)
    model = net(**hyperparameters).to(device)
    
    with open(os.path.join(model_dir, "model.pth"), "rb") as f:
        logger.info("Loading the dog-classifier model")
        checkpoint = torch.load(f , map_location =device)
        model.load_state_dict(checkpoint)
        logger.info('MODEL-LOADED')
        logger.info('model loaded successfully')
    model.eval()
    return model




def input_fn(request_body, content_type=JPEG_CONTENT_TYPE):
    logger.info('Deserializing the input data.')
    # process an image uploaded to the endpoint
    #if content_type == JPEG_CONTENT_TYPE: return io.BytesIO(request_body)
    logger.debug(f'Request body CONTENT-TYPE is: {content_type}')
    logger.debug(f'Request body TYPE is: {type(request_body)}')
    if content_type == JPEG_CONTENT_TYPE:
        logger.debug('Loading JPEG content')
        return Image.open(io.BytesIO(request_body))
    
    # process a URL submitted to the endpoint
    
    if content_type == JSON_CONTENT_TYPE:
        #img_request = requests.get(url)
        logger.debug(f'Request body is: {request_body}')
        request = json.loads(request_body)
        logger.debug(f'Loaded JSON object: {request}')
        url = request['url']
        img_content = requests.get(url).content
        return Image.open(io.BytesIO(img_content))
    
    raise Exception('Requested unsupported ContentType in content_type: {}'.format(content_type))

# inference
def predict_fn(input_object, model):
    logger.info('In predict fn')
    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    logger.info("transforming input")
    input_object=test_transform(input_object).to(device)
    
    with torch.no_grad():
        logger.info("Calling model")
        prediction = model(input_object.unsqueeze(0))
    return prediction