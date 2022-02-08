import torch
from PIL import Image
import numpy as np
import gc
from torchvision import transforms

import cv2
from flask import Flask, jsonify, request,render_template
from flask_cors import CORS
import os

import boto3
from dotenv import load_dotenv,find_dotenv
from botocore.exceptions import ClientError

load_dotenv(find_dotenv())


static_dir = str(os.path.abspath(os.path.join(__file__ , "..", 'templates/')))

app = Flask(__name__, static_folder=static_dir, static_url_path="", template_folder=static_dir)
cors = CORS(app, resources={r"/api/*": {"origins": '*'}})
s3_client = boto3.client(
's3',
aws_access_key_id=os.getenv("AWS_ACCESS_KEY"),
aws_secret_access_key=os.getenv("AWS_SECRET_KEY"),
)
S3_BUCKET = os.getenv("S3_BUCKET_NAME")

download_path = "download/model_weights.pth"
download_file = "model_weights.pth"
disable_print = True

def predict(model, images,dataset_classes=0):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    was_training = model.training
    model.eval()
    global prediction
    with torch.no_grad():
        inputs = images.to(device)
        #Adding a batch size of 1 since Torch uses 4D inputs including the batch!
        inputs = inputs[None,:]
        if disable_print:
            print(inputs.shape)
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        for j in range(inputs.size()[0]):
            if disable_print:
                print(f"Predicted: {dataset_classes[preds[j]]}")
            prediction = f"Predicted: {dataset_classes[preds[j]]}"
            #imshow(inputs.cpu().data[j])
        model.train(mode=was_training)
    if device.type == "cuda":
        inputs = inputs.cpu()
    gc.collect()
    return prediction

@app.route('/',methods=['GET'])
def index():
    return render_template("index.html")

@app.route('/api/download',methods=['GET'])
def download_model():
    if os.path.isfile(download_file):
        os.remove("model_weights.pth")
        if disable_print:
            print ("Existing Model was removed")
    s3_client.download_file(S3_BUCKET, download_path, download_file)
    if disable_print:
        print("Model Downloaded Successfully!")
    return jsonify({"message":"New model has been downloaded for inference"})

@app.route('/api/predict', methods=['POST'])
def predictor():
    file = request.files['Image'].read()
    npimg = np.frombuffer(file, np.uint8)
    img = cv2.imdecode(npimg,cv2.IMREAD_COLOR)
    if os.path.isfile(download_file):
        model = torch.load(download_file)
        model.eval()
        dataset_classes= ['electricUtility', 'downTrees', 'fire', 'flood', 'structural']            
        transform = transforms.Compose([transforms.Resize((230,230)),transforms.ToTensor()])    
        #Read a single image here, transform it, predict!
        image = Image.fromarray(img.astype("uint8")).convert('RGB').resize((240,240),resample=Image.LANCZOS)
        x = Image.fromarray(np.uint8(np.array(image))) # Memory Efficient way
        x = transform(x)
        prediction=predict(model,x,dataset_classes=dataset_classes)
        return jsonify(prediction)
    else:
        return jsonify({"message":"Please download the model via /download endpoint first to create inference!"})


if __name__ == '__main__':
    app.run(port=8080, debug=True, host='127.0.0.1')

