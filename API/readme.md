# AWS Hackathon Challenge FLASK-API 

## How to run the API Code Locally

To initiall setup the enviroment 

```
virtualenv venv
source venv/bin/activate (on Ubuntu / Apple)
./venv/Script/activate (Windows)
pip3 install -r requirements.txt

```
You can download the latest model by using the endpoint 
```
http://127.0.0.1/api/download
```
To download on live version 
```
http://127.0.0.1/api/download

#For live version use this

http://ec2-54-224-68-136.compute-1.amazonaws.com:8080/api/download
```
To predict using the model call this endpoint
```
http://127.0.0.1/api/predict

#For live version use this

http://ec2-54-224-68-136.compute-1.amazonaws.com:8080/api/predict
```

You may use POSTMAN to see the params as well, the predict takes a form-data picture whereas downloading the model is a GET request.


To finally run the API locally
```
python api.py
```
