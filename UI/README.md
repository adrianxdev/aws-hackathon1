# RECOVER-EASE AWS Hackathon Entry

## What is this?
A web app that allows you to use custom trained model built for the disaster recovery hackaton. 

The app is written in Node.js and uses a custom model deployed on EC2 to perform the image recognition and analysis. 


## Functionality
- grab location (must be running localhost or over https)
- upload image
- get textual description (labels) for image
- option to submit a request via email



## Repo Contents
- /views = for speed and simplicity, everything is in the index.ejs


## Install UI
```
npm install
```
## Run it
run the webserver:
```
node app.js
```
point your browser at the local/remoteIP port 3000 to load the HTML form

e.g http://127.0.0.1:3000/

