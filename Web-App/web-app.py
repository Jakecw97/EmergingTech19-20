#Import flask
import flask as fl

#Import numpy, allows use of many matrices, arrays and other mathamatical features. CV2 allows for image use. Base64 allows encoding and decoding of binary data.
import numpy as np
import cv2
import base64

#Python library for image use.
from PIL import Image, ImageOps

#Import the trained model
from keras.models import load_model

#Load in the trained model
trainedModel = load_model('../predictionModel.h5')

#Initiate the flask application
app = fl.Flask(__name__)

#Sizing
height = 28
width = 28
size = height, width

# Routes to index.html, the default for this project.
@app.route('/')
def home():
    # returns html file (index.html displayed)
    return fl.render_template('index.html')

# Routes to the predict page, this converts the image
@app.route('/predict', methods=['POST'])
def convertImage():
	#Encoding
    encoded = fl.request.values[('imgBase64')]
	#Decoding
    decoded = base64.b64decode(encoded[22:])

    #Save image from canvas
    with open('Image.png', 'wb') as f:
        f.write(decoded)
	#Opens the Image
    userImg = Image.open("Image.png")
	#Resizes the image
    newImg = ImageOps.fit(userImg, size, Image.ANTIALIAS)
    #Save the resized image
    newImg.save("ResizedImage.png")

    #Load ResizedImage
    cv2Image = cv2.imread("ResizedImage.png")

    #Convert the resized image to grayscale
    grayScaleImage = cv2.cvtColor(cv2Image, cv2.COLOR_BGR2GRAY)

    grayScaleArray = np.array(grayScaleImage, dtype=np.float32).reshape(1, 784)
    grayScaleArray /= 255

    #Get and Set methods for prediction
    setPrediction = trainedModel.predict(grayScaleArray)
    getPrediction = np.array(setPrediction[0])
	#Sets the predicted number
    predNum = str(np.argmax(getPrediction))
    #Prints and rerutns the predicted number
    print(predNum)
    return predNum

#Application running
app.run(threaded=False)