# from tensorflow.keras.preprocessing.image import load_img
# from keras.preprocessing.image import img_to_array
# from keras.applications.vgg16 import preprocess_input
# from keras.applications.vgg16 import decode_predictions
# from keras.applications.vgg16 import VGG16
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.models import model_from_json
import numpy as np
import cv2
def getPrediction(filename):
    image = cv2.imread('./static/uploads/'+filename)
    resize1= cv2.resize(image,(224,224), interpolation = cv2.INTER_AREA)
    json_file = open('./model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    model1 = model_from_json(loaded_model_json)
    # load weights into new model
    model1.load_weights("./model.h5")
    json_file = open('./modelsegout.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    model2 = model_from_json(loaded_model_json)
    # load weights into new model
    model2.load_weights("./modelsegout.h5")
    # resize1 = cv2.resize(image,(224,224), interpolation = cv2.INTER_AREA)
    resized =np.array(resize1).reshape(-1,224,224,3)/255.0
    res=model1.predict(resized).reshape(224,224)
    res1=model2.predict(resized).reshape(224,224)
    img31 = np.zeros((224,448,3), np.uint8)
    imgs=res.reshape(224,224,1)
    res[res>=0.95]=1
    res[res<0.95]=0
    res1[res1>=0.4]=1
    res1[res1<0.4]=0
    imgs[imgs>=0.4]=255
    imgs[imgs<0.4]=0
    img2 = cv2.merge((res,res,res))
    res1=np.where((res1==0)|(res1==1), 1-res1, res1)
    img3 = cv2.merge((res1,res1,res1))
    kernel = np.ones((3, 3), np.uint8)
    ero = cv2.erode(res,kernel,iterations = 5)
    resdil = cv2.dilate(ero,kernel,iterations = 3)
    img2_fg = cv2.bitwise_or(resdil,resdil,mask =res1.astype(np.uint8))
    img2_fg = img2_fg.reshape(224,224)
    img2_fg1 = cv2.merge((img2_fg,img2_fg,img2_fg))
    _, thresh = cv2.threshold(img2_fg.astype(np.uint8),0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    connectivity = 8  # You need to choose 4 or 8 for connectivity type
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(thresh , connectivity , cv2.CV_32S)
    # num_labels : count ,  img3 - segmented
    # image = load_img('upload/content/gdrive/MyDrive/U-nets/'+filename, target_size=(224, 224))
    # image = img_to_array(image)
    # image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
    # image = preprocess_input(image)
    # yhat = model.predict(image)
    # label = decode_predictions(yhat)
    # label = label[0][0]
    # print('%s (%.2f%%)' % (label[1], label[2]*100))
    return img2_fg1*255.0,num_labels