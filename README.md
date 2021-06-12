# Segementation of Tomatoes

This is a web app which segments the tomatoes from the images that is uploaded on the web page. The tomatos are shaded in white color and the rest of the objects in the image including the background is shaded black.


## Features

- Detect and segments most of tomatos in the image
- Gives an approximate count of tomato by counting the connected components in the segmented image.

> Before counting the connected components of segmented tomatos,
> some preprocessing techniques such as eroding and dilation is 
> applied with a kernel size of (3,3)

## Details of Model used
I have used mainly two models for more accurate segmentation of the tomatos:
- Model 1 : This model is used to segment the tomato as a whole
- Model 2 : This model is used to predict the border of each of the tomatos.

Then combination of this two models is used to  segment the occluding tomatos carefully. The structure of model that is used for Model 1 and Model 2 is U-net which is known for biomedical segmentation tasks. Model 1 is trained till it achievied an accuracy of 96.8 percent and Model 2 is trained till it achieved an accuracy of 94.8.

## Steps to run the webapp 
The json file of the model can be found in the repo. Use it to train the model and place the weights of the two models in the cloned folder.
First clone the repo and inside the folder named Tomatoseg , type the following:
```sh
pip install -r requirements.txt
export FLASK_APP=index.py
run flask
```


Verify the deployment by navigating to your server address in
your preferred browser.

```sh
127.0.0.1:8000
```

## License

MIT
