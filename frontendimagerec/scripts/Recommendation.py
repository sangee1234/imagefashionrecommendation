from pyexpat import features
from PIL import Image
import pandas as pd
import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

#from scipy.spatial.distance import cdist
# from keras.applications import vgg16
# from keras.models import Model
# from keras.applications.imagenet_utils import preprocess_input
import os
from pathlib import Path

import numpy as np

import torch
# import tensorflow as tf
# from tensorflow.keras.utils import load_img, img_to_array
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
import torchvision.models as models
from torchvision.models._utils import IntermediateLayerGetter

def get_similar_images(category, input_image):
  # vgg_model = vgg16.VGG16(weights='imagenet')
  # feat_extractor = Model(inputs=vgg_model.input, outputs=vgg_model.get_layer("fc2").output)
  # img = input_image.resize((224,224))
  # img_arr = img_to_array(img)
  # img_dim = np.expand_dims(img_arr, axis=0)
  # processed_img = preprocess_input(img_dim.copy())
  try:
    vgg_model = models.vgg19(pretrained=True)
    torch.save(vgg_model.state_dict(),"model_vgg.pth")
    # model = models.vgg16(pretrained=False)
    # if (dict_path != None):
    #     model.load_state_dict(torch.load(dict_path))
    # return_layers = {'15': 'out_layer15'}
    # model_with_multuple_layer = IntermediateLayerGetter(vgg_model.features, return_layers=return_layers)
    # features = model_with_multuple_layer(torch.from_numpy(input_image))
    features = vgg_model(torch.from_numpy(input_image))

  except Exception as e:
    print(e)

  features = vgg_model(input_image)


  source_features = feat_extractor.predict(processed_img)
  df = pd.read_csv("namazon_"+category+".csv")
  df[features] = features
 
  df[dist] =  cdist(df.features, source_features, metric="cosine")
  df_sorted = df.sort_values(by=['dist'])
  

def get_recommendations(input_image):
    cfg2 = get_cfg()
    cfg2.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
    # model_path = os.path('../../Downloads/model_final.pth')
    model_path = str(os.path.join(Path.home(), "Downloads/model_final.pth"))
    cfg2.MODEL.WEIGHTS = model_path
    cfg2.MODEL.DEVICE = 'cpu'
    predictor = DefaultPredictor(cfg2)
    outputs = predictor(input_image) 
    classes = outputs["instances"].pred_classes
    boxes = outputs["instances"].pred_boxes
    for idx, b in enumerate(boxes):
        cropped_img = img.crop(b)
        get_similar_images(classes[idx], cropped_img)
#   img = load_img(input_image_path, target_size=(224, 224))
