import os
import json
import pandas as pd

# with open("annotations.csv",'w') as f:
#     for im in os.listdir('annos'):
#         with open ("annos/"+im,'r') as img:
#             try:
#                 d_im = json.load(img)
#                 for key in d_im.keys():
                    
#                         if 'item' in key:
#                             value = d_im[key]
#                             f.write(im+":"+str(value["bounding_box"])+":"+value["category_name"]+"\n")
#             finally:
#                 continue

