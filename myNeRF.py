# import os
# import re
# import gc
# import drjit as dr
import mitsuba as mi
import numpy as np
import cv2
import matplotlib.pyplot as plt
from myNeRF_helper import NeRF

mi.set_variant('cuda_ad_rgb', 'llvm_ad_rgb')

training_dir = 'D:/Self Project/DaliProject/training'
myNeRF = NeRF(training_dir,width=500,height=500, num_iterations_per_stage=50, CHECKPOINT_EVERY=15)
myNeRF.prep()
#myNeRF.train()
#myNeRF.render_final_images()
#myNeRF.plot_list(myNeRF.ref_images_np,"Ref")
# myNeRF.resume_last_ckpt()
# final_images = [mi.render(myNeRF.scene, sensor=myNeRF.sensors[i], spp=128) for i in range(22,28)]
# myNeRF.plot_list(final_images)
ref = [myNeRF.ref_images_np[i] for i in range(22,28)]
myNeRF.plot_list(ref)
#myNeRF.loss_plot()

# #Try plotting other angles
# other_angle_sensor = []
# other_angles = []
# number_of_cams = 33
# for i in range(number_of_cams):
#     start = -360.0/2
#     step = 360.0 / (number_of_cams-1)
#     other_angles.append(start + i*step)

# for angle in other_angles:
#     other_angle_sensor.append(mi.load_dict({
#             'type': 'perspective',
#             'fov': 45,
#             'to_world': mi.ScalarTransform4f().translate([0.5, 0.5, 0.5]) \
#                                             .rotate([0, 1, 0], angle)   \
#                                             .look_at(target=[0, 0, 0],
#                                                     origin=[0, 0, 1.3],
#                                                     up=[0, 1, 0]),
#             'film': {
#                 'type': 'hdrfilm',
#                 'width': 468,
#                 'height': 936,
#                 'filter': {'type': 'box'},
#                 'pixel_format': 'rgb'
#             }
#         }))

# test_img = [mi.render(myNeRF.scene, sensor=other_angle_sensor[i], spp=512) for i in range(len(other_angle_sensor))]
# for i,img in enumerate(test_img):
#     img_name = './dataset/' + 'knight' + str(i) + '.jpeg'
#     img_clipped = np.clip(img, 0.0, 1.0)          # clamp any values slightly outside [0,1]
#     img_uint8   = (img_clipped * 255).astype(np.uint8)
#     pic = mi.util.convert_to_bitmap(img_uint8)
#     pic.write(img_name)