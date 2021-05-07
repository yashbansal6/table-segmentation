from . import unet
model_from_name = {}

model_from_name["unet"] = unet.unet
model_from_name["mobilenet_unet"] = unet.mobilenet_unet