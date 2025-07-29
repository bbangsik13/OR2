import torchvision
# Used for debugging
def save_tensor_img(img, name='rendering'):
    torchvision.utils.save_image(img, name+".png")

