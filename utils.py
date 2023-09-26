from PIL import Image

def keep_image_size_open (path,size=(128,128)):
    
    img = Image.open(path)
    temp = max(img.size)
    mask = Image.new('RGB',(128,128),(255,255,255))
    mask.paste(img,(0,0)) #paste to the left-top corner

    return mask

#96*103 - 128*128
