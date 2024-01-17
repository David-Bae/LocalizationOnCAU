import cv2 as cv2
import numpy as np

err_not_np_img= "not a numpy array or list of numpy array"
epsilon = 1e-10     # Variable for Floating point comparison

# Verify whether image is numpy image.
def verify_image(image):
    if isinstance(image, np.ndarray):
        pass
    else:
        raise Exception(err_not_np_img)


def BGR2HLS(bgrImg):
    height, width, _ = bgrImg.shape

    # Create HSL image of same size with RGB image
    hlsImg = np.zeros((height, width, 3))

    for y in range(height):
        for x in range(width):
            b, g, r = bgrImg[y, x] / 255
            maxRGB = max(r, g, b)
            minRGB = min(r, g, b)

            # Calculating Hue
            diff = maxRGB - minRGB
            if diff == 0:
                h = 0
            elif abs(maxRGB - r) < epsilon:     # Max is R
                h = 30 * ((g-b)/diff)
            elif abs(maxRGB - g) < epsilon:     # Max is G
                h = 30 * ((b-r)/diff + 2)
            elif abs(maxRGB - b) < epsilon:     # Max is B
                h = 30 * ((r-g)/diff + 4)

            if h < 0:
                h = h + 180

            # Calculating Lightness
            l = (maxRGB + minRGB) / 2

            # Calculating Saturation
            if diff == 0:
                s = 0
            else:
                s = diff / (1 - abs(2*l - 1))

            hlsImg[y, x] = np.array([h, l*255, s*255])

    hlsImg = np.array(hlsImg, dtype = np.uint8)
    return hlsImg

def HLS2BGR(hlsImg):
    height, width, _ = hlsImg.shape

    # Create HSL image of same size with RGB image
    bgrImg = np.zeros((height, width, 3))
    height, width = 1,1

    for y in range(height):
        for x in range(width):
            h, l, s = hlsImg[y, x]
            l = l / 255
            s = s / 255

            chroma = (1-abs(2*l-1))*s
            secondary = chroma*(1-abs((h/30)%2 -1))
            match = l - chroma/2

            if 0 <= h < 30:
                r, g, b = chroma, secondary, 0
            elif 30 <= h < 60:
                r, g, b = secondary, chroma, 0
            elif 60 <= h < 90:
                r, g, b = 0, chroma, secondary
            elif 90 <= h < 120:
                r, g, b = 0, secondary, chroma
            elif 120 <= h < 150:
                r, g, b = secondary, 0, chroma
            elif 150 <= h < 180:
                r, g, b = chroma, 0, secondary

            r, g, b = (r+match)*255, (g+match)*255, (b+match)*255
            
            bgrImg[y, x] = np.array([b, g, r])

    bgrImg = np.array(bgrImg, dtype = np.uint8)
    return bgrImg

############################################################################################################
def draw_line(img, start, end, color):
    x1, y1 = start
    x2, y2 = end
    dx = abs(x2 - x1)
    dy = abs(y2 - y1)
    x, y = x1, y1
    sx = -1 if x1 > x2 else 1
    sy = -1 if y1 > y2 else 1
    
    if dx > dy:
        err = dx / 2.0
        while x != x2:
            img[y, x] = color
            err -= dy
            if err < 0:
                y += sy
                err += dx
            x += sx
    else:
        err = dy / 2.0
        while y != y2:
            img[y, x] = color
            err -= dx
            if err < 0:
                x += sx
                err += dy
            y += sy
    img[y, x] = color

def blur(image, kernel_size):
    kernel_height, kernel_width = kernel_size
    height, width = image.shape[:2]
    blurred = np.zeros_like(image)

    for y in range(height):
        for x in range(width):
            y_start, y_end = max(0, y - kernel_height // 2), min(height, y + kernel_height // 2 + 1)
            x_start, x_end = max(0, x - kernel_width // 2), min(width, x + kernel_width // 2 + 1)
            kernel = image[y_start:y_end, x_start:x_end]
            blurred[y, x] = np.mean(kernel, axis=(0, 1))

    return blurred

def generate_random_lines(imshape,slant,drop_length):
    drops=[]
    area=imshape[0]*imshape[1]
    no_of_drops=area//600

    for i in range(no_of_drops): ## If You want heavy rain, try increasing this
        if slant<0:
            x= np.random.randint(slant,imshape[1])
        else:
            x= np.random.randint(0,imshape[1]-slant)
        y= np.random.randint(0,imshape[0]-drop_length)
        drops.append((x,y))
    return drops

def rain_process(image,slant,drop_length,drop_color,rain_drops):
    
    image_t= image.copy()

    for rain_drop in rain_drops:
        draw_line(image_t,(rain_drop[0],rain_drop[1]),(rain_drop[0]+slant,rain_drop[1]+drop_length),drop_color)

    image = blur(image_t,(5,5))  # rainy view are blurry
    image = day2night(image, 0.2)   # dark on rainy day

    return image

##rain_type='drizzle','heavy','torrential'
def add_rain(image, drop_length=20, drop_color=(200,200,200)): ## (200,200,200) a shade of gray
    imshape = image.shape

    slant= np.random.randint(-10,10) ##generate random slant if no slant value is given
    rain_drops= generate_random_lines(imshape,slant,drop_length)

    output= rain_process(image, slant, drop_length, drop_color, rain_drops)# slant_extreme to slant
    image_RGB=output

    return image_RGB

# Day to Night ############################################################################################
def day2night(img, coeff):
    verify_image(img)
    coeff = 1 - coeff

    image_HLS = BGR2HLS(img) ## Conversion to HLS
    image_HLS = np.array(image_HLS, dtype = np.float64) 
    image_HLS[:,:,1] = image_HLS[:,:,1]*coeff ## scale pixel values up or down for channel 1(Lightness)
    if(coeff>1):
        image_HLS[:,:,1][image_HLS[:,:,1]>255]  = 255 ##Sets all values above 255 to 255
    else:
        image_HLS[:,:,1][image_HLS[:,:,1]<0]=0
    image_HLS = np.array(image_HLS, dtype = np.uint8)
    #image_RGB = HLS2BGR(image_HLS) ## Conversion to RGB
    image_RGB = cv2.cvtColor(image_HLS, cv2.COLOR_HLS2BGR) ## Conversion to RGB

    return image_RGB
# Rotation ################################################################################################
def rotate(image, angle):
    height, width = image.shape[:2]

    center = (image.shape[1] // 2, image.shape[0] // 2)
    rad = np.radians(angle)
    rotation_matrix = np.array([[np.cos(rad), -np.sin(rad)],
                                [np.sin(rad), np.cos(rad)]])

    rotated = np.zeros_like(image)

    for x in range(width):
        for y in range(height):
            offset = np.array([x - center[0], y - center[1]])
            rotated_offset = np.dot(rotation_matrix, offset)
            new_x, new_y = int(rotated_offset[0] + center[0]), int(rotated_offset[1] + center[1])

            if 0 <= new_x < width and 0 <= new_y < height:
                rotated[y, x] = image[new_y, new_x]

    return rotated
############################################################################################################

# Cropping #################################################################################################
def cropping(img, left, up, right, down):
    cropped = img[up:down, left:right]

    return cropped
############################################################################################################

if __name__ == '__main__':
    img = cv2.imread("data/original_data/1.jpg", cv2.IMREAD_COLOR)

    rotated = rotate(img, 10)


    cv2.imwrite("data/1.jpg", rotated)