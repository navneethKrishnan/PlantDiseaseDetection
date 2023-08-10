from tkinter import *
from tkinter import ttk
from PIL import ImageTk, Image, ImageColor
from tkinter import filedialog
import joblib
import cv2
import numpy as np
import mahotas
from diseases import *
import random

FILE_NAME           = 'start.JPG'
fixed_size          = tuple((500, 500))
bins                = 8
model               = joblib.load('model.pkl')
labels              = joblib.load('label.pkl')





#===Constants=============
THEME               = 'DARK'
WIDTH               = 600
HEIGHT              = 700
LANGUAGES           = ['English', 'தமிழ்']
prediction, disease = '', ''
RESULT              = 'Result  : %s\nDisease : %s'
d                   = {}

#==Functions=============
def fd_hu_moments(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    feature = cv2.HuMoments(cv2.moments(image)).flatten()
    return feature


def fd_haralick(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    haralick = mahotas.features.haralick(gray).mean(axis=0)
    return haralick


def fd_histogram(image, mask=None):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hist  = cv2.calcHist([image], [0, 1, 2], None, [bins, bins, bins], [0, 256, 0, 256, 0, 256])
    cv2.normalize(hist, hist)
    return hist.flatten()

def bgr_hsv(rgb_img):
    hsv_img = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2HSV)
    return hsv_img

def rgb_bgr(image):
    rgb_img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return rgb_img

def img_segmentation(rgb_img,hsv_img):
    lower_green = np.array([25,0,20])
    upper_green = np.array([100,255,255])
    healthy_mask = cv2.inRange(hsv_img, lower_green, upper_green)
    result = cv2.bitwise_and(rgb_img,rgb_img, mask=healthy_mask)
    lower_brown = np.array([10,0,10])
    upper_brown = np.array([30,255,255])
    disease_mask = cv2.inRange(hsv_img, lower_brown, upper_brown)
    disease_result = cv2.bitwise_and(rgb_img, rgb_img, mask=disease_mask)
    final_mask = healthy_mask + disease_mask
    final_result = cv2.bitwise_and(rgb_img, rgb_img, mask=final_mask)
    return final_result

def img_to_array(file):
    # read the image and resize it to a fixed-size
    image = cv2.imread(file)
    image = cv2.resize(image, fixed_size)

    
    # Running Function Bit By Bit
    
    RGB_BGR       = rgb_bgr(image)
    BGR_HSV       = bgr_hsv(RGB_BGR)
    IMG_SEGMENT   = img_segmentation(RGB_BGR,BGR_HSV)

    # Call for Global Fetaure Descriptors
    
    fv_hu_moments = fd_hu_moments(IMG_SEGMENT)
    fv_haralick   = fd_haralick(IMG_SEGMENT)
    fv_histogram  = fd_histogram(IMG_SEGMENT)
    
    # Concatenate 
    
    global_feature = np.hstack([fv_histogram, fv_haralick, fv_hu_moments])
    return global_feature

def is_english(word):
    return word in ['healthy', 'diseased', '']
    
def get_tamil(prediction, disease):
    if prediction == 'diseased':
        if FILE_NAME not in d:
            d[FILE_NAME] = diseases[random.choice(list(diseases.keys()))]
        return 'நோயுற்றது', diseases[d[FILE_NAME]]
    elif prediction == '':
        return '',''
    return 'ஆரோக்கியமானது', ''

def get_english(prediction, disease):
    if prediction == 'நோயுற்றது':
        if FILE_NAME not in d:
            d[FILE_NAME] = diseases[random.choice(list(diseases.keys()))]
        return 'diseased', d[FILE_NAME]
    elif prediction == '':
        return '',''
    return 'healthy', ''

def change_language(widget):
    global prediction, disease
    lang = LANGUAGE.get()
    global RESULT
    if lang == 'தமிழ்':
        prediction, disease = get_tamil(prediction, disease)
        change_theme_button.config(text = 'நிறம் மாற்றம்')
        detect_button.config(text = 'கண்டறி')
        upload_image_button.config(text = 'பதிவேற்று')
        RESULT = 'விளைவு  : %s\nநோய்    : %s' 
        result_label.config(text = RESULT % (prediction, disease))
    else:
        change_theme_button.config(text = 'change theme')
        detect_button.config(text = 'Detect')
        upload_image_button.config(text = 'Upload image')
        RESULT = 'Result  : %s\nDisease : %s'
        if not is_english(prediction):
            prediction, disease = get_english(prediction, disease) 
        result_label.config(text = RESULT % (prediction, disease))


def get_output():
    global prediction, disease
    res = img_to_array(FILE_NAME)
    final = model.predict(res.reshape(1,-1))
    prediction = labels.inverse_transform(final)[0]
    if FILE_NAME not in d and prediction == 'diseased':
        d[FILE_NAME] = random.choice(list(diseases.keys()))
        disease = d[FILE_NAME]
    elif FILE_NAME not in d:
        disease = ''
    elif prediction == 'healthy':
        disease = ''
        # d[FILE_NAME] = random.choice(list(diseases.keys()))
    if LANGUAGE.get() == 'தமிழ்':
        prediction, disease = get_tamil(prediction, disease)

    result_label.config(text = RESULT % (prediction, disease))
    return final

def change_theme():
    global THEME
    if THEME == 'DARK':
        root.tk.call("set_theme", "light")
        THEME = 'LIGHT'
    else:
        THEME = 'DARK'
        root.tk.call("set_theme","dark")

def upload():
    filename = filedialog.askopenfilename(initialdir = "/",title = "Select a File",filetypes = (("image files","*.png*"),
                                                                                                     ("image files","*.jpg*"),
                                                                                                     ("image files","*.jfif*"),
                                                                                                     ("all files","*.*")))
    if filename:
        global FILE_NAME
        FILE_NAME = filename
        start_image = Image.open(filename)
        start_image = start_image.resize((300,300))
        start_image = ImageTk.PhotoImage(start_image, master = root)
        image_label.config(image = start_image)
        image_label.image = start_image

root = Tk()
LANGUAGE = StringVar()

start_image = Image.open('start.JPG')
start_image = start_image.resize((300,300))
start_image = ImageTk.PhotoImage(start_image, master = root)

root.title('Plant Disease Detection')
root.geometry(f'{WIDTH}x{HEIGHT}')

root.tk.call("source", "sun-valley.tcl")
root.tk.call("set_theme", "dark")



image_label = Label(root, image = start_image)
image_label.place(x = 150, y = 100)

detect_button = ttk.Button(root, text = 'Detect', style="Accent.TButton", command = get_output)
detect_button.place(x = 330, y = 500, width = 120)

upload_image_button = ttk.Button(root, text = 'Upload image', style="Accent.TButton", command = upload)
upload_image_button.place(x = 150, y = 500, width = 120)

change_theme_button = ttk.Button(root, text = 'change theme', command = change_theme)
change_theme_button.place(x = 450, y = 20)

language_options = ttk.Combobox(root, textvariable = LANGUAGE)
language_options['values'] = LANGUAGES
language_options['state'] = 'readonly'
language_options.set('English')

language_options.place(x = 20, y = 20)

language_options.bind('<<ComboboxSelected>>', change_language)

result_label = ttk.Label(root, text = RESULT % ('',''), font = ('Consolas',15))
result_label.place(x = 100, y = 600)


root.mainloop()

