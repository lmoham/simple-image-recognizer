from platform import platform
from keras.models import load_model
from tensorflow.python import platform
import numpy as np
from tkinter import *
import tkinter.filedialog as fd
import cv2
from PIL import Image
from PIL import ImageTk


labels = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
model = load_model('Trained_model.h5', compile=False)


def select_image():
    # Grab reference to panel
    global panelA, panelB

    # User will input desired image file
    path = fd.askopenfilename()

    # Ensure user selected an image
    if len(path) > 0:
        # Reading image
        input_image = cv2.imread(path)

        # Converting image to PIL format
        input_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)
        input_image = Image.fromarray(input_image)

        # Converting image to CIFAR-10 image format
        converted_image = input_image.resize((32, 32), resample=Image.LANCZOS)
        image_array = np.array(converted_image)
        image_array = image_array.astype('float32')
        image_array /= 255
        image_array = image_array.reshape(1, 32, 32, 3)
        prediction = model.predict(image_array)
        answer2 = labels[np.argmax(prediction)]
        answer = ("Output: \n" + answer2)
        input_image = input_image.resize((260,230))

        # Converting image into format understandable by Tk function
        p1 = ImageTk.PhotoImage(input_image)

        if panelA is None or panelB is None:
            # First panel shows input image
            panelA = Label(image=p1)
            panelA.p1 = p1
            panelA.pack(side="top", padx=10, pady=10)

            # Second panel shows result of prediction
            panelB = Label(text=answer)
            panelB.pack(side="right", padx=10, pady=10)

        else:
            # If panels already have something, then simply update them
            panelA.configure(image=input_image)
            panelB.configure(text=answer)
            panelA.image=input_image
            panelB.text=answer


# Initialize the user interface
root = Tk()
panelA = None
panelB = None

# Button to allow user to input image file
button = Button(root, text="Select an image", Command = select_image())
button.pack(side="bottom", fill="both", expand="no", padx="10", pady="10")

# Start GUI
root.mainloop()




