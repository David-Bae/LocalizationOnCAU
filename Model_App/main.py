from tkinter import *
from tkinter import filedialog
from PIL import Image, ImageTk
import torch
from dataset import transform
from model import create_model
from config import PRETRAINED_MODEL

class ImageViewerApp:
    def __init__(self, root, model_path):
        self.root = root
        self.root.title("Image Viewer")
        self.root.geometry("800x600")
###############################################################################
        self.model = create_model('ResNet')
        self.model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
###############################################################################

        self.img_label = Label(root)
        self.img_label.grid(row=0, column=0, columnspan=2, pady=10, sticky="nsew")

        self.img = None
        self.photo = None

        self.show_empty_image()

        open_button = Button(root, text="Upload Image", command=self.open_file, font=("Helvetica", 12), padx=10, pady=5)
        open_button.grid(row=1, column=0, padx=5, pady=(0, 20))  # Add pady to add space below the button

        confirm_button = Button(root, text="Where I am?", command=self.confirm_action, font=("Helvetica", 12), padx=10, pady=5)
        confirm_button.grid(row=1, column=1, padx=5, pady=(0, 20))  # Add pady to add space below the button

        # Configure row and column weights for centering
        root.grid_rowconfigure(0, weight=1)
        root.grid_columnconfigure(0, weight=1)
        root.grid_columnconfigure(1, weight=1)

        self.image_mapping = {0: "map_0.png", 1: "map_1.png", 2: "map_2.png", 3: "map_3.png", 4: "map_4.png",
                              5: "map_5.png", 6: "map_6.png", 7: "map_7.png", 8: "map_8.png", 9: "map_9.png"}

        image_path = './map'

        self.image_mapping = {key: image_path + '/' + value for key, value in self.image_mapping.items()}

    def open_file(self):
        file_path = filedialog.askopenfilename()

        if file_path:
            self.show_image(file_path)

    def show_image(self, file_path):
        self.img = Image.open(file_path)
        self.img = self.img.resize((500, 500))
        self.photo = ImageTk.PhotoImage(self.img)

        self.img_label.config(image=self.photo)
        self.img_label.image = self.photo

    def show_empty_image(self):
        empty_img = Image.new("RGB", (500, 500), "white")
        empty_photo = ImageTk.PhotoImage(empty_img)

        self.img_label.config(image=empty_photo)
        self.img_label.image = empty_photo

    def confirm_action(self):
        if self.photo:
            self.model.eval()
            image = transform(self.img).unsqueeze(0)
            with torch.no_grad():
                outputs = self.model(image)
                _, predicted = torch.max(outputs, 1)

            if 0 <= predicted.item() <= 9:
                image_filename = self.image_mapping[predicted.item()]
                self.show_confirmed_image(image_filename)
            else:
                print("Invalid number")

    def show_confirmed_image(self, image_filename):
        confirmed_image_window = Toplevel(self.root)
        confirmed_image_window.title("You're located here")

        img = Image.open(image_filename)
        img = img.resize((500, 500))
        photo = ImageTk.PhotoImage(img)

        img_label = Label(confirmed_image_window, image=photo)
        img_label.image = photo
        img_label.pack()

if __name__ == "__main__":
    root = Tk()
    app = ImageViewerApp(root, PRETRAINED_MODEL)
    root.mainloop()
