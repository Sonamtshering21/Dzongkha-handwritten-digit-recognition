import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image, ImageDraw, ImageOps
import numpy as np
import tkinter as tk
from tkinter import Canvas, Button, Label, filedialog
import io
import os

# Load the trained model from the same directory
def load_model():
    # Get the path to the model in the same directory as this script
    model_path = os.path.join(os.path.dirname(__file__), 'model.h5')
    
    # Initialize model with same architecture as training
    model = models.resnet18()
    model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
    model.fc = nn.Linear(model.fc.in_features, 10)
    
    # Load trained weights
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    return model

# Preprocess the drawn image (same as validation transform)
def preprocess_image(image):
    transform = transforms.Compose([
        transforms.Resize(64),
        transforms.CenterCrop(64),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])
    return transform(image).unsqueeze(0)  # Add batch dimension

# Prediction function
def predict_digit(model, image_tensor):
    with torch.no_grad():
        outputs = model(image_tensor)
        _, predicted = torch.max(outputs.data, 1)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
    return predicted.item(), probabilities[0].numpy()

# Canvas drawing app
class DigitRecognizerApp:
    def __init__(self, root, model):
        self.root = root
        self.model = model
        self.setup_ui()
        
    def setup_ui(self):
        self.root.title("Dzongkha Digit Recognizer")
        
        # Canvas for drawing
        self.canvas = Canvas(self.root, width=300, height=300, bg="white", cursor="cross")
        self.canvas.pack()
        
        # Prediction label
        self.prediction_label = Label(self.root, text="Draw a Dzongkha digit (0-9)", font=("Helvetica", 16))
        self.prediction_label.pack()
        
        # Confidence bars
        self.confidence_frame = tk.Frame(self.root)
        self.confidence_frame.pack()
        self.confidence_bars = []
        for i in range(10):
            frame = tk.Frame(self.confidence_frame)
            frame.pack(side=tk.LEFT)
            Label(frame, text=str(i)).pack()
            bar = Canvas(frame, width=20, height=100, bg="white")
            bar.pack()
            self.confidence_bars.append(bar)
        
        # Buttons
        button_frame = tk.Frame(self.root)
        button_frame.pack()
        
        Button(button_frame, text="Predict", command=self.predict).pack(side=tk.LEFT, padx=5)
        Button(button_frame, text="Clear", command=self.clear_canvas).pack(side=tk.LEFT, padx=5)
        Button(button_frame, text="Save Image", command=self.save_image).pack(side=tk.LEFT, padx=5)
        
        # Drawing variables
        self.last_x, self.last_y = None, None
        self.canvas.bind("<B1-Motion>", self.paint)
        self.canvas.bind("<ButtonRelease-1>", self.reset)
        
    def paint(self, event):
        if self.last_x and self.last_y:
            self.canvas.create_line(self.last_x, self.last_y, event.x, event.y, 
                                   width=15, fill="black", capstyle=tk.ROUND, smooth=True)
        self.last_x = event.x
        self.last_y = event.y
        
    def reset(self, event):
        self.last_x, self.last_y = None, None
        
    def clear_canvas(self):
        self.canvas.delete("all")
        self.prediction_label.config(text="Draw a Dzongkha digit (0-9)")
        for bar in self.confidence_bars:
            bar.delete("all")
        
    def save_image(self):
        file_path = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=[("PNG files", "*.png"), ("All files", "*.*")],
            title="Save drawing as"
        )
        if file_path:
            # Save as PostScript first
            self.canvas.postscript(file=file_path + '.eps') 
            # Convert to PNG
            img = Image.open(file_path + '.eps')
            img.save(file_path, 'png')
            os.remove(file_path + '.eps')  # Clean up EPS file
            
    def predict(self):
        try:
            # Get canvas drawing as image
            ps = self.canvas.postscript(colormode='mono')
            img = Image.open(io.BytesIO(ps.encode('utf-8')))
            img = ImageOps.invert(img.convert('L'))
            
            # Preprocess and predict
            img_tensor = preprocess_image(img)
            prediction, confidences = predict_digit(self.model, img_tensor)
            
            # Update UI
            self.prediction_label.config(text=f"Prediction: {prediction}")
            
            # Update confidence bars
            max_conf = max(confidences) if max(confidences) > 0 else 1
            for i, bar in enumerate(self.confidence_bars):
                height = int((confidences[i] / max_conf) * 100)
                bar.delete("all")
                bar.create_rectangle(0, 100-height, 20, 100, fill="blue")
                bar.create_text(10, 110, text=f"{confidences[i]:.2f}")
                
        except Exception as e:
            self.prediction_label.config(text=f"Error: {str(e)}")

# Main execution
if __name__ == "__main__":
    # Initialize
    model = load_model()
    
    # Create GUI
    root = tk.Tk()
    app = DigitRecognizerApp(root, model)
    root.mainloop()