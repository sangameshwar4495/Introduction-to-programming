from flask import Flask, render_template, request
from PIL import Image
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from model import Cifar10CnnModel  # Assuming your model is saved in a separate Python file

app = Flask(__name__)

# Load the trained model
model = Cifar10CnnModel()
model.load_state_dict(torch.load('cifar10_model.pth'))
model.eval()

# Define image transformations
transform = transforms.Compose([
    transforms.Resize((32, 32)),  # Resize to fit the model input size
    transforms.ToTensor(),        # Convert PIL image to tensor
])

# Define the class labels
classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        file = request.files['file']
        image = Image.open(file)
        image_tensor = transform(image).unsqueeze(0)  # Add batch dimension

        # Perform inference
        with torch.no_grad():
            outputs = model(image_tensor)

        # Interpret the results
        _, predicted = torch.max(outputs, 1)
        predicted_class = classes[predicted.item()]

        return render_template('result.html', predicted_class=predicted_class)
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True,port=5500)