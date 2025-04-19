import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image, ImageTk
import tkinter as tk
from tkinter import filedialog

class MonCNN(nn.Module):
    def __init__(self, num_classes):
        super(MonCNN, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 16 * 16, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.fc_layers(x)
        return x

# Configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_classes = 3

# Chargement du modèle
model = MonCNN(num_classes)
model.load_state_dict(torch.load("modele_cnn.pth", map_location=device))
model.to(device)
model.eval()

# Liste des classes (doit correspondre à l'ordre de ImageFolder)
classes = ['metal', 'papier', 'plastique']

# Couleurs associées à chaque classe
couleurs = {
    'papier': 'blue',
    'metal': 'gray',
    'plastique': 'green'
}

# Pré-traitement des images
transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor()
])

def predire_image():
    chemin = filedialog.askopenfilename()
    if chemin:
        # Affichage de l’image
        image_pil = Image.open(chemin).convert('RGB')
        image_affiche = image_pil.resize((200, 200))
        photo = ImageTk.PhotoImage(image_affiche)
        label_img.config(image=photo)
        label_img.image = photo

        # Préparation image pour modèle
        image_tensor = transform(image_pil).unsqueeze(0).to(device)

        with torch.no_grad():
            output = model(image_tensor)
            probabilities = torch.softmax(output, dim=1)
            confidence, prediction = torch.max(probabilities, 1)

            classe = classes[prediction.item()]
            pourcentage = confidence.item() * 100

        label_resultat.config(
            text=f"Classe : {classe} ({pourcentage:.1f}%)",
            fg=couleurs[classe]
        )

# Interface graphique
fenetre = tk.Tk()
fenetre.title("Prédiction CNN")
fenetre.geometry("300x300")

btn = tk.Button(fenetre, text="Choisir une image", command=predire_image)
btn.pack()

label_img = tk.Label(fenetre)
label_img.pack()

label_resultat = tk.Label(fenetre, text="", font=("Arial", 14))
label_resultat.pack()

fenetre.mainloop()
