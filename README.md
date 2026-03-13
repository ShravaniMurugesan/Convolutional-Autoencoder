# Convolutional Autoencoder for Image Denoising

## AIM

To develop a convolutional autoencoder for image denoising application.

## Problem Statement and Dataset
Noise is a common issue in real-world image data, which affects performance in image analysis tasks. An autoencoder can be trained to remove noise from images, effectively learning compressed representations that help in reconstruction. The MNIST dataset (28x28 grayscale handwritten digits) will be used for this task. Gaussian noise will be added to simulate real-world noisy data.

## DESIGN STEPS

### STEP 1:
Load MNIST dataset and convert to tensors.

### STEP 2:
Apply Gaussian noise to images for training.

### STEP 3:
Design encoder-decoder architecture for reconstruction.

### STEP 4:
Use MSE loss to measure reconstruction quality.

### STEP 5:
Train autoencoder using Adam optimizer efficiently.

### STEP 6:
Evaluate model on noisy and clean images.

### STEP 7:
Visualize results comparing original, noisy, denoised versions.

### STEP 8:
Improve performance by tuning hyperparameters carefully.


## PROGRAM
### Name: SHRAVANI M
### Register Number: 212224230263


```
class DenoisingAutoencoder(nn.Module):
    def __init__(self):
        super(DenoisingAutoencoder, self).__init__()
        # Define your layers here
        # Example:
        self.encoder = nn.Sequential(
            nn.Linear(28 * 28, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 12),
            nn.ReLU(),
            nn.Linear(12, 3)
        )
        self.decoder = nn.Sequential(
            nn.Linear(3, 12),
            nn.ReLU(),
            nn.Linear(12, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 28 * 28),
            nn.Sigmoid()  # For reconstruction, sigmoid is often used
        )
    def forward(self, x):
        # Include your code here
        x = x.view(-1, 28*28)  # Flatten the input image
        x = self.encoder(x)
        x = self.decoder(x)
        x = x.view(-1, 1, 28, 28)  # Reshape to image dimensions
        return x

#Initialize model, loss function and optimizer
model = DenoisingAutoencoder().to(device)
summary(model, (1, 28, 28))
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

def train(model, loader, criterion, optimizer, epochs=5):
    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        for data in loader:
            inputs, _ = data
            inputs = inputs.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
```

## OUTPUT

### Model Summary

<img width="842" height="620" alt="image" src="https://github.com/user-attachments/assets/6e258286-78e6-428a-bad9-0a4ed9c4f3ab" />


### Original vs Noisy Vs Reconstructed Image

<img width="731" height="602" alt="image" src="https://github.com/user-attachments/assets/33ef4d73-aaf4-415a-af66-e3c7ad1d5401" />
<img width="1207" height="627" alt="image" src="https://github.com/user-attachments/assets/eb240b08-9f7e-40f5-91cf-0ef9147aac1e" />




## RESULT
A convolutional autoencoder for image denoising application is developed successfully.
