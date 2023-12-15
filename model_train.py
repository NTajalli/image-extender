import os
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.utils import save_image
from PIL import Image
import matplotlib.pyplot as plt
from torchvision.models import vgg19, VGG19_Weights
from torch.nn import functional as F

# Function to load images from a folder
def load_images_from_folder(folder, target_size=(256, 256)):
    images = []
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder, filename))
        if img is not None:
            img = cv2.resize(img, target_size)
            images.append(img)
    return images

# Custom Dataset Class with Augmentation
class GANDataset(Dataset):
    def __init__(self, folder_path, target_size=(256, 256), transform=None):
        self.image_files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]
        self.target_size = target_size
        self.transform = transform

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img = cv2.imread(self.image_files[idx])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, self.target_size)

        zoomed = generate_zoomed_images(img)[0]

        if self.transform:
            original = self.transform(Image.fromarray(img))
            zoomed = self.transform(Image.fromarray(zoomed))

        return original, zoomed

# Add Self-Attention Layer
class SelfAttention(nn.Module):
    """ Self attention Layer"""
    def __init__(self, in_dim):
        super(SelfAttention, self).__init__()
        self.chanel_in = in_dim
        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        m_batchsize, C, width, height = x.size()
        proj_query = self.query_conv(x).view(m_batchsize, -1, width*height).permute(0, 2, 1)
        proj_key = self.key_conv(x).view(m_batchsize, -1, width*height)
        energy = torch.bmm(proj_query, proj_key)
        attention = F.softmax(energy, dim=-1)
        proj_value = self.value_conv(x).view(m_batchsize, -1, width*height)

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(m_batchsize, C, width, height)

        out = self.gamma*out + x
        return out

# Define the Generator and Discriminator in PyTorch
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.init_size = 8  # Initial size of features

        # Network to process the input image
        self.input_img_processor = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
        )

        # Network to process the noise vector
        self.noise_processor = nn.Sequential(
            nn.Linear(100, 128 * self.init_size ** 2),
            nn.BatchNorm1d(128 * self.init_size ** 2),
            nn.ReLU(True)
        )

        # Convolutional layers for upsampling
        self.upsample_blocks = nn.Sequential(
            nn.Conv2d(256, 128, 3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.Upsample(scale_factor=2),  # Upsample to 16x16
            nn.Conv2d(128, 128, 3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.Upsample(scale_factor=2),  # Upsample to 32x32
            nn.Conv2d(128, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.Upsample(scale_factor=2),  # Upsample to 64x64
            nn.Upsample(scale_factor=2),  # Upsample to 128x128
            nn.Conv2d(64, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.Upsample(scale_factor=2),  # Upsample to 256x256
            nn.Upsample(scale_factor=2),  # Upsample to 512x512
            nn.Conv2d(64, 3, 3, stride=1, padding=1),
            nn.Tanh()  # Output layer
        )

    def forward(self, z, input_img):
        # Process the input image
        img_features = self.input_img_processor(input_img)

        # Process the noise vector
        z = self.noise_processor(z)
        z = z.view(z.size(0), 128, self.init_size, self.init_size)

        # Resize z to match the spatial dimensions of img_features
        z = F.interpolate(z, size=(img_features.size(2), img_features.size(3)), mode='bilinear', align_corners=False)

        # Concatenate the features from noise and input image
        combined = torch.cat([z, img_features], 1)

        # Generate the image
        img = self.upsample_blocks(combined)
        return img


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        # Note: The number of input channels is 6 because we concatenate the generated image and the resized original image.
        self.model = nn.Sequential(
            nn.Conv2d(6, 64, 4, stride=2, padding=1),  # Input: 512x512, Output: 256x256
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(0.25),

            nn.Conv2d(64, 128, 4, stride=2, padding=1),  # Input: 256x256, Output: 128x128
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(0.25),

            nn.Conv2d(128, 256, 4, stride=2, padding=1),  # Input: 128x128, Output: 64x64
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(0.25),

            nn.Conv2d(256, 512, 4, stride=2, padding=1),  # Input: 64x64, Output: 32x32
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(0.25),

            nn.Conv2d(512, 1024, 4, stride=2, padding=1),  # Input: 32x32, Output: 16x16
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(0.25),

            nn.Conv2d(1024, 2048, 4, stride=2, padding=1),  # Input: 16x16, Output: 8x8
            nn.BatchNorm2d(2048),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(0.25),

            nn.Conv2d(2048, 4096, 4, stride=2, padding=1),  # Input: 8x8, Output: 4x4
            nn.BatchNorm2d(4096),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(0.25),

            nn.Conv2d(4096, 1, 4, stride=1, padding=0),  # Input: 4x4, Output: 1x1
            nn.AdaptiveAvgPool2d(1),
            nn.Sigmoid()
        )

    def forward(self, gen_img, input_img):
        # Resize input_img to match the size of gen_img (512x512)
        input_img_resized = F.interpolate(input_img, size=(512, 512), mode='bilinear', align_corners=False)

        # Concatenate the generated image and the resized input image
        combined_img = torch.cat([gen_img, input_img_resized], 1)

        # Pass the combined image through the discriminator model
        validity = self.model(combined_img)
        return validity.view(-1, 1)



class VGGPerceptualLoss(nn.Module):
    def __init__(self):
        super(VGGPerceptualLoss, self).__init__()
        vgg_weights = VGG19_Weights.IMAGENET1K_V1
        self.vgg = vgg19(weights=vgg_weights).features[:26].eval()
        for param in self.vgg.parameters():
            param.requires_grad = False

    def forward(self, gen_images, real_images):
        vgg_gen = self.vgg(gen_images)
        vgg_real = self.vgg(real_images)
        perceptual_loss = F.mse_loss(vgg_gen, vgg_real)
        return perceptual_loss

def generate_zoomed_images(image, zoom_factor=1.5, num_zooms=1):
    height, width, _ = image.shape
    zoomed_images = []

    for i in range(num_zooms):
        new_width = int(width / (zoom_factor ** (i + 1)))
        new_height = int(height / (zoom_factor ** (i + 1)))
        start_x = (width - new_width) // 2
        start_y = (height - new_height) // 2

        cropped_img = image[start_y:start_y+new_height, start_x:start_x+new_width]
        resized_img = cv2.resize(cropped_img, (width, height), interpolation=cv2.INTER_LINEAR)

        zoomed_images.append(resized_img)

    return zoomed_images

# Training Loop with Model Checkpointing and Enhanced Visualization
def train_gan(generator, discriminator, dataloader, epochs, device):
    adversarial_loss = nn.BCELoss()
    perceptual_loss_criterion = VGGPerceptualLoss().to(device)
    l1_loss_criterion = nn.L1Loss()

    optimizer_G = torch.optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))
    output_dir = './gan_output_images'
    scaler = GradScaler()  # Initialize gradient scaler for mixed precision
    some_frequency = 1

    for epoch in range(epochs):
        for i, (real_images, zoomed_images) in enumerate(dataloader):
            real_images, zoomed_images = real_images.to(device), zoomed_images.to(device)
            batch_size = real_images.size(0)
            real_images_resized = F.interpolate(real_images, size=(512, 512), mode='bilinear', align_corners=False)

            # Forward pass for discriminator
            optimizer_D.zero_grad()
            with autocast():
                gen_images = generator(torch.randn(batch_size, 100, device=device), real_images_resized)
                real_loss = adversarial_loss(discriminator(real_images_resized, real_images_resized), torch.ones(batch_size, 1, device=device))
                fake_loss = adversarial_loss(discriminator(gen_images.detach(), real_images_resized), torch.zeros(batch_size, 1, device=device))
                d_loss = (real_loss + fake_loss) / 2
            scaler.scale(d_loss).backward()
            scaler.step(optimizer_D)

            # Forward pass for generator
            optimizer_G.zero_grad()
            with autocast():
                g_loss_adv = adversarial_loss(discriminator(gen_images, real_images_resized), torch.ones(batch_size, 1, device=device))
                g_loss_l1 = l1_loss_criterion(gen_images, real_images_resized)
                g_loss_perc = perceptual_loss_criterion(gen_images, real_images_resized)
                g_loss_total = g_loss_adv + 0.1 * g_loss_l1 + 0.01 * g_loss_perc
            scaler.scale(g_loss_total).backward()
            scaler.step(optimizer_G)

            scaler.update()  # Update scaler

            # Logging and saving images
            if i % some_frequency == 0:
                print(f"[Epoch {epoch}/{epochs}] [Batch {i}/{len(dataloader)}] [D loss: {d_loss.item()}] [G loss: {g_loss_total.item()}]")
                save_image(real_images_resized.data, os.path.join(output_dir, f"epoch_{epoch}_batch_{i}_real.png"), nrow=5, normalize=True)
                save_image(gen_images.data, os.path.join(output_dir, f"epoch_{epoch}_batch_{i}_generated.png"), nrow=5, normalize=True)
                
        
def generate_test_image(generator, device, latent_dim=100):
    z = torch.randn(1, latent_dim).to(device)
    with torch.no_grad():
        generated_img = generator(z).cpu()
    return generated_img

# Load dataset for training with updated transformations
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

train_dataset = GANDataset(folder_path='./test_images', target_size=(512, 512), transform=transform)
train_dataloader = DataLoader(train_dataset, batch_size=6, shuffle=True, num_workers=4)

# Initialize generator and discriminator
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
generator = Generator().to(device)
discriminator = Discriminator().to(device)

# Start training
train_gan(generator, discriminator, train_dataloader, epochs=200, device=device)

# Generate a test image
generated_img = generate_test_image(generator, device)

# Convert generated image for visualization
generated_img = generated_img.squeeze(0).permute(1, 2, 0)
generated_img = (generated_img * 0.5 + 0.5).numpy()


