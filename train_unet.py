import os
import glob
from sys import platform

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np
from PIL import Image

# ---------------------------------------------------------
# 1. Dataset Loader
# ---------------------------------------------------------
class SynthSegDataset(Dataset):
    def __init__(self, root_dir='outputs', transform=None):
        self.rgb_dir = os.path.join(root_dir, 'rgb')
        self.mask_dir = os.path.join(root_dir, 'mask')
        self.rgb_files = sorted(glob.glob(os.path.join(self.rgb_dir, '*.png')))
        self.transform = transform
        
        if len(self.rgb_files) == 0:
            print(f"Warning: No images found in {self.rgb_dir}. Did you run the generator?")

    def __len__(self):
        return len(self.rgb_files)

    def __getitem__(self, idx):
        rgb_path = self.rgb_files[idx]
        basename = os.path.basename(rgb_path)
        mask_path = os.path.join(self.mask_dir, basename)

        # Load RGB image
        img = Image.open(rgb_path).convert('RGB')
        # Load mask image
        mask_img = Image.open(mask_path).convert('RGB')

        # Convert to numpy arrays
        # Image shape: (H, W, 3) -> (3, H, W)
        # We also scale image to [0, 1]
        img_np = np.array(img, dtype=np.float32) / 255.0
        img_np = np.transpose(img_np, (2, 0, 1))

        # Extracted Mask Logic
        # Parse visual colored mask to Class IDs
        mask_array = np.array(mask_img, dtype=np.int32)
        semantic_mask = np.zeros(mask_array.shape[:2], dtype=np.int64)
        
        R = mask_array[:, :, 0]
        G = mask_array[:, :, 1]
        B = mask_array[:, :, 2]
        
        # 1. Sky = Green (R<50, G>200, B<50)
        sky = (R < 50) & (G > 200) & (B < 50)
        semantic_mask[sky] = 1
        
        # 2. Ground = Yellow (R>200, G>200, B<50)
        ground = (R > 200) & (G > 200) & (B < 50)
        semantic_mask[ground] = 2
        
        # 3. House = Orange (R>200, 100<G<200, B<50)
        house = (R > 200) & (G > 80) & (G < 200)
        semantic_mask[house] = 3
        
        # 4. Ego Vehicle = Blue (R<50, G<50, B>200)
        ego = (R < 50) & (G < 50) & (B > 200)
        semantic_mask[ego] = 4
        
        # 5. Cars = Red/Magenta (R>200, G<50, B var)
        cars = (R > 200) & (G < 50)
        semantic_mask[cars] = 5

        img_tensor = torch.from_numpy(img_np)
        mask_tensor = torch.from_numpy(semantic_mask)

        return img_tensor, mask_tensor

# ---------------------------------------------------------
# 2. Simple UNET Model
# ---------------------------------------------------------
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)

class SimpleUNet(nn.Module):
    def __init__(self, num_classes=6):
        super().__init__()
        # Encoder
        self.enc1 = DoubleConv(3, 64)
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = DoubleConv(64, 128)
        self.pool2 = nn.MaxPool2d(2)
        
        # Bottleneck
        self.bot = DoubleConv(128, 256)
        
        # Decoder
        self.up1 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec1 = DoubleConv(256, 128)
        self.up2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec2 = DoubleConv(128, 64)
        
        self.out_conv = nn.Conv2d(64, num_classes, kernel_size=1)

    def forward(self, x):
        e1 = self.enc1(x)
        p1 = self.pool1(e1)
        
        e2 = self.enc2(p1)
        p2 = self.pool2(e2)
        
        b = self.bot(p2)
        
        u1 = self.up1(b)
        d1 = self.dec1(torch.cat([u1, e2], dim=1))
        
        u2 = self.up2(d1)
        d2 = self.dec2(torch.cat([u2, e1], dim=1))
        
        return self.out_conv(d2)

# ---------------------------------------------------------
# 3. Training Loop
# ---------------------------------------------------------
def train():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Maximum classes observed (Sky=1, Ground=2, House=3, Ego=4, Cars=5) + Padding/Void(0)
    num_classes = 6
    
    model = SimpleUNet(num_classes=num_classes).to(device)
    
    # Check dataset
    dataset = SynthSegDataset(root_dir='outputs')
    if len(dataset) == 0:
        print("Cannot start training without dataset.")
        return

    # Train/Val split (80% / 20%)
    val_size = int(0.2 * len(dataset))
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    # Batch size is 1 or 2 here to accommodate large image sizes like 1600x900
    train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=2, shuffle=False)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    epochs = 10
    print("Starting training...")
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0

        for batch_idx, (images, masks) in enumerate(train_loader):
            images = images.to(device)
            masks = masks.to(device)
            
            optimizer.zero_grad()
            
            preds = model(images)
            loss = criterion(preds, masks)
            
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            
            if batch_idx % 5 == 0:
                print(f"Epoch [{epoch+1}/{epochs}] Train Batch {batch_idx} Loss: {loss.item():.4f}")
                
        avg_train_loss = train_loss / max(1, len(train_loader))
        
        # Validation Loop
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, masks in val_loader:
                images = images.to(device)
                masks = masks.to(device)
                preds = model(images)
                loss = criterion(preds, masks)
                val_loss += loss.item()
                
        avg_val_loss = val_loss / max(1, len(val_loader))
        print(f"--- Epoch {epoch+1} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} ---")
        
    print("Training finished. Saving model...")
    torch.save(model.state_dict(), "semantic_unet_synth.pth")
    print("Model saved to semantic_unet_synth.pth")

if __name__ == '__main__':
    train()
