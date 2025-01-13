import os
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from models import ColorizingNet
from utils import ColirizationDataset, init_model
from PIL import Image
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.cuda.amp as amp
import torch.nn.functional as F


if __name__ == "__main__":
    batch_size = 16
    num_epochs = 150
    learning_rate = 2e-4

    # Dataloaders
    transformation_train = transforms.Compose([
        transforms.Resize((256, 256), Image.BICUBIC),
        transforms.ToTensor(),
        transforms.RandomHorizontalFlip()
    ])
    transformation_val = transforms.Compose([
        transforms.Resize((256, 256), Image.BICUBIC),
        transforms.ToTensor(),
    ])
    train_dir = 'data/img_color/data/train_color/'
    val_dir = 'data/img_color/data/test_color/'
    imgs_train = [os.path.join(train_dir, img) for img in os.listdir(train_dir)]
    imgs_val = [os.path.join(val_dir, img) for img in os.listdir(val_dir)]
    train_dataset = ColirizationDataset(imgs_train, transformation_train)
    val_dataset = ColirizationDataset(imgs_val, transformation_val)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = ColorizingNet(device)
    init_model(model, device)

    # Load pretrained weights
    pretrained_path = "pretrained_colorization_model.pth"
    if os.path.exists(pretrained_path):
        model.load_state_dict(torch.load(pretrained_path))
        print("Loaded pretrained model weights.")

    # Scheduler and early stopping setup
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, betas=(0.5, 0.999))
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)
    early_stopping_patience = 10
    best_val_loss = float("inf")
    early_stopping_counter = 0

    # Mixed precision training setup
    scaler = amp.GradScaler()

    # Training loop
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for gray_imgs, color_imgs in train_loader:
            gray_imgs = gray_imgs.to(device)
            color_imgs = color_imgs.to(device)

            optimizer.zero_grad()
            with amp.autocast():
                generated_ab = model(gray_imgs)

                # Resize generated output to match the target size
                generated_ab = F.interpolate(generated_ab, size=(color_imgs.size(2), color_imgs.size(3)),
                                             mode='bilinear', align_corners=False)

                # Debugging shapes
                print("Resized Output shape (generated_ab):", generated_ab.shape)
                print("Target shape (color_imgs):", color_imgs.shape)

                loss = model.pixelwise_loss(generated_ab, color_imgs)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item()

        avg_train_loss = running_loss / len(train_loader)
        print(f"Epoch [{epoch+1}/{num_epochs}], Training Loss: {avg_train_loss:.4f}")

        # Validation loop
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for gray_imgs, color_imgs in val_loader:
                gray_imgs = gray_imgs.to(device)
                color_imgs = color_imgs.to(device)

                with amp.autocast():
                    generated_ab = model(gray_imgs)

                    # Resize output to match target size
                    generated_ab = F.interpolate(generated_ab, size=(color_imgs.size(2), color_imgs.size(3)),
                                                 mode='bilinear', align_corners=False)

                    loss = model.pixelwise_loss(generated_ab, color_imgs)
                val_loss += loss.item()

        avg_val_loss = val_loss / len(val_loader)
        print(f"Validation Loss: {avg_val_loss:.4f}")

        # Scheduler step
        scheduler.step(avg_val_loss)

        # Early stopping
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), "best_colorization_model.pth")
            print("Saved best model!")
            early_stopping_counter = 0
        else:
            early_stopping_counter += 1
            if early_stopping_counter >= early_stopping_patience:
                print("Early stopping triggered.")
                break

        # Save model checkpoint periodically
        if (epoch + 1) % 10 == 0:
            torch.save(model.state_dict(), f"colorization_checkpoint_epoch_{epoch+1}.pth")
            print(f"Checkpoint saved at epoch {epoch+1}.")
