import os
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.models import vgg16, VGG16_Weights, mobilenet_v2, MobileNet_V2_Weights, efficientnet_b0, EfficientNet_B0_Weights
from torchmetrics.classification import MulticlassAccuracy, MulticlassJaccardIndex, MulticlassF1Score
import matplotlib.pyplot as plt
from torchmetrics import ConfusionMatrix
import pandas as pd
import seaborn as sns
import torchvision.models as models

# -----------------------------
# Model Definition
# -----------------------------
class MultiTaskNet(nn.Module):
    def __init__(self, num_seg_classes, num_cls_classes, backbone="mobilenet_v2"):
        super(MultiTaskNet, self).__init__()
        if backbone == "vgg16":
            base_model = vgg16(weights=VGG16_Weights.IMAGENET1K_V1).features
            encoder_channels = 512
        elif backbone == "mobilenet_v2":
            base_model = mobilenet_v2(weights=MobileNet_V2_Weights.IMAGENET1K_V1).features
            encoder_channels = 1280
        elif backbone == "efficientnet_b0":
            base_model = efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1).features
            encoder_channels = 1280
        else:
            raise ValueError("Unsupported backbone")

        self.encoder = base_model
        # Classification head
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(encoder_channels, num_cls_classes)
        )
        # Segmentation head
        self.segmentation = nn.Conv2d(encoder_channels, num_seg_classes, kernel_size=1)

    def forward(self, x):
        features = self.encoder(x)
        cls_out = self.classifier(features)
        seg_out = self.segmentation(features)
        seg_out = nn.functional.interpolate(seg_out, size=x.shape[2:], mode="bilinear", align_corners=False)
        return seg_out, cls_out
    
class MultiTaskUnet(nn.Module):
    def __init__(self, num_seg_classes, num_cls_classes, backbone="mobilenet_v2"):
        super(MultiTaskUnet, self).__init__()
        
        self.backbone_name = backbone

        if backbone == "vgg16":
            base_model = vgg16(weights=VGG16_Weights.IMAGENET1K_V1).features
            encoder_channels = 512
            self.encoder_stages = nn.ModuleList([
                base_model[:5],    # Block 1 (output 64 channels)
                base_model[5:10],  # Block 2 (output 128 channels)
                base_model[10:17], # Block 3 (output 256 channels)
                base_model[17:24], # Block 4 (output 512 channels)
                base_model[24:],   # Block 5 (output 512 channels)
            ])
            decoder_in_channels = [512, 512, 256, 128]
            decoder_out_channels = [256, 128, 64, 32]

        elif backbone == "mobilenet_v2":
            base_model = mobilenet_v2(weights=MobileNet_V2_Weights.IMAGENET1K_V1).features
            encoder_channels = 1280
            self.encoder_stages = nn.ModuleList([
                base_model[:2],    # 16 channels
                base_model[2:4],   # 24 channels
                base_model[4:7],   # 32 channels
                base_model[7:14],  # 96 channels
                base_model[14:],   # 1280 channels
            ])
            decoder_in_channels = [1280, 96, 32, 24]
            decoder_out_channels = [96, 32, 24, 16]

        else:
            raise ValueError("Unsupported backbone for this UNet-like architecture")

        self.encoder = base_model
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(encoder_channels, num_cls_classes)
        )
        
        # Segmentation decoder with skip connections
        self.decoder = nn.ModuleList()
        for i in range(len(decoder_in_channels) - 1):
            self.decoder.append(
                self.create_up_block(
                    decoder_in_channels[i], 
                    decoder_in_channels[i+1], 
                    decoder_out_channels[i+1]
                )
            )

        self.final_up_conv = nn.ConvTranspose2d(decoder_out_channels[-1], num_seg_classes, kernel_size=2, stride=2)

    def create_up_block(self, in_c, skip_c, out_c):
        return nn.Sequential(
            nn.ConvTranspose2d(in_c, in_c // 2, kernel_size=2, stride=2),
            nn.Conv2d(in_c // 2 + skip_c, out_c, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_c, out_c, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        # Encoder passes
        skip_features = []
        for stage in self.encoder_stages:
            x = stage(x)
            skip_features.append(x)
        
        # Classification branch
        cls_out = self.classifier(x)
        
        # Segmentation branch (U-Net decoder)
        seg_out = x
        for i, up_block in enumerate(self.decoder):
            seg_out = up_block[0](seg_out)
            # Concatenate with skip connection
            seg_out = torch.cat([seg_out, skip_features[len(skip_features) - 2 - i]], dim=1)
            seg_out = up_block[1:](seg_out)
        
        seg_out = self.final_up_conv(seg_out)
        
        return seg_out, cls_out

# -----------------------------
# Dataset
# -----------------------------
class NeuroDataset(Dataset):
    def __init__(self, image_root, mask_root, transform=None):
        self.image_root = image_root
        self.mask_root = mask_root
        self.transform = transform
        self.classes = ["normal", "parkinson"]
        self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(self.classes)}
        self.samples = []

        for cls_name in self.classes:
            img_dir = os.path.join(image_root, cls_name)
            mask_dir = os.path.join(mask_root, cls_name)
            if not os.path.exists(img_dir):
                continue
            for fname in os.listdir(img_dir):
                if fname.endswith((".png", ".jpg", ".jpeg")):
                    img_path = os.path.join(img_dir, fname)
                    mask_path = os.path.join(mask_dir, fname) if os.path.exists(os.path.join(mask_dir, fname)) else None
                    self.samples.append((img_path, mask_path, self.class_to_idx[cls_name]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, mask_path, cls_label = self.samples[idx]
        image = Image.open(img_path).convert("RGB")
        if mask_path and os.path.exists(mask_path):
            mask = Image.open(mask_path)
        else:
            mask = Image.fromarray(np.zeros((224, 224), dtype=np.uint8))

        if self.transform:
            image = self.transform(image)
            mask = transforms.Resize((224, 224))(mask)
            mask = torch.tensor(np.array(mask), dtype=torch.long)

        return image, mask, cls_label

# -----------------------------
# Training Script for Parkinson’s
# -----------------------------
def train_parkinsons(backbone="efficientnet_b0", num_epochs=30, batch_size=4, model_class=MultiTaskNet):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_seg_classes = 2  # background + ROI
    num_cls_classes = 2  # Normal / Parkinson
    class_names = ["normal", "parkinson"]
    
    model = model_class(num_seg_classes, num_cls_classes, backbone=backbone).to(device)

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor()
    ])

    dataset = NeuroDataset(
        image_root="neurodegenerative_disease_data/parkinsons_mri_dataset",
        mask_root="neurodegenerative_disease_data/parkinsons_mask",
        transform=transform
    )

    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    criterion_cls = nn.CrossEntropyLoss()
    criterion_seg = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    results_data = []
    best_val_miou = 0.0

    for epoch in range(num_epochs):
        # ---------------- Train ----------------
        model.train()
        running_loss = 0.0
        train_cls_acc = MulticlassAccuracy(task='multiclass', num_classes=num_cls_classes).to(device)
        train_seg_miou = MulticlassJaccardIndex(task='multiclass', num_classes=num_seg_classes).to(device)
        train_seg_dice = MulticlassF1Score(task='multiclass', num_classes=num_seg_classes, average="macro").to(device)
        
        for images, seg_labels, cls_labels in train_loader:
            images, seg_labels, cls_labels = images.to(device), seg_labels.to(device), cls_labels.to(device)

            seg_out, cls_out = model(images)
            loss_cls = criterion_cls(cls_out, cls_labels)
            loss_seg = criterion_seg(seg_out, seg_labels)
            loss = loss_cls + loss_seg

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            cls_preds = cls_out.argmax(dim=1)
            seg_preds = seg_out.argmax(dim=1)
            train_cls_acc.update(cls_preds, cls_labels)
            train_seg_miou.update(seg_preds, seg_labels)
            train_seg_dice.update(seg_preds, seg_labels)

        train_loss = running_loss / len(train_loader.dataset)
        train_acc = train_cls_acc.compute().item()
        train_miou = train_seg_miou.compute().item()
        train_dice = train_seg_dice.compute().item()

        # ---------------- Validation ----------------
        model.eval()
        val_loss = 0.0
        val_cls_acc = MulticlassAccuracy(task='multiclass', num_classes=num_cls_classes).to(device)
        val_seg_miou = MulticlassJaccardIndex(task='multiclass', num_classes=num_seg_classes).to(device)
        val_seg_dice = MulticlassF1Score(task='multiclass', num_classes=num_seg_classes, average="macro").to(device)
        
        all_val_preds = []
        all_val_labels = []

        with torch.no_grad():
            for images, seg_labels, cls_labels in val_loader:
                images, seg_labels, cls_labels = images.to(device), seg_labels.to(device), cls_labels.to(device)

                seg_out, cls_out = model(images)
                loss_cls = criterion_cls(cls_out, cls_labels)
                loss_seg = criterion_seg(seg_out, seg_labels)
                val_loss += (loss_cls + loss_seg).item() * images.size(0)

                cls_preds = cls_out.argmax(dim=1)
                seg_preds = seg_out.argmax(dim=1)

                val_cls_acc.update(cls_preds, cls_labels)
                val_seg_miou.update(seg_preds, seg_labels)
                val_seg_dice.update(seg_preds, seg_labels)
                
                all_val_preds.extend(cls_preds.cpu().numpy())
                all_val_labels.extend(cls_labels.cpu().numpy())

        val_loss = val_loss / len(val_loader.dataset)
        val_acc = val_cls_acc.compute().item()
        val_miou = val_seg_miou.compute().item()
        val_dice = val_seg_dice.compute().item()

        results_data.append({
            'epoch': epoch + 1,
            'train_loss': train_loss, 'train_acc': train_acc, 'train_miou': train_miou, 'train_dice': train_dice,
            'val_loss': val_loss, 'val_acc': val_acc, 'val_miou': val_miou, 'val_dice': val_dice
        })

        if val_miou > best_val_miou:
            best_val_miou = val_miou
            torch.save(model.state_dict(), f"{backbone}_parkinsons_best_multitask_model.pth")
            print("=> Best model saved!")

        print(f"Epoch [{epoch+1}/{num_epochs}] "
              f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, "
              f"Train mIoU: {train_miou:.4f}, Train Dice: {train_dice:.4f} | "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, "
              f"Val mIoU: {val_miou:.4f}, Val Dice: {val_dice:.4f}")

    # Save results to CSV
    df = pd.DataFrame(results_data)
    df.to_csv(f"{backbone}_parkinsons_training_metrics.csv", index=False)
    print(f"Training metrics saved to {backbone}_parkinsons_training_metrics.csv")
    
    # ---------------- Confusion Matrix ----------------
    cm_metric = ConfusionMatrix(task='multiclass', num_classes=num_cls_classes).to(device)
    cm = cm_metric(torch.tensor(all_val_preds).to(device), torch.tensor(all_val_labels).to(device))
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm.cpu().numpy(), annot=True, fmt="d", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title(f"Confusion Matrix for {backbone} Model (Parkinson's)")
    plt.tight_layout()
    plt.savefig(f"{backbone}_parkinsons_confusion_matrix.png")
    plt.show()
    
    # ---------------- Plot curves ----------------
    plt.figure(figsize=(14,6))
    plt.subplot(1,2,1)
    plt.plot(df['epoch'], df['train_loss'], label="Train Loss")
    plt.plot(df['epoch'], df['val_loss'], label="Val Loss")
    plt.xlabel("Epoch"); plt.ylabel("Loss")
    plt.legend(); plt.title("Loss Curve")

    plt.subplot(1,2,2)
    plt.plot(df['epoch'], df['train_acc'], label="Train Acc")
    plt.plot(df['epoch'], df['val_acc'], label="Val Acc")
    plt.plot(df['epoch'], df['train_miou'], label="Train mIoU")
    plt.plot(df['epoch'], df['val_miou'], label="Val mIoU")
    plt.plot(df['epoch'], df['train_dice'], label="Train Dice")
    plt.plot(df['epoch'], df['val_dice'], label="Val Dice")
    plt.xlabel("Epoch"); plt.ylabel("Metric")
    plt.legend(); plt.title(f"Classification + Segmentation Metrics (Parkinson's)")
    plt.tight_layout()
    plt.savefig(f"{backbone}_parkinsons_training_curves.png")
    plt.show()

if __name__ == "__main__":
    # Test original model with all three backbones
    print("--- Training with standard MultiTaskNet for Parkinson's ---")
    backbones_to_train = ["efficientnet_b0", "vgg16", "mobilenet_v2"]
    for backbone in backbones_to_train:
        print(f"\n--- Starting training with backbone: {backbone} ---")
        train_parkinsons(backbone=backbone, num_epochs=30, batch_size=4, model_class=MultiTaskNet)

    # Test the new U-Net-like model
    print("\n--- Training with U-Net-like MultiTaskNet for Parkinson's ---")
    unet_backbones = ["vgg16", "mobilenet_v2"]
    for backbone in unet_backbones:
        print(f"\n--- Starting training with U-Net-like model and backbone: {backbone} ---")
        train_parkinsons(backbone=backbone, num_epochs=30, batch_size=4, model_class=MultiTaskUnet)