import os
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.models import vgg16, VGG16_Weights, mobilenet_v2, MobileNet_V2_Weights, efficientnet_b0, EfficientNet_B0_Weights
from torchmetrics.classification import MulticlassAccuracy, MulticlassPrecision, MulticlassRecall, MulticlassF1Score, MulticlassJaccardIndex


# -----------------------------
# Model Definition (patched)
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

        # ---- Classification head ----
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(encoder_channels, num_cls_classes)
        )

        # ---- Segmentation head ----
        self.segmentation = nn.Conv2d(encoder_channels, num_seg_classes, kernel_size=1)

    def forward(self, x):
        features = self.encoder(x)

        # Classification
        cls_out = self.classifier(features)

        # Segmentation (upsample to input size)
        seg_out = self.segmentation(features)
        seg_out = nn.functional.interpolate(
            seg_out, size=x.shape[2:], mode="bilinear", align_corners=False
        )

        return seg_out, cls_out



# -----------------------------
# Dataset
# -----------------------------
class NeuroDataset(Dataset):
    def __init__(self, image_root, mask_root, disease="alzheimers", transform=None, num_seg_classes=3):
        self.image_root = image_root
        self.mask_root = mask_root
        self.transform = transform
        self.num_seg_classes = num_seg_classes
        self.disease = disease
        self.samples = []

        # Build label dictionary
        if disease == "alzheimers":
            classes = ["NonDemented", "VeryMildDemented", "MildDemented", "ModerateDemented"]
        elif disease == "parkinsons":
            classes = ["normal", "parkinson"]
        else:
            raise ValueError("Unknown disease type")

        self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(classes)}

        # Collect image + mask paths
        for cls_name in classes:
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

        # Load MRI image
        image = Image.open(img_path).convert("RGB")

        # Load mask if available, otherwise create dummy mask
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
# Training Script
# -----------------------------
def train_alzheimer(backbone="mobilenet_v2", num_epochs=30, batch_size=4):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    num_seg_classes = 3
    num_cls_classes = 4  # Alzheimer stages

    model = MultiTaskNet(num_seg_classes, num_cls_classes, backbone=backbone).to(device)

    # ------------------- Transforms -------------------
    transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor()
])


    # ------------------- Dataset -------------------
    dataset = NeuroDataset(
        image_root="NEURODEGENERATIVE_DATA/alzheimer_mri_dataset/OriginalDataset",
        mask_root="NEURODEGENERATIVE_DATA/alzheimer_mask/OriginalDataset",
        disease="alzheimers",
        transform=transform
    )

    # Split into train and validation (80-20 split)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # ------------------- Loss & Optimizer -------------------
    criterion_cls = nn.CrossEntropyLoss()
    criterion_seg = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    # ------------------- Metrics -------------------
    acc = MulticlassAccuracy(num_classes=num_cls_classes).to(device)
    prec = MulticlassPrecision(num_classes=num_cls_classes, average="macro").to(device)
    rec = MulticlassRecall(num_classes=num_cls_classes, average="macro").to(device)
    f1 = MulticlassF1Score(num_classes=num_cls_classes, average="macro").to(device)
    miou = MulticlassJaccardIndex(num_classes=num_seg_classes).to(device)

    for epoch in range(num_epochs):
        model.train()
        for images, seg_labels, cls_labels in train_loader:
            images, seg_labels, cls_labels = images.to(device), seg_labels.to(device), cls_labels.to(device)

            seg_out, cls_out = model(images)
            loss_cls = criterion_cls(cls_out, cls_labels)
            loss_seg = criterion_seg(seg_out, seg_labels)
            loss = loss_cls + loss_seg

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # ---------------- Validation ----------------
        model.eval()
        acc.reset(); prec.reset(); rec.reset(); f1.reset(); miou.reset()
        with torch.no_grad():
            for images, seg_labels, cls_labels in val_loader:
                images, seg_labels, cls_labels = images.to(device), seg_labels.to(device), cls_labels.to(device)
                seg_out, cls_out = model(images)

                cls_preds = cls_out.argmax(dim=1)
                seg_preds = seg_out.argmax(dim=1)

                acc.update(cls_preds, cls_labels)
                prec.update(cls_preds, cls_labels)
                rec.update(cls_preds, cls_labels)
                f1.update(cls_preds, cls_labels)
                miou.update(seg_preds, seg_labels)

        print(f"Epoch [{epoch+1}/{num_epochs}] "
              f"Acc: {acc.compute():.4f}, "
              f"Precision: {prec.compute():.4f}, "
              f"Recall: {rec.compute():.4f}, "
              f"F1: {f1.compute():.4f}, "
              f"mIoU: {miou.compute():.4f}")


if __name__ == "__main__":
    train_alzheimer(backbone="efficientnet_b0", num_epochs=30, batch_size=4)
