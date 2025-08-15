# ae/train_all_folds.py
import torch
from torch.utils.data import DataLoader
from ultralytics.nn.modules.ae import AutoEncoder
from gpr_dataset import GPRDataset
from pathlib import Path
import yaml

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BASE_DIR = Path("../data/UMay/yolo_gas")
SAVE_DIR = Path("./pretrained_folds4")
SAVE_DIR.mkdir(parents=True, exist_ok=True)

EPOCHS = 5
BATCH_SIZE = 16
LR = 1e-3

INTENSITY = 1
N_AUG = 5

"""
for fold in range(1, 11):
    fold_name = f"fold{fold:02d}"
    yaml_path = BASE_DIR / f"yolo_split_{fold_name}" / f"data_aug_I{INTENSITY:02d}_N{N_AUG:02d}_variance_filtered3.yaml"
"""

for fold in range(1, 6):
    fold_name = f"fold{fold:02d}"
    yaml_path = BASE_DIR / f"yolo_split_{fold_name}" / "data_aug_1var_5_gs.yaml"

    if not yaml_path.exists():
        print(f"[WARN] YAML not found: {yaml_path}")
        continue

    with open(yaml_path, "r") as f:
        data_cfg = yaml.safe_load(f)
    train_dir = BASE_DIR / f"yolo_split_{fold_name}" / data_cfg.get("train")
    if not train_dir.exists():
        print(f"[WARN] Invalid or missing train path: {train_dir}")
        continue

    print(f"\n📂 Training AE on {fold_name}: {train_dir}")
    dataset = GPRDataset(train_dir)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)

    ae = AutoEncoder(loss_type="hybrid").to(device)
    optimizer = torch.optim.Adam(ae.parameters(), lr=LR)

    for epoch in range(EPOCHS):
        ae.train()
        total_loss = 0.0
        total_psnr = 0.0
        count = 0

        for imgs in dataloader:
            imgs = imgs.to(device)
            _, loss = ae(imgs, return_loss=True)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            with torch.no_grad():
                psnr = ae.compute_psnr(imgs)
                total_psnr += psnr * imgs.size(0)
                count += imgs.size(0)

            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        avg_psnr = total_psnr / count
        print(f"[Fold {fold_name}] Epoch {epoch + 1:02d} | AE Loss: {avg_loss:.4f} | PSNR: {avg_psnr:.2f} dB")

    # 🔽 여기서 저장!
    torch.save(ae.state_dict(), SAVE_DIR / f"pretrained_ae_{fold_name}.pth")
    print(f"✅ Saved AE model: pretrained_ae_{fold_name}.pth")

