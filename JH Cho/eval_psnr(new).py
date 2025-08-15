# ae/eval_psnr_all_folds.py
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from ultralytics.nn.modules.ae import AutoEncoder
from gpr_dataset import GPRDataset
from pathlib import Path
import yaml

# ===== 기본 설정 (train_all_folds.py와 호환) =====
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BASE_DIR = Path("../data/UMay/yolo_gas")
SAVE_DIR = Path("./pretrained_folds4")      # 학습 스크립트에서 저장한 폴더
YAML_NAME = "data_aug_1var_5_gs.yaml"       # 학습에 쓴 YAML 이름
BATCH_SIZE = 16

FOLD_START, FOLD_END = 1, 5                 # 평가할 폴드 범위
SPLIT_KEY_ORDER = ("val", "valid", "validation", "test", "train")  # 우선순위 (원하면 train만 쓰게 바꿔도 됨)


def load_yaml(p: Path):
    with open(p, "r") as f:
        return yaml.safe_load(f)


def resolve_split_dir(base_dir: Path, fold_name: str, data_cfg: dict):
    """
    YAML에서 val/valid/validation/test/train 순으로 경로 키를 찾아 존재하는 디렉터리를 반환.
    전부 없으면 None.
    """
    for k in SPLIT_KEY_ORDER:
        rel = data_cfg.get(k)
        if rel:
            split_dir = base_dir / f"yolo_split_{fold_name}" / rel
            if split_dir.exists():
                return split_dir, k
    return None, None


@torch.no_grad()
def reconstruct(ae: AutoEncoder, imgs: torch.Tensor) -> torch.Tensor:
    """
    AE가 get_reconstruction 제공 시 사용, 아니면 forward.
    출력이 [-1,1]일 수도 있으니 [0,1]로 스케일/클램프.
    """
    out = ae.get_reconstruction(imgs) if hasattr(ae, "get_reconstruction") else ae(imgs)
    # 출력 범위 보정
    if out.min() < 0.0 or out.max() > 1.0:
        out = (out + 1) / 2
    return out.clamp(0, 1)


def compute_psnr_from_pair(x: torch.Tensor, y: torch.Tensor, max_val: float = 1.0, eps: float = 1e-8) -> float:
    """
    배치 평균 PSNR 계산. x, y: (B, C, H, W), [0,1] 범위 가정.
    """
    x = x.clamp(0, max_val)
    y = y.clamp(0, max_val)
    mse = (x - y).pow(2).flatten(1).mean(dim=1)  # (B,)
    psnr_per_img = 20 * torch.log10(torch.tensor(max_val, device=x.device)) - 10 * torch.log10(mse + eps)
    return psnr_per_img.mean().item()


@torch.no_grad()
def eval_psnr_on_dir(weights_path: Path, data_dir: Path) -> float:
    ds = GPRDataset(data_dir)
    if len(ds) == 0:
        print(f"[WARN] Empty dataset: {data_dir}")
        return None

    dl = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    ae = AutoEncoder(loss_type="hybrid").to(device).eval()
    state = torch.load(str(weights_path), map_location=device)
    ae.load_state_dict(state, strict=True)

    total_psnr, count = 0.0, 0
    for imgs in dl:
        imgs = imgs.to(device).clamp(0, 1)  # 입력 [0,1] 보장
        i_hat = reconstruct(ae, imgs)
        psnr = compute_psnr_from_pair(imgs, i_hat)  # 배치 평균 PSNR
        total_psnr += psnr * imgs.size(0)
        count += imgs.size(0)

    return float(total_psnr / max(count, 1))


def main():
    results = []

    for fold in range(FOLD_START, FOLD_END + 1):
        fold_name = f"fold{fold:02d}"
        yaml_path = BASE_DIR / f"yolo_split_{fold_name}" / YAML_NAME
        if not yaml_path.exists():
            print(f"[WARN] YAML not found: {yaml_path}")
            continue

        data_cfg = load_yaml(yaml_path)
        split_dir, split_key = resolve_split_dir(BASE_DIR, fold_name, data_cfg)
        if split_dir is None:
            print(f"[WARN] No valid split path found in YAML (looked for {SPLIT_KEY_ORDER}): {yaml_path}")
            continue

        weights_path = SAVE_DIR / f"pretrained_ae_{fold_name}.pth"
        if not weights_path.exists():
            print(f"[WARN] Weights not found: {weights_path}")
            continue

        print(f"\n🔎 Fold {fold_name}")
        print(f"   - split({split_key}): {split_dir}")
        print(f"   - weights:           {weights_path}")

        psnr = eval_psnr_on_dir(weights_path, split_dir)
        if psnr is None:
            print(f"[WARN] Failed PSNR for {fold_name}")
            continue

        print(f"📊 PSNR (mean): {psnr:.2f} dB")
        results.append((fold_name, psnr))

    if results:
        best = max(results, key=lambda x: x[1])
        print("\n✅ Best")
        print(f"→ {best[0]} | {best[1]:.2f} dB")

        print("\n📈 Summary")
        for fn, p in results:
            print(f"- {fn}: {p:.2f} dB")
    else:
        print("\n❌ No PSNR results.")


if __name__ == "__main__":
    main()
