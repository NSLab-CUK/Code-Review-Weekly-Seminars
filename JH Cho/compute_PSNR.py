import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
from ultralytics.nn.modules.ae import AutoEncoder


def compute_psnr(img1, img2, eps=1e-8):
    mse = F.mse_loss(img1, img2)
    if mse == 0:
        return float('inf')
    return 20 * torch.log10(1.0 / torch.sqrt(mse + eps))


def main():
    # 1. 모델 로드
    model = AutoEncoder()
    model.load_state_dict(torch.load("pretrained_folds4/pretrained_ae_fold01.pth"))
    model.eval()

    # 2. 이미지 로드 및 전처리 (1채널)
    img = Image.open("your_input.png").convert("L")
    transform = transforms.Compose([
        transforms.ToTensor(),  # (1, H, W)
    ])
    img_tensor = transform(img).unsqueeze(0)  # (1, 1, H, W)

    # 3. AE 재구성
    with torch.no_grad():
        i_hat = model.get_reconstruction(img_tensor)

    # 4. PSNR 계산
    psnr = compute_psnr(img_tensor, i_hat)
    print(f"📊 PSNR: {psnr:.2f} dB")


if __name__ == "__main__":
    main()
