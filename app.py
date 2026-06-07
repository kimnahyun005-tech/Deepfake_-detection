import streamlit as st
import torch
import numpy as np
from PIL import Image, ImageFilter
from torchvision import transforms
from torchvision.models import efficientnet_b4
from lightning_modules.detector import DeepfakeDetector
import io
import os
import yaml
from torch.utils.data import DataLoader
# 🔥 아나콘다 모의고사 엔진을 그대로 복제하기 위한 라이브러리
import pytorch_lightning as pl
from datasets.hybrid_loader import HybridDeepfakeDataset

# 1. 페이지 설정
st.set_page_config(page_title="Deepfake Noise Detector", page_icon="🔍", layout="centered")

# === 📂 config.yaml 설정 자동 로드 및 동기화 ===
with open("config.yaml", encoding="utf-8") as f:
    cfg = yaml.safe_load(f)

input_mode = cfg.get("input_mode", "rgb") 
checkpoint_filename = cfg.get("checkpoint_filename", "best_model")
checkpoint_path = f"models/{checkpoint_filename}.ckpt"

# 2. 노이즈 추출 전처리 함수
class ArtifactMapTransform:
    def __init__(self, blur_radius=2):
        self.blur_radius = blur_radius
    def __call__(self, img):
        img = img.convert("RGB")
        blurred = img.filter(ImageFilter.GaussianBlur(radius=self.blur_radius))
        img_np, blurred_np = np.array(img).astype(np.float32), np.array(blurred).astype(np.float32)
        artifact = np.abs(img_np - blurred_np)
        artifact = artifact / (artifact.max() + 1e-8) * 255
        return Image.fromarray(artifact.astype(np.uint8))

# 규격 자동 구성
if input_mode == "rgb":
    transform = transforms.Compose([
        transforms.Resize((380, 380)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
else:
    transform = transforms.Compose([
        transforms.Resize((380, 380)),
        ArtifactMapTransform(blur_radius=2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

# 3. 모델 불러오기
@st.cache_resource
def load_trained_model(cp_path):
    backbone = efficientnet_b4()
    features = backbone.classifier[1].in_features
    backbone.classifier = torch.nn.Sequential(
        torch.nn.Dropout(0.4),
        torch.nn.Linear(features, 2)
    )
    model = DeepfakeDetector.load_from_checkpoint(cp_path, model=backbone, map_location="cpu")
    model.eval()
    return model

model_loaded = False
error_message = ""

try:
    model = load_trained_model(checkpoint_path)
    model_loaded = True
except FileNotFoundError:
    error_message = f"📁 '{checkpoint_path}' 파일이 지정된 폴더에 없습니다. 경로를 다시 확인해 주세요."
except Exception as e:
    error_message = f"🚨 시스템 에러 발생: {str(e)}"

# 4. 웹사이트 UI 디자인
st.title("🔍 딥페이크 탐지 시스템 (엔진 완벽 동기화)")
st.info(f"⚙️ 연동 모드: {input_mode.upper()} | 체크포인트: {checkpoint_filename}.ckpt")

uploaded_file = st.file_uploader("검사할 이미지 파일을 업로드하세요...", type=['jpg', 'jpeg', 'png'])

if uploaded_file is not None:
    # 🔥 [해결 1] 원본 파일의 확장자(.png, .jpg 등)를 그대로 추출하여 포맷 파괴 방지
    file_ext = os.path.splitext(uploaded_file.name)[1]
    temp_path = f"temp_inference_image{file_ext}"
    
    st.write("---")
    st.subheader("🛠️ Robustness Test ")
    
    jpeg_quality = st.slider(
        "JPEG 압축률 (Quality)",
        min_value=10, max_value=100, value=100, step=5,
        help="값이 낮아질수록 고주파 노이즈가 찌그러지는 악조건을 시뮬레이션합니다."
    )
    
    image_display = Image.open(uploaded_file).convert("RGB")
    if jpeg_quality < 100:
        image_display.save(temp_path, format="JPEG", quality=jpeg_quality)
        st.warning(f"⚠️ 현재 이미지는 JPEG Quality {jpeg_quality} 수준으로 열화된 상태입니다.")
    else:
        # 원본 확장자 그대로 바이너리 저장 (노이즈 오염 원천 차단)
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
    
    st.image(image_display, caption='분석 대상 이미지', use_container_width=True)
    
    if not model_loaded:
        st.error(error_message)
    else:
        with st.spinner("아나콘다 검증 엔진을 구동하여 정밀 분석 중..."):
            # 🔥 [해결 2] 아나콘다 robustness_test.py의 검증 파이프라인을 100% 똑같이 실행
            # 0번(Real)이라고 가정한 뒤, 엔진이 내린 정답률(Accuracy)을 통해 역추적합니다.
            temp_sources = [(temp_path, 0)] 
            eval_dataset = HybridDeepfakeDataset(temp_sources, transform=transform)
            val_loader = DataLoader(eval_dataset, batch_size=1, shuffle=False, num_workers=0)
            
            trainer = pl.Trainer(
                accelerator="gpu" if torch.cuda.is_available() else "cpu",
                devices=1, logger=False, enable_progress_bar=False
            )
            
            # 아나콘다와 완전히 동일한 검증 루틴 실행
            results = trainer.validate(model, dataloaders=val_loader, verbose=False)
            
            is_real = False
            if results:
                acc = results[0].get("val_acc", None)
                if acc is None:
                    acc = results[0].get("val_accuracy", 0.0)
                
                # 가정한 정답(Real)과 모델의 예측이 일치하면 acc가 1.0(100%)이 나옵니다.
                if acc > 0.5:
                    is_real = True

        st.write("---")
        st.subheader("📊 분석 결과")
        
        if is_real:
            st.success("✅ **변조 흔적이 없는 원본 이미지로 판정되었습니다. (Real)**")
            st.balloons() # 축하 풍선 이펙트 추가!
        else:
            st.error("🚨 **고주파 패턴 변형이 감지되어 딥페이크 조작으로 판정되었습니다. (Fake)**")
            
    # 임시 파일 삭제
    if os.path.exists(temp_path):
        os.remove(temp_path)
