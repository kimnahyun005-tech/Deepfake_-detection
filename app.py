import streamlit as st
import torch
import numpy as np
from PIL import Image, ImageFilter
from torchvision import transforms
from torchvision.models import efficientnet_b4, vit_b_16
from lightning_modules.detector import DeepfakeDetector
import io
import yaml
import os
import urllib.request  # 🚀 구글 requests 대신 기본 다운로드 도구 사용

# 1. 페이지 설정
st.set_page_config(page_title="Deepfake Noise Detector", page_icon="🔍", layout="centered")

# === 🚀 깃허브 릴리즈 직링크 다운로드 함수 (구글 드라이브 함수에서 변경됨) ===
def download_model_file(url, destination):
    with st.spinner("☁️ 깃허브에서 모델 파일(.ckpt)을 안전하게 다운로드 중입니다. (1~2분 소요)..."):
        try:
            urllib.request.urlretrieve(url, destination)
            st.success("✅ 새 모델 다운로드 및 설치 완료!")
        except Exception as e:
            st.error(f"🚨 모델 다운로드 실패: {e}")
            st.stop()

# [순서 1] 기본 디렉토리 경로 설정
BASE_DIR = os.path.dirname(os.path.abspath(__file__)) if "__file__" in locals() else "."

# [순서 2] YAML 설정 파일 읽기
try:
    with open(os.path.join(BASE_DIR, "config.yaml"), encoding="utf-8") as f:
        cfg_rgb = yaml.safe_load(f)
    with open(os.path.join(BASE_DIR, "config_artifact.yaml"), encoding="utf-8") as f:
        cfg_art = yaml.safe_load(f)
except FileNotFoundError as e:
    st.error(f"📁 설정 파일을 찾을 수 없습니다: {e}")
    st.stop()

# === 🚀 [수정] URL을 분석해서 파일명과 모델 종류를 자동 세팅하는 치트키 버전 ===

# ⚠️ 여기에 깃허브에서 복사한 주소를 넣으면 밑에서 자동으로 이름을 추출해!
GITHUB_RELEASE_URL = "https://github.com/kimnahyun005-tech/Deepfake_-detection/releases/download/v1.0/vit_best-v2.ckpt"

# [자동 설정 1] URL 맨 끝에서 'vit_best-v2'라는 이름을 자동으로 뜯어냅니다.
checkpoint_filename = GITHUB_RELEASE_URL.split("/")[-1].replace(".ckpt", "")

# [자동 설정 2] 변수 및 경로 정의 (설정 파일이 없어도 주소 기준으로 강제 동기화)
input_mode = "artifact"
checkpoint_path = os.path.join(BASE_DIR, "models", f"{checkpoint_filename}.ckpt")

# === 🔥 [핵심] 옛날에 받아진 엉뚱한 파일이나 가짜 파일 강제 제거 ===
if os.path.exists(checkpoint_path):
    file_size_mb = os.path.getsize(checkpoint_path) / (1024 * 1024)
    if file_size_mb < 50.0: 
        os.remove(checkpoint_path)

# [순서 5] 자동 다운로드 실행 구문
if not os.path.exists(checkpoint_path):
    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
    download_model_file(GITHUB_RELEASE_URL, checkpoint_path)

# === 🚀 [자동 설정 3] 파일 이름에 'vit'가 들어가면 자동으로 ViT 구조로 스위칭! ===
is_vit = "vit" in checkpoint_filename.lower()
img_size = 224 if is_vit else 380
    
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

# 🔥 모델 종류에 따라 해상도 자동 스위칭
is_vit = "vit" in checkpoint_filename.lower()
img_size = 224 if is_vit else 380

# 개별 텐서 변환 정의
rgb_transform = transforms.Compose([
    transforms.Resize((img_size, img_size)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

artifact_transform = transforms.Compose([
    transforms.Resize((img_size, img_size)),
    ArtifactMapTransform(blur_radius=2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 3. 모델 불러오기 (3채널 기본 상태로 원상복구)
@st.cache_resource
def load_trained_model(cp_path, is_vit_model):
    if is_vit_model:
        backbone = vit_b_16()
        # 6채널로 강제 변환하던 코드를 지우고, 기본 3채널 상태를 유지합니다.
        features = backbone.heads.head.in_features
        backbone.heads.head = torch.nn.Sequential(
            torch.nn.Dropout(0.4),
            torch.nn.Linear(features, 2)
        )
    else:
        backbone = efficientnet_b4()
        features = backbone.classifier[1].in_features
        backbone.classifier = torch.nn.Sequential(
            torch.nn.Dropout(0.4),
            torch.nn.Linear(features, 2)
        )
    
    model = DeepfakeDetector.load_from_checkpoint(
        cp_path, model=backbone, map_location="cpu", weights_only=False
    )
    model.eval()
    return model

model_loaded = False
error_message = ""

try:
    model = load_trained_model(checkpoint_path, is_vit)
    model_loaded = True
except FileNotFoundError:
    error_message = f"📁 '{checkpoint_path}' 파일이 없습니다. 경로와 파일명을 확인해 주세요."
except Exception as e:
    error_message = f"🚨 시스템 에러 발생: {str(e)}"

# 4. 웹사이트 UI 디자인
st.title("🔍 딥페이크 탐지 시스템")
st.info(f"⚙️ 모드: RGB + ARTIFACT MAP 연합 | 해상도: {img_size}x{img_size} | 모델: {checkpoint_filename}.ckpt")

uploaded_file = st.file_uploader("검사할 이미지 파일을 업로드하세요...", type=['jpg', 'jpeg', 'png'])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    
    st.write("---")
    st.subheader("🛠️ Robustness Test ")
    
    jpeg_quality = st.slider(
        "JPEG 압축률 (Quality)",
        min_value=10, max_value=100, value=100, step=5,
        help="값이 낮아질수록 고주파 노이즈가 찌그러지는 악조건을 시뮬레이션합니다."
    )
    
    if jpeg_quality < 100:
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", quality=jpeg_quality)
        buffer.seek(0)
        image = Image.open(buffer)
        st.warning(f"⚠️ 현재 이미지는 JPEG Quality {jpeg_quality} 수준으로 열화된 상태입니다.")
    
    st.image(image, caption='분석 대상 이미지', use_container_width=True)
    
    if not model_loaded:
        st.error(error_message)
    else:  # 👈 'if'와 'else'는 세로 줄이 딱 맞아야 해!
        with st.spinner("이미지 특성 및 노이즈 맵을 정밀 분석 중입니다..."):
            rgb_tensor = rgb_transform(image)          # [3, H, W]
            art_tensor = artifact_transform(image)    # [3, H, W]
            
            # 위에서 만든 art_tensor에 배치 차원 추가
            input_tensor = art_tensor.unsqueeze(0)     # [1, 3, H, W]
            
            with torch.no_grad():
                outputs = model(input_tensor)
                probabilities = torch.nn.functional.softmax(outputs, dim=1)[0]
                
                prob_0 = probabilities[0].item() * 100
                prob_1 = probabilities[1].item() * 100
                
        st.write("---")
        st.subheader("📊 분석 결과")
        
        real_prob = prob_1
        fake_prob = prob_0
            
        col1, col2 = st.columns(2)
        col1.metric(label="Real (원본) 확률", value=f"{real_prob:.2f}%")
        col2.metric(label="Fake (조작) 확률", value=f"{fake_prob:.2f}%")
        
        st.write("---")
        if fake_prob > 50.0:
            st.error(f"⚠️ **딥페이크 조작이 의심됩니다!** (가짜 확률 {fake_prob:.1f}%)")
        else:
            st.success(f"✅ **변조 흔적이 없는 원본 이미지로 보입니다.** (진짜 확률 {real_prob:.1f}%)")
