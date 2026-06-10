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
import urllib.request  # 구글 드라이브 자동 다운로드용

# 1. 페이지 설정
st.set_page_config(page_title="Deepfake Noise Detector", page_icon="🔍", layout="centered")

# 구글 드라이브 다운로드 함수 (필요할 때 호출)
def download_file_from_google_drive(file_id, destination):
    URL = f"https://docs.google.com/uc?export=download&id={file_id}"
    with st.spinner("☁️ 서버에서 새 모델 파일(.ckpt)을 최초 1회 다운로드 중입니다. (1~2분 소요)..."):
        try:
            urllib.request.urlretrieve(URL, destination)
            st.success("✅ 새 모델 다운로드 완료!")
        except Exception as e:
            st.error(f"🚨 모델 다운로드 실패: {e}")
            st.stop()

# [순서 1] 기본 디렉토리 경로 설정을 먼저 합니다.
BASE_DIR = os.path.dirname(os.path.abspath(__file__)) if "__file__" in locals() else "."

# [순서 2] YAML 설정 파일을 먼저 읽어와야 합니다.
try:
    with open(os.path.join(BASE_DIR, "config.yaml"), encoding="utf-8") as f:
        cfg_rgb = yaml.safe_load(f)
    with open(os.path.join(BASE_DIR, "config_artifact.yaml"), encoding="utf-8") as f:
        cfg_art = yaml.safe_load(f)
except FileNotFoundError as e:
    st.error(f"📁 설정 파일을 찾을 수 없습니다: {e}")
    st.stop()

# [순서 3] 읽어온 설정 파일(cfg_art)에서 이름을 꺼내 변수를 정의합니다. (★ NameError 해결 포인트)
input_mode = cfg_art.get("input_mode", "artifact") 
checkpoint_filename = cfg_art.get("checkpoint_filename", "artifact_model") 

# [순서 4] 변수가 확실히 생성된 후, 그것을 활용해 모델 경로를 정의합니다.
checkpoint_path = os.path.join(BASE_DIR, "models", f"{checkpoint_filename}.ckpt")

# [순서 5] 만약 구글 드라이브 자동 다운로드 기능을 쓸 거라면 여기에 배치합니다.
# (안 쓸 거라면 이 if문 블록 전체를 지우거나 주석 처리해도 무방해!)
if not os.path.exists(checkpoint_path):
    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
    
    # ⚠️ 여기에 본인의 구글 드라이브 파일 ID를 넣으세요!
    GOOGLE_DRIVE_FILE_ID = "1pz9WFmKZrrPlUBE2EeRiWJ_HhtSJAox7" 
    
    download_file_from_google_drive(GOOGLE_DRIVE_FILE_ID, checkpoint_path)

# --- 이 아래부터 원래 있던 2번 노이즈 추출 전처리 함수 코드를 이어 붙이면 돼! ---
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

# 3. 모델 불러오기
@st.cache_resource
def load_trained_model(cp_path, is_vit_model):
    if is_vit_model:
        backbone = vit_b_16()
        features = backbone.heads.head.in_features
        # ⚠️ 중요: RGB(3개 채널) + Artifact(3개 채널) = 총 6개 채널 입력에 맞춰 모델 첫 레이어 변경 필요할 수 있음
        # 만약 main_trainer.py에서 첫 conv 레이어 채널을 6으로 바꿨다면 app.py 모델 정의에도 반영해야 함
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
    
    model = DeepfakeDetector.load_from_checkpoint(cp_path, model=backbone, map_location="cpu")
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
    else:
        with st.spinner("이미지 특성 및 노이즈 맵을 정밀 분석 중입니다..."):
            # 🔥 [RGB]와 [Artifact Map] 개별 변환 후 채널 방향으로 병합 (Concat)
            rgb_tensor = rgb_transform(image)          # [3, H, W]
            art_tensor = artifact_transform(image)    # [3, H, W]
            
            # 두 텐서를 채널축(dim=0) 기준으로 합쳐서 6채널 생성 후, 배치 차원 추가
            input_tensor = torch.cat([rgb_tensor, art_tensor], dim=0).unsqueeze(0) # [1, 6, H, W]
            
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
