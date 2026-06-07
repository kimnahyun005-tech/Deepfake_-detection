import streamlit as st
import torch
import numpy as np
from PIL import Image, ImageFilter
from torchvision import transforms
from torchvision.models import efficientnet_b4
from lightning_modules.detector import DeepfakeDetector
import io
import yaml  # 🔥 아나콘다 테스트와 똑같이 설정파일을 읽기 위해 추가!

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

# 🔥 [핵심 보완] config.yaml의 설정과 완벽하게 일치하도록 전처리를 자동 구성합니다.
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

# 3. 모델 불러오기 (경로가 바뀌면 캐시가 자동 갱신되도록 수정)
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
st.title("🔍 딥페이크 탐지 시스템 (자동 동기화)")
st.info(f"⚙️ 현재 모델 연동 모드: **{input_mode.upper()}** | 사용 중인 체크포인트: **{checkpoint_filename}.ckpt**")

uploaded_file = st.file_uploader("검사할 이미지 파일을 업로드하세요...", type=['jpg', 'jpeg', 'png'])

if uploaded_file is not None:
    # 안전장치: 업로드 즉시 투명도가 제거된 3채널 RGB로 변환
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
        st.warning(f"⚠️ 현재 이미지는 JPEG Quality {jpeg_quality} 수준으로 압축 및 열화된 상태입니다.")
    
    st.image(image, caption=f'분석 대상 이미지 (JPEG Quality: {jpeg_quality})', use_container_width=True)
    
    if not model_loaded:
        st.error(error_message)
    else:
        with st.spinner("이미지 특성을 분석하는 중입니다..."):
            input_tensor = transform(image).unsqueeze(0)
            with torch.no_grad():
                outputs = model(input_tensor)
                probabilities = torch.nn.functional.softmax(outputs, dim=1)[0]
                
                # 순수 클래스 점수 추출
                prob_0 = probabilities[0].item() * 100
                prob_1 = probabilities[1].item() * 100
                
        st.write("---")
        st.subheader("📊 분석 결과")
        
        # 🔥 [마법의 치트키 구문] 인덱스 반전 문제를 마우스 딸깍으로 즉시 해결하는 라디오 버튼
        mapping_option = st.radio(
            "🔄 만약 진짜/가짜 결과가 완벽하게 정반대로 뒤집혀서 나온다면 아래 매칭을 전환하세요!",
            ["[옵션 A] 0번=원본(Real), 1번=조작(Fake)", "[옵션 B] 0번=조작(Fake), 1번=원본(Real)"],
            index=0
        )
        
        if mapping_option == "[옵션 A] 0번=원본(Real), 1번=조작(Fake)":
            real_prob = prob_0
            fake_prob = prob_1
        else:
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
