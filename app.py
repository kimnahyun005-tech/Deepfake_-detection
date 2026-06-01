import streamlit as st
import torch
import numpy as np
from PIL import Image, ImageFilter
from torchvision import transforms
from torchvision.models import efficientnet_b4
from lightning_modules.detector import DeepfakeDetector

# 1. 페이지 설정
st.set_page_config(page_title="Deepfake Noise Detector", page_icon="🔍", layout="centered")

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

transform = transforms.Compose([
    ArtifactMapTransform(blur_radius=2),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 3. 모델 불러오기 (★ map_location="cpu" 추가로 노트북 에러 완벽 해결!)
@st.cache_resource
def load_trained_model():
    backbone = efficientnet_b4()
    features = backbone.classifier[1].in_features
    backbone.classifier = torch.nn.Sequential(
        torch.nn.Dropout(0.4),
        torch.nn.Linear(features, 2)
    )
    checkpoint_path = "models/best_model.ckpt"
    
    # 코랩(GPU) 파일을 내 노트북(CPU)에서 읽을 수 있게 강제 변환 주입!
    model = DeepfakeDetector.load_from_checkpoint(checkpoint_path, model=backbone, map_location="cpu")
    model.eval()
    return model

# 에러 추적을 위한 가드 장치
model_loaded = False
error_message = ""

try:
    model = load_trained_model()
    model_loaded = True
except FileNotFoundError:
    error_message = "📁 실제로 'models/best_model.ckpt' 파일이 지정된 폴더에 없습니다. 경로를 다시 확인해 주세요."
except Exception as e:
    error_message = f"🚨 시스템 에러 발생 (컴퓨터가 아픈 진짜 이유): {str(e)}"

# 4. 웹사이트 UI 디자인
st.title("🔍 노이즈 기반 딥페이크 탐지 시스템")
st.write("이미지를 업로드하면 미세 노이즈 패턴을 분석하여 진짜/가짜 확률을 정밀하게 판별합니다.")

uploaded_file = st.file_uploader("검사할 이미지 파일을 업로드하세요...", type=['jpg', 'jpeg', 'png'])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='업로드된 이미지', use_column_width=True)
    
    if not model_loaded:
        st.error(error_message)
    else:
        with st.spinner("이미지에서 미세 노이즈 특성을 추출하여 분석하는 중입니다..."):
            input_tensor = transform(image).unsqueeze(0)
            with torch.no_grad():
                outputs = model(input_tensor)
                probabilities = torch.nn.functional.softmax(outputs, dim=1)[0]
                real_prob = probabilities[0].item()
                fake_prob = probabilities[1].item()
                
        # 결과 출력
        st.subheader("📊 분석 결과")
        col1, col2 = st.columns(2)
        col1.metric(label="Real (원본) 확률", value=f"{real_prob * 100:.2f}%")
        col2.metric(label="Fake (조작) 확률", value=f"{fake_prob * 100:.2f}%")
        
        st.write("---")
        if fake_prob > 0.5:
            st.error(f"⚠️ **딥페이크 조작이 의심됩니다!** (가짜 확률 {fake_prob*100:.1f}%)")
        else:
            st.success(f"✅ **변조 흔적이 없는 원본 이미지로 보입니다.** (진짜 확률 {real_prob*100:.1f}%)")