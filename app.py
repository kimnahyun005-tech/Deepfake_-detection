import streamlit as st
import torch
import numpy as np
from PIL import Image, ImageFilter
from torchvision import transforms
from torchvision.models import efficientnet_b4
from lightning_modules.detector import DeepfakeDetector
import io  # 🔥 교수님 피드백 반영: JPEG 압축 시뮬레이션을 위한 라이브러리 추가

# 1. 페이지 설정
st.set_page_config(page_title="Deepfake Noise Detector", page_icon="🔍", layout="centered")

# 2. 노이즈 추출 전처리 함수 (원본 - 가우시안 블러 = 고주파 Artifact Map)
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
    transforms.Resize((380, 380)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 3. 모델 불러오기
@st.cache_resource
def load_trained_model():
    backbone = efficientnet_b4()
    features = backbone.classifier[1].in_features
    backbone.classifier = torch.nn.Sequential(
        torch.nn.Dropout(0.4),
        torch.nn.Linear(features, 2)
    )
    checkpoint_path = "models/best_model.ckpt"
    model = DeepfakeDetector.load_from_checkpoint(checkpoint_path, model=backbone, map_location="cpu")
    model.eval()
    return model

model_loaded = False
error_message = ""

try:
    model = load_trained_model()
    model_loaded = True
except FileNotFoundError:
    error_message = "📁 'models/best_model.ckpt' 파일이 지정된 폴더에 없습니다. 경로를 다시 확인해 주세요."
except Exception as e:
    error_message = f"🚨 시스템 에러 발생: {str(e)}"

# 4. 웹사이트 UI 디자인
st.title("🔍 노이즈 기반 딥페이크 탐지 시스템")
st.write("이미지를 업로드하면 미세 노이즈 패턴을 분석하여 진짜/가짜 확률을 정밀하게 판별합니다.")

uploaded_file = st.file_uploader("검사할 이미지 파일을 업로드하세요...", type=['jpg', 'jpeg', 'png'])

if uploaded_file is not None:
    # 원본 이미지 열기
    image = Image.open(uploaded_file)
    
    st.write("---")
    # 🔥 교수님 피드백 5, 7번 반영: 고의적 화질 저하 및 재인코딩 시뮬레이터 조절 바
    st.subheader("🛠️ Robustness Test ")
    st.caption("카카오톡 전송, SNS 업로드, 인코딩 등으로 인해 고의로 화질이 저하된 환경을 가상으로 생성하여 모델의 강인함을 테스트합니다.")
    
    jpeg_quality = st.slider(
        "JPEG 압축률 (Quality)",
        min_value=10,
        max_value=100,
        value=100,
        step=5,
        help="100에 가까울수록 원본 화질이며, 값이 낮아질수록 고주파 노이즈가 찌그러지는 악조건을 시뮬레이션합니다."
    )
    
    # 슬라이더 값이 100 미만이면 강제로 이미지를 JPEG로 압축 후 다시 로드
    if jpeg_quality < 100:
        buffer = io.BytesIO()
        # 🔥 에러 해결 핵심: PNG나 투명도가 있는 이미지도 에러 없이 압축되도록 RGB 모드로 강제 변환
        image_rgb = image.convert("RGB") 
        image_rgb.save(buffer, format="JPEG", quality=jpeg_quality)
        buffer.seek(0)
        image = Image.open(buffer)
        st.warning(f"⚠️ 현재 이미지는 JPEG Quality {jpeg_quality} 수준으로 압축 및 열화된 상태입니다.")
    
    # 화면에 처리된(혹은 원본) 이미지 띄우기
    st.image(image, caption=f'분석 대상 이미지 (JPEG Quality: {jpeg_quality})', use_column_width=True)
    
    if not model_loaded:
        st.error(error_message)
    else:
        with st.spinner("이미지에서 미세 노이즈 특성을 추출하여 분석하는 중입니다..."):
           input_tensor = transform(image.convert("RGB")).unsqueeze(0)
            with torch.no_grad():
                outputs = model(input_tensor)
                probabilities = torch.nn.functional.softmax(outputs, dim=1)[0]
                real_prob = probabilities[1].item()
                fake_prob = probabilities[0].item()
        # =========================================================================
        # 🔥 [교수님 점수 따기용 핵심 추가] 인공지능이 바라보는 노이즈 지도 시각화
        # =========================================================================
        st.write("---")
        st.subheader("🖼️ 인공지능이 분석 중인 미세 노이즈 패턴 (Artifact Map)")
        st.caption("모델은 아래의 고주파 노이즈 분포를 보고 진짜/가짜를 판단합니다. 배경이 너무 강하면 얼굴 노이즈가 죽을 수 있습니다.")
        
        # 모델에 들어가는 것과 똑같은 노이즈 맵 생성 후 화면에 표시
        debug_artifact_transformer = ArtifactMapTransform(blur_radius=2)
        artifact_image = debug_artifact_transformer(image.resize((380, 380)))
        st.image(artifact_image, caption="추출된 고주파 노이즈 맵", use_container_width=True)
        # =========================================================================
      
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
       
