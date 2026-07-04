import streamlit as st
from PIL import Image, ImageOps, ImageEnhance
from controlnet_aux import LineartAnimeDetector
import numpy as np
import io

# 1. 웹 페이지 스타일 및 레이아웃 설정
st.set_page_config(
    page_title="선따기 AI 에이전트",
    page_icon="🎨",
    layout="wide"
)

st.title("🎨 선화 추출 에이전트 (V3)")
st.write("이미지를 업로드하면 채색 연습용 깔끔한 선화로 변환합니다. **'G펜 모드'**를 켜면 연필 같은 자글자글한 선이 디지털 잉크처럼 깔끔해집니다.")
st.markdown("---")

# 2. AI 모델 로드
@st.cache_resource
def load_lineart_model():
    return LineartAnimeDetector.from_pretrained("lllyasviel/Annotators")

try:
    with st.spinner("🔮 AI 펜터치 엔진을 가동하는 중입니다..."):
        processor = load_lineart_model()
except Exception as e:
    st.error(f"AI 모델 로딩 오류: {e}")
    st.stop()

# 3. 사이드바 컨트롤 (G펜 모드 이진화 설정 추가)
st.sidebar.header("🛠️ 펜터치 세부 설정")

st.sidebar.subheader("1. 색상 보정")
enable_cleaner = st.sidebar.toggle("기본 잔상 제거", value=True)
contrast_factor = st.sidebar.slider("기본 선명도", min_value=1.0, max_value=4.0, value=2.0, step=0.1)

st.sidebar.markdown("---")
st.sidebar.subheader("2. ✒️ G펜 모드 (자글자글한 선 제거)")
use_gpen = st.sidebar.toggle("G펜 모드 켜기 (100% 흑백 처리)", value=True)
# 임계값(Threshold) 슬라이더: 낮으면 연한 선이 날아가고, 높으면 연한 선도 검게 변함
threshold_val = st.sidebar.slider(
    "선 두께 / 노이즈 조절", 
    min_value=50, max_value=250, value=180, 
    help="값이 낮을수록 얇고 깔끔해지지만 일부 선이 끊길 수 있습니다. 높을수록 선이 굵어집니다."
)

# 4. 메인 화면 구성
uploaded_file = st.file_uploader("선화를 추출할 이미지를 업로드하세요 (PNG, JPG, JPEG)", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    original_image = Image.open(uploaded_file).convert("RGB")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🖼️ 원본 이미지")
        st.image(original_image, use_container_width=True)
        
    with col2:
        st.subheader("✏️ 추출된 펜터치 선화")
        
        if st.button("✨ 깔끔하게 선 따기 시작", type="primary"):
            with st.spinner("⚡ AI가 펜터치 라인을 정밀 분석 중입니다..."):
                try:
                    # [Step 1] AI 추출 & 색상 반전
                    raw_lineart = processor(original_image)
                    processed_image = ImageOps.invert(raw_lineart.convert("RGB"))
                    
                    # [Step 2] 기본 보정 (대비 증가)
                    if enable_cleaner:
                        processed_image = ImageOps.grayscale(processed_image)
                        enhancer = ImageEnhance.Contrast(processed_image)
                        processed_image = enhancer.enhance(contrast_factor)
                        processed_image = processed_image.convert("RGB")
                    
                    # [Step 3] ★핵심: G펜 모드 (이진화 알고리즘)
                    if use_gpen:
                        # 이미지를 숫자 배열(Numpy array)로 변환
                        img_array = np.array(processed_image.convert('L'))
                        
                        # 설정한 값(threshold_val)보다 밝은 픽셀은 255(흰색), 어두운 픽셀은 0(검은색)으로 강제 변환
                        img_array = np.where(img_array > threshold_val, 255, 0).astype(np.uint8)
                        
                        # 다시 이미지로 복구
                        processed_image = Image.fromarray(img_array).convert("RGB")
                    
                    st.image(processed_image, use_container_width=True)
                    st.success("🎉 선화 추출이 완료되었습니다! 왼쪽 사이드바에서 두께를 조절해보세요.")
                    
                    buf = io.BytesIO()
                    processed_image.save(buf, format="PNG")
                    byte_im = buf.getvalue()
                    
                    st.download_button(
                        label="💾 깔끔한 선화 다운로드 (.png)",
                        data=byte_im,
                        file_name="webtoon_lineart_gpen.png",
                        mime="image/png"
                    )
                    
                except Exception as e:
                    st.error(f"오류 발생: {e}")
