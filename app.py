import streamlit as st
from PIL import Image
from controlnet_aux import LineartAnimeDetector
import io

# 웹 페이지 제목 및 레이아웃 설정
st.set_page_config(page_title="웹툰 선따기 AI", page_icon="🎨", layout="centered")

st.title("🎨 무검열 웹툰 선화 추출 에이전트")
st.write("이미지를 업로드하면 채색 연습용 깔끔한 선화(Line Art)를 추출합니다.")

# 모델 로드 함수 (캐싱을 적용하여 접속할 때마다 새로 다운받지 않도록 함)
@st.cache_resource
def load_lineart_model():
    # 애니메이션 스타일에 최적화된 선화 추출 모델인 lllyasviel/Annotators에서 로드
    return LineartAnimeDetector.from_pretrained("lllyasviel/Annotators")

# 모델 준비 (사용자에게 로딩 바 표시)
try:
    with st.spinner("AI 엔진을 깨우는 중입니다... (최초 실행 시 1~2분 소요)"):
        processor = load_lineart_model()
except Exception as e:
    st.error(f"모델을 불러오는 중 오류가 발생했습니다: {e}")
    st.stop()

# 이미지 업로드 컴포넌트
uploaded_file = st.file_uploader("선화를 추출할 이미지를 선택하세요 (PNG, JPG, JPEG)", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    # 업로드된 이미지를 PIL 이미지로 변환
    image = Image.open(uploaded_file).convert("RGB")
    
    # 화면을 두 구역으로 나누어 원본과 결과물 배치
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🖼️ 원본 이미지")
        st.image(image, use_container_width=True)
        
    with col2:
        st.subheader("✏️ 추출된 선화")
        # 실행 버튼을 누르면 AI 연산 시작
        if st.button("AI 선따기 시작"):
            with st.spinner("⚡ 펜터치 작업 중..."):
                # AI 모델 실행하여 선화 추출
                result_image = processor(image)
                st.image(result_image, use_container_width=True)
                
                # 다운로드 버튼 생성을 위해 바이너리로 변환
                buf = io.BytesIO()
                result_image.save(buf, format="PNG")
                byte_im = buf.getvalue()
                
                st.download_button(
                    label="💾 선화 다운로드 (.png)",
                    data=byte_im,
                    file_name="extracted_lineart.png",
                    mime="image/png"
                )
