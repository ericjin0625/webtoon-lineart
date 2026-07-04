import streamlit as st
from PIL import Image, ImageOps, ImageEnhance
from controlnet_aux import LineartAnimeDetector
import io

# 1. 웹 페이지 스타일 및 레이아웃 설정
st.set_page_config(
    page_title="무검열 웹툰 선따기 AI 에이전트",
    page_icon="🎨",
    layout="wide"  # 화면을 넓게 쓰기 위해 wide 모드로 설정
)

# 제목 섹션
st.title("🎨 무검열 웹툰 선화 추출 에이전트 (V2)")
st.write("기존의 검은 배경 형광색 선 문제를 해결하고, **흰 배경에 깔끔한 검은색 펜터치**로 변환해주는 완전 무료 에이전트입니다.")
st.markdown("---")

# 2. AI 모델 로드 (캐싱 처리로 속도 최적화)
@st.cache_resource
def load_lineart_model():
    # 애니메이션/웹툰 스타일에 최적화된 Lineart 모델 로드
    return LineartAnimeDetector.from_pretrained("lllyasviel/Annotators")

try:
    with st.spinner("🔮 AI 펜터치 엔진을 가동하는 중입니다... (최초 실행 시 약 1~2분 소요)"):
        processor = load_lineart_model()
except Exception as e:
    st.error(f"AI 모델을 불러오는 중 오류가 발생했습니다: {e}")
    st.stop()

# 3. 사이드바 컨트롤 (선 두께 및 깔끔함 조정 기능 추가)
st.sidebar.header("🛠️ 선화 세부 설정")
st.sidebar.write("결과물이 만족스럽지 않다면 아래 옵션을 조절해보세요.")

# 선화 보정 옵션들
enable_cleaner = st.sidebar.toggle("선화 주변 잔상/노이즈 제거 (추천)", value=True)
contrast_factor = st.sidebar.slider("선명도 (대비 조절)", min_value=1.0, max_value=4.0, value=2.0, step=0.1)

# 4. 메인 화면 구성 (업로드 및 비교)
uploaded_file = st.file_uploader("선화를 추출할 이미지를 업로드하세요 (PNG, JPG, JPEG)", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    # 이미지 열기 및 RGB 변환
    original_image = Image.open(uploaded_file).convert("RGB")
    
    # 2단 레이아웃 구성
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🖼️ 원본 이미지")
        st.image(original_image, use_container_width=True)
        
    with col2:
        st.subheader("✏️ 추출된 펜터치 선화")
        
        # 버튼 클릭 시 작동
        if st.button("✨ 깔끔하게 선 따기 시작", type="primary"):
            with st.spinner("⚡ AI가 펜터치 라인을 정밀 분석 중입니다..."):
                try:
                    # [Step 1] AI 모델 실행 (기본 결과는 검은 배경 + 흰색 선)
                    raw_lineart = processor(original_image)
                    
                    # [Step 2] 색상 반전 처리 (★핵심: 흰 배경 + 검은색 선으로 변환)
                    inverted_lineart = ImageOps.invert(raw_lineart.convert("RGB"))
                    
                    # [Step 3] 추가 보정 작업 (노이즈 제거 및 선명도 업그레이드)
                    processed_image = inverted_lineart
                    
                    if enable_cleaner:
                        # 회색조(Grayscale)로 변환하여 미세한 색상 노이즈 제거
                        processed_image = ImageOps.grayscale(processed_image)
                        
                        # 극단적인 이진화 대신, 소프트한 문턱치 처리를 원할 경우 대비(Contrast) 증가
                        enhancer = ImageEnhance.Contrast(processed_image)
                        processed_image = enhancer.enhance(contrast_factor)
                        
                        # 다시 RGB 형식으로 복원
                        processed_image = processed_image.convert("RGB")
                    
                    # 결과 화면 표시
                    st.image(processed_image, use_container_width=True)
                    st.success("🎉 선화 추출이 완료되었습니다!")
                    
                    # [Step 4] 다운로드 기능 제공
                    buf = io.BytesIO()
                    processed_image.save(buf, format="PNG")
                    byte_im = buf.getvalue()
                    
                    st.download_button(
                        label="💾 깔끔한 선화 다운로드 (.png)",
                        data=byte_im,
                        file_name="webtoon_lineart_clean.png",
                        mime="image/png"
                    )
                    
                except Exception as e:
                    st.error(f"선화 추출 중 오류가 발생했습니다: {e}")
                    st.info("💡 팁: 이미지 해상도가 너무 높으면 스트림릿 서버 메모리 한계로 에러가 날 수 있습니다. 이미지 크기를 조금 줄여서 시도해보세요.")
