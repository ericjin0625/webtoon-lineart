import streamlit as st
from PIL import Image, ImageOps, ImageEnhance
from controlnet_aux import LineartAnimeDetector
import numpy as np
import io
import zipfile

# 1. 웹 페이지 스타일 및 레이아웃 설정
st.set_page_config(
    page_title="무검열 웹툰 선따기 AI 에이전트",
    page_icon="🎨",
    layout="wide"
)

st.title("🎨 무검열 웹툰 선화 추출 에이전트 (V4: 대량 처리 모드)")
st.write("여러 장의 이미지를 한 번에 올리고, 한꺼번에 선화로 변환하여 **압축 파일(ZIP)**로 일괄 다운로드할 수 있습니다.")
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

# 3. 사이드바 컨트롤
st.sidebar.header("🛠️ 펜터치 세부 설정")
st.sidebar.write("※ 설정한 값은 업로드한 **모든 이미지에 동일하게 적용**됩니다.")

st.sidebar.subheader("1. 색상 보정")
enable_cleaner = st.sidebar.toggle("기본 잔상 제거", value=True)
contrast_factor = st.sidebar.slider("기본 선명도", min_value=1.0, max_value=4.0, value=2.0, step=0.1)

st.sidebar.markdown("---")
st.sidebar.subheader("2. ✒️ G펜 모드 (자글자글한 선 제거)")
use_gpen = st.sidebar.toggle("G펜 모드 켜기 (100% 흑백 처리)", value=True)
threshold_val = st.sidebar.slider(
    "선 두께 / 노이즈 조절", 
    min_value=50, max_value=250, value=180, 
    help="값이 낮을수록 얇고 깔끔해지지만 일부 선이 끊길 수 있습니다. 높을수록 선이 굵어집니다."
)

# 4. 메인 화면 구성 (★핵심: 여러 장 업로드 허용)
# accept_multiple_files=True 옵션을 주면 수십, 수백 장을 한 번에 선택할 수 있습니다.
uploaded_files = st.file_uploader(
    "선화를 추출할 이미지들을 모두 드래그해서 올려주세요 (다중 선택 가능)", 
    type=["png", "jpg", "jpeg"], 
    accept_multiple_files=True
)

if uploaded_files:
    st.info(f"총 **{len(uploaded_files)}**장의 이미지가 대기 중입니다.")
    
    if st.button("✨ 일괄 선 따기 시작 (Batch Process)", type="primary"):
        # ZIP 파일을 만들기 위한 메모리 준비
        zip_buffer = io.BytesIO()
        
        # 진행률을 보여주기 위한 프로그레스 바
        progress_text = "작업 진행 중..."
        my_bar = st.progress(0, text=progress_text)
        
        # 압축 파일 생성 시작
        with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zip_file:
            for idx, uploaded_file in enumerate(uploaded_files):
                filename = uploaded_file.name
                original_image = Image.open(uploaded_file).convert("RGB")
                
                # 프로그레스 바 실시간 업데이트
                progress_percent = int(((idx + 1) / len(uploaded_files)) * 100)
                my_bar.progress(progress_percent, text=f"[{idx+1}/{len(uploaded_files)}] '{filename}' 처리 중...")
                
                try:
                    # [Step 1] AI 추출 & 색상 반전
                    raw_lineart = processor(original_image)
                    processed_image = ImageOps.invert(raw_lineart.convert("RGB"))
                    
                    # [Step 2] 기본 보정
                    if enable_cleaner:
                        processed_image = ImageOps.grayscale(processed_image)
                        enhancer = ImageEnhance.Contrast(processed_image)
                        processed_image = enhancer.enhance(contrast_factor)
                        processed_image = processed_image.convert("RGB")
                    
                    # [Step 3] G펜 모드
                    if use_gpen:
                        img_array = np.array(processed_image.convert('L'))
                        img_array = np.where(img_array > threshold_val, 255, 0).astype(np.uint8)
                        processed_image = Image.fromarray(img_array).convert("RGB")
                    
                    # 화면에 미리보기 표시
                    col1, col2 = st.columns(2)
                    with col1:
                        st.image(original_image, caption=f"원본: {filename}", use_container_width=True)
                    with col2:
                        st.image(processed_image, caption=f"선화: {filename}", use_container_width=True)
                    st.divider()
                    
                    # [Step 4] 결과물을 압축 파일(ZIP)에 추가
                    img_buffer = io.BytesIO()
                    processed_image.save(img_buffer, format="PNG")
                    # 파일명 앞에 'lineart_'를 붙여서 압축 파일 내부에 저장
                    zip_file.writestr(f"lineart_{filename.rsplit('.', 1)[0]}.png", img_buffer.getvalue())
                    
                except Exception as e:
                    st.error(f"'{filename}' 처리 중 오류 발생: {e}")
        
        # 모든 작업이 끝났을 때
        my_bar.progress(100, text="✨ 모든 작업이 완료되었습니다!")
        st.success("모든 선화 추출이 완료되었습니다. 아래 버튼을 눌러 ZIP 파일로 한 번에 다운로드하세요.")
        
        # 5. 일괄 다운로드 버튼 제공
        st.download_button(
            label="📦 완성된 선화 전체 다운로드 (ZIP)",
            data=zip_buffer.getvalue(),
            file_name="my_linearts_batch.zip",
            mime="application/zip",
            type="primary"
        )
