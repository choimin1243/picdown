import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import io
import base64
import pandas as pd
import matplotlib.font_manager as fm
import platform
import os
import matplotlib as mpl
from matplotlib import font_manager

def download_nanum_font():
    """Nanum 폰트를 다운로드하고 설치합니다"""
    import requests
    import zipfile
    
    # NanumGothic 폰트 다운로드 URL
    font_url = "https://github.com/naver/nanumfont/raw/master/NanumFont_TTF_ALL.zip"
    
    # 폰트 저장 경로
    font_dir = os.path.join(os.path.expanduser("~"), ".fonts")
    os.makedirs(font_dir, exist_ok=True)
    
    # 이미 폰트가 존재하는지 확인
    if os.path.exists(os.path.join(font_dir, "NanumGothic.ttf")):
        st.success("나눔 폰트가 이미 설치되어 있습니다.")
        return True
    
    try:
        # 폰트 다운로드
        st.info("나눔 폰트 다운로드 중...")
        response = requests.get(font_url, stream=True)
        
        if response.status_code == 200:
            zip_path = os.path.join(font_dir, "nanum_fonts.zip")
            
            # 다운로드한 파일 저장
            with open(zip_path, 'wb') as f:
                f.write(response.content)
            
            # 압축 해제
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(font_dir)
            
            # 압축 파일 삭제
            os.remove(zip_path)
            
            # 폰트 캐시 업데이트
            font_manager.fontManager.addfont(os.path.join(font_dir, "NanumGothic.ttf"))
            
            st.success("나눔 폰트가 성공적으로 설치되었습니다.")
            return True
        else:
            st.error(f"폰트 다운로드 실패: {response.status_code}")
            return False
            
    except Exception as e:
        st.error(f"폰트 설치 중 오류 발생: {str(e)}")
        return False

def set_matplotlib_korean_font():
    """한글 폰트 설정 함수"""
    system_name = platform.system()
    
    # 기본 폰트 설정
    if system_name == "Windows":
        # 윈도우의 경우 맑은 고딕 폰트 사용
        font_name = "Malgun Gothic"
    elif system_name == "Darwin":  # macOS
        font_name = "AppleGothic"
    else:  # Linux 등
        font_name = "NanumGothic"
    
    # 폰트가 시스템에 있는지 확인
    font_found = False
    available_fonts = [f.name for f in font_manager.fontManager.ttflist]
    
    if font_name in available_fonts:
        font_found = True
    elif "NanumGothic" in available_fonts:
        font_name = "NanumGothic"
        font_found = True
    elif "NanumBarunGothic" in available_fonts:
        font_name = "NanumBarunGothic"
        font_found = True
    elif "Noto Sans CJK KR" in available_fonts:
        font_name = "Noto Sans CJK KR"
        font_found = True
    elif "Noto Sans KR" in available_fonts:
        font_name = "Noto Sans KR"
        font_found = True
    
    # 폰트가 없으면 기본 상태로 진행
    if not font_found:
        st.warning("시스템에 한글 폰트가 없습니다. 폰트 설치를 시도합니다.")
        try:
            download_nanum_font()
            # 폰트 다시 확인
            font_manager._rebuild()
            if "NanumGothic" in [f.name for f in font_manager.fontManager.ttflist]:
                font_name = "NanumGothic"
                font_found = True
        except:
            st.error("폰트 설치에 실패했습니다. 그래프의 한글이 제대로 표시되지 않을 수 있습니다.")
    
    # 폰트 설정
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = [font_name, 'sans-serif']
    plt.rcParams['axes.unicode_minus'] = False  # 마이너스 기호 깨짐 방지
    
    # 테스트 폰트로 사용할 서체 설정
    mpl.rcParams['font.sans-serif'] = [font_name, 'DejaVu Sans', 'Bitstream Vera Sans',
                                 'Computer Modern Sans Serif', 'Lucida Grande',
                                 'Verdana', 'Geneva', 'Lucid', 'Arial', 'Helvetica',
                                 'Avant Garde', 'sans-serif']

def create_custom_profile_graph(df):
    if len(df) < 2:
        st.error("최소 2개의 점이 필요합니다.")
        return None
    
    # 데이터프레임에서 시간, 온도, 습도 추출
    time_points = df['시간(h)'].tolist()
    temp_points = df['온도(°C)'].tolist()
    humidity_points = df['습도(%)'].tolist()
    
    # 그래프 생성
    fig, ax1 = plt.subplots(figsize=(10, 6))
    
    # 온도 그래프 (왼쪽 y축)
    ax1.set_xlabel('시간(h)')
    ax1.set_ylabel('온도(°C)', color='red')
    ax1.plot(time_points, temp_points, 'r-', linewidth=2, marker='o', label='온도')
    ax1.tick_params(axis='y', labelcolor='red')
    ax1.grid(True, linestyle='--', alpha=0.7)
    
    # 습도 그래프 (오른쪽 y축)
    ax2 = ax1.twinx()
    ax2.set_ylabel('습도(% R.H.)', color='blue')
    ax2.plot(time_points, humidity_points, 'b-', linewidth=2, marker='o', label='습도')
    ax2.tick_params(axis='y', labelcolor='blue')
    
    # 주요 시간 포인트에 수직선 추가
    for t in time_points[1:-1]:  # 처음과 마지막 제외
        ax1.axvline(x=t, color='gray', linestyle='--', alpha=0.5)
    
    # X축 간격 설정
    ax1.set_xticks(time_points)
    ax1.set_xticklabels([f"{t}" for t in time_points])
    
    # 범례 추가 (ASCII로 변경)
    ax1.legend(['Temperature'], loc='upper left')
    ax2.legend(['Humidity'], loc='upper right')
    
    # 그리드 설정
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # 그래프 제목 (ASCII로 변경)
    plt.title('Temperature & Humidity Profile')
    
    # 그래프 여백 조정
    plt.tight_layout()
    
    return fig

def get_image_download_link(fig, filename="temperature_profile.png", text="Download Graph"):
    """이미지 다운로드 링크 생성"""
    if fig is None:
        return ""
    
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=300)
    buf.seek(0)
    b64 = base64.b64encode(buf.read()).decode()
    href = f'<a href="data:image/png;base64,{b64}" download="{filename}">{text}</a>'
    return href

def main():
    # 앱 시작 시 한글 폰트 설정
    set_matplotlib_korean_font()
    
    st.title("온습도 프로파일 생성기 - 사용자 정의 포인트")
    
    # 사용 방법 설명
    st.markdown("""
    ### 사용 방법
    1. 아래에서 포인트를 추가하세요 (시간, 온도, 습도)
    2. 여러 포인트를 추가하여 프로파일을 생성하세요
    3. 필요 없는 포인트를 삭제할 수 있습니다
    4. 최소 2개 이상의 포인트가 필요합니다
    """)
    
    # 폰트 체크 섹션 (문제 해결용)
    if st.checkbox("폰트 문제 해결"):
        st.write("시스템에서 사용 가능한 한글 폰트:")
        available_fonts = [f.name for f in fm.fontManager.ttflist if any(korean in f.name.lower() for korean in ['hangul', 'korean', 'malgun', 'gothic', 'nanum', 'batang', 'gulim', 'noto'])]
        st.write(available_fonts)
        
        system_info = f"운영체제: {platform.system()}, 버전: {platform.version()}"
        st.write(system_info)
        
        if not available_fonts:
            st.error("한글 폰트가 발견되지 않았습니다.")
            if st.button("나눔 폰트 설치 시도"):
                download_nanum_font()
        else:
            st.success(f"발견된 한글 폰트: {', '.join(available_fonts)}")
    
    # 세션 상태에 데이터프레임 초기화
    if 'profile_df' not in st.session_state:
        st.session_state.profile_df = pd.DataFrame(columns=['시간(h)', '온도(°C)', '습도(%)'])
        # 기본 시작점과 끝점 추가
        st.session_state.profile_df = pd.concat([
            st.session_state.profile_df,
            pd.DataFrame([{'시간(h)': 0, '온도(°C)': 25, '습도(%)': 0}])
        ], ignore_index=True)
    
    # 새 포인트 추가 섹션
    st.subheader("새 포인트 추가")
    col1, col2, col3, col4 = st.columns([2, 2, 2, 1])
    
    with col1:
        new_time = st.number_input("시간(h)", value=0.0, min_value=0.0, step=0.5)
    with col2:
        new_temp = st.number_input("온도(°C)", value=25, min_value=-50, max_value=200)
    with col3:
        new_humidity = st.number_input("습도(%)", value=0, min_value=0, max_value=100)
    with col4:
        if st.button("추가", key="add_point"):
            # 이미 같은 시간이 있는지 확인
            if new_time in st.session_state.profile_df['시간(h)'].values:
                st.warning(f"시간 {new_time}h에 이미 포인트가 존재합니다. 먼저 삭제하거나 다른 시간을 선택하세요.")
            else:
                new_point = pd.DataFrame([{
                    '시간(h)': new_time,
                    '온도(°C)': new_temp,
                    '습도(%)': new_humidity
                }])
                st.session_state.profile_df = pd.concat([st.session_state.profile_df, new_point], ignore_index=True)
                st.session_state.profile_df = st.session_state.profile_df.sort_values('시간(h)').reset_index(drop=True)
    
    # 표시할 데이터프레임
    st.subheader("프로파일 포인트")
    
    # 포인트 테이블 표시 및 편집
    edited_df = st.data_editor(
        st.session_state.profile_df, 
        num_rows="dynamic",
        key="profile_editor",
        column_config={
            "시간(h)": st.column_config.NumberColumn(min_value=0.0, step=0.5),
            "온도(°C)": st.column_config.NumberColumn(min_value=-50, max_value=200),
            "습도(%)": st.column_config.NumberColumn(min_value=0, max_value=100)
        },
        use_container_width=True
    )
    
    # 편집된 데이터프레임 저장
    st.session_state.profile_df = edited_df.sort_values('시간(h)').reset_index(drop=True)
    
    # 미리 정의된 템플릿 버튼
    st.subheader("미리 정의된 템플릿")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("표준 85°C/85% 테스트 프로파일"):
            # 기본 85/85 테스트 프로파일
            st.session_state.profile_df = pd.DataFrame([
                {'시간(h)': 0.0, '온도(°C)': 25, '습도(%)': 0},
                {'시간(h)': 0.5, '온도(°C)': 55, '습도(%)': 50},
                {'시간(h)': 1.0, '온도(°C)': 85, '습도(%)': 85},
                {'시간(h)': 1001.0, '온도(°C)': 85, '습도(%)': 85},
                {'시간(h)': 1002.0, '온도(°C)': 25, '습도(%)': 0},
                {'시간(h)': 1004.0, '온도(°C)': 25, '습도(%)': 0}
            ])
    
    with col2:
        if st.button("모든 포인트 삭제"):
            st.session_state.profile_df = pd.DataFrame(columns=['시간(h)', '온도(°C)', '습도(%)'])
            # 기본 시작점 추가
            st.session_state.profile_df = pd.concat([
                st.session_state.profile_df,
                pd.DataFrame([{'시간(h)': 0, '온도(°C)': 25, '습도(%)': 0}])
            ], ignore_index=True)
    
    # 그래프 표시 언어 선택
    language_option = st.radio(
        "그래프 언어 선택:",
        ("한글", "영어"),
        horizontal=True
    )
    
    # 그래프 생성
    if len(st.session_state.profile_df) >= 2:
        if language_option == "영어":
            # 영어로 그래프 생성
            fig = create_custom_profile_graph(st.session_state.profile_df)
        else:
            # 한글로 그래프 생성 (기존 함수 사용)
            fig = create_custom_profile_graph(st.session_state.profile_df)
            
        if fig:
            st.pyplot(fig)
            # 다운로드 링크 생성
            st.markdown(get_image_download_link(fig), unsafe_allow_html=True)
    else:
        st.warning("그래프를 생성하려면 최소 2개의 포인트가 필요합니다.")

if __name__ == "__main__":
    main()
