import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import io
import base64
import pandas as pd
import matplotlib.font_manager as fm
import platform
import koreanize_matplotlib

def set_matplotlib_korean_font():
    """한글 폰트 설정 함수"""
    system_name = platform.system()
    
    if system_name == "Windows":
        # 윈도우의 경우 맑은 고딕 폰트 사용
        font_name = "Malgun Gothic"
    elif system_name == "Darwin":  # macOS
        font_name = "AppleGothic"
    else:  # Linux 등
        font_name = "NanumGothic"
        
    # 폰트 설정
    plt.rcParams['font.family'] = font_name
    plt.rcParams['axes.unicode_minus'] = False  # 마이너스 기호 깨짐 방지

def create_custom_profile_graph(df):
    if len(df) < 2:
        st.error("최소 2개의 점이 필요합니다.")
        return None
    
    # 한글 폰트 설정
    set_matplotlib_korean_font()
    
    # 데이터프레임에서 시간, 온도, 습도 추출
    time_points = df['시간(h)'].tolist()
    temp_points = df['온도(°C)'].tolist()
    humidity_points = df['습도(%)'].tolist()
    
    # 그래프 생성
    fig, ax1 = plt.subplots(figsize=(10, 6))
    
    # 온도 그래프 (왼쪽 y축)
    ax1.set_xlabel('시간(h)')
    ax1.set_ylabel('온도(°C)', color='red')
    ax1.plot(time_points, temp_points, 'r-', linewidth=2, marker='o')
    ax1.tick_params(axis='y', labelcolor='red')
    ax1.grid(True, linestyle='--', alpha=0.7)
    
    # 습도 그래프 (오른쪽 y축)
    ax2 = ax1.twinx()
    ax2.set_ylabel('습도(% R.H.)', color='blue')
    ax2.plot(time_points, humidity_points, 'b-', linewidth=2, marker='o')
    ax2.tick_params(axis='y', labelcolor='blue')
    
    # 주요 시간 포인트에 수직선 추가
    for t in time_points[1:-1]:  # 처음과 마지막 제외
        ax1.axvline(x=t, color='gray', linestyle='--', alpha=0.5)
    
    # X축 간격 설정
    ax1.set_xticks(time_points)
    ax1.set_xticklabels([f"{t}" for t in time_points])
    
    # 그리드 설정
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # 그래프 제목
    plt.title('온습도 프로파일')
    
    # 그래프 여백 조정
    plt.tight_layout()
    
    return fig

def get_image_download_link(fig, filename="temperature_profile.png", text="그래프 다운로드"):
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
        available_fonts = [f.name for f in fm.fontManager.ttflist if any(korean in f.name.lower() for korean in ['hangul', 'korean', 'malgun', 'gothic', 'nanum', 'batang', 'gulim'])]
        st.write(available_fonts)
        
        system_info = f"운영체제: {platform.system()}, 버전: {platform.version()}"
        st.write(system_info)
        st.write("만약 위 목록에 한글 폰트가 없다면, 한글 폰트를 설치해야 합니다.")
    
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
    
    # 그래프 생성
    if len(st.session_state.profile_df) >= 2:
        fig = create_custom_profile_graph(st.session_state.profile_df)
        if fig:
            st.pyplot(fig)
            # 다운로드 링크 생성
            st.markdown(get_image_download_link(fig), unsafe_allow_html=True)
    else:
        st.warning("그래프를 생성하려면 최소 2개의 포인트가 필요합니다.")

if __name__ == "__main__":
    main()
