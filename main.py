import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import io
import base64
import pandas as pd
import matplotlib.font_manager as fm
import platform
from matplotlib.gridspec import GridSpec

def set_matplotlib_font():
    system_name = platform.system()
    if system_name == "Windows":
        font_name = "Arial"
    elif system_name == "Darwin":
        font_name = "Arial"
    else:
        font_name = "Arial"
    plt.rcParams['font.family'] = font_name
    plt.rcParams['axes.unicode_minus'] = False

def create_stepped_profile_graph(df):
    if len(df) < 1:
        st.error("At least 1 point is required.")
        return None

    set_matplotlib_font()

    # Check segments to truncate (segments with 90+ hours)
    segment_times = df['Segment Time'].tolist()
    truncated_segments = [time >= 90 for time in segment_times]
    
    # Calculate cumulative display time
    cumulative_display_time = [0]  # Start at 0
    current_time = 0
    
    # Calculate actual display time for each segment
    for i, time in enumerate(segment_times):
        if truncated_segments[i]:  # 90+ hour segment
            current_time += 1  # Show only the last 1 hour
        else:  # Less than 90 hour segment
            current_time += time  # Show full time
        cumulative_display_time.append(current_time)
    
    # Temperature and humidity points
    temp_points = df['Temperature(°C)'].tolist()
    humidity_points = df['Humidity(%)'].tolist()
    
    # 각 습도 포인트가 0인지 아닌지 확인
    humidity_zeros = [h == 0 for h in humidity_points]
    
    # Calculate display times for point markers
    display_times = [cumulative_display_time[i+1] for i in range(len(temp_points))]
    
    # Generate detailed time, temperature, and humidity data
    detailed_times = []
    detailed_temps = []
    
    # 각 구간별로 습도 데이터 분리 저장 (0인 구간 분리를 위해)
    humidity_segments = []
    time_segments = []
    
    # Handle first segment
    first_temp = temp_points[0]
    first_humidity = humidity_points[0]
    
    # Process first segment differently based on its length
    if truncated_segments[0]:
        # 90+ hours - include only the last hour
        segment_start = cumulative_display_time[0]  # 0
        segment_end = cumulative_display_time[1]    # 1
        real_start = segment_times[0] - 1           # Total time - 1
        real_end = segment_times[0]                 # Total time
        
        # Generate detailed data for the last hour
        for t in np.linspace(segment_start, segment_end, 20):
            # Convert t to real time (0-1 range -> (total time-1)-total time range)
            real_t = real_start + (t - segment_start) * (real_end - real_start) / (segment_end - segment_start)
            real_ratio = (real_t - real_start) / (real_end - real_start) if real_end > real_start else 0
            
            detailed_times.append(t)  # Display time
            detailed_temps.append(first_temp)  # First segment has constant value
    else:
        # Less than 90 hours - include the whole segment
        segment_start = cumulative_display_time[0]
        segment_end = cumulative_display_time[1]
        
        for t in np.linspace(segment_start, segment_end, 20):
            detailed_times.append(t)
            detailed_temps.append(first_temp)
    
    # 첫 번째 구간 습도 데이터 생성
    current_times = []
    current_humidity_data = []
    
    if not humidity_zeros[0]:  # 첫 구간이 0이 아니면
        segment_start = cumulative_display_time[0]
        segment_end = cumulative_display_time[1]
        # 더 많은 포인트를 생성하여 점선이 더 균일하게 표시되도록 함
        for t in np.linspace(segment_start, segment_end, 50):
            current_times.append(t)
            current_humidity_data.append(first_humidity)
        
        # 구간 데이터 저장
        time_segments.append(current_times)
        humidity_segments.append(current_humidity_data)
    
    # Process remaining segments
    for i in range(1, len(temp_points)):
        current_temp = temp_points[i-1]
        next_temp = temp_points[i]
        current_humidity = humidity_points[i-1]
        next_humidity = humidity_points[i]
        
        # Current segment's display start/end time
        segment_start = cumulative_display_time[i]
        segment_end = cumulative_display_time[i+1]
        
        if truncated_segments[i]:
            # 90+ hour segment - map real time to display time
            real_start = segment_times[i] - 1  # Total time - 1
            real_end = segment_times[i]        # Total time
            
            for t in np.linspace(segment_start, segment_end, 20)[1:]:
                # Convert t to real time
                real_t = real_start + (t - segment_start) * (real_end - real_start) / (segment_end - segment_start)
                real_ratio = (real_t - real_start) / (real_end - real_start) if real_end > real_start else 0
                
                detailed_times.append(t)  # Display time
                detailed_temps.append(current_temp + real_ratio * (next_temp - current_temp))
        else:
            # Less than 90 hour segment - normal processing
            for t in np.linspace(segment_start, segment_end, 20)[1:]:
                ratio = (t - segment_start) / (segment_end - segment_start) if segment_end > segment_start else 0
                
                detailed_times.append(t)  # Display time
                detailed_temps.append(current_temp + ratio * (next_temp - current_temp))
        
        # 현재 구간의 습도 처리
        current_times = []
        current_humidity_data = []
        
        # 이전 구간과 현재 구간 모두 0이 아닌 경우 (값->값)
        if not humidity_zeros[i-1] and not humidity_zeros[i]:
            # 더 많은 포인트를 생성하여 점선이 더 균일하게 표시되도록 함
            for t in np.linspace(segment_start, segment_end, 50)[1:]:
                ratio = (t - segment_start) / (segment_end - segment_start) if segment_end > segment_start else 0
                h_value = current_humidity + ratio * (next_humidity - current_humidity)
                current_times.append(t)
                current_humidity_data.append(h_value)
            
            # 마지막 구간인 경우 새 세그먼트 추가
            if len(current_times) > 0:
                time_segments.append(current_times)
                humidity_segments.append(current_humidity_data)
        
        # 이전 구간이 0, 현재 구간이 0이 아닌 경우 (0->값)
        elif humidity_zeros[i-1] and not humidity_zeros[i]:
            # 새 세그먼트 시작 (현재 구간 시작점부터 끝까지 다음 습도값으로 고정)
            # 더 많은 포인트를 생성하여 점선이 더 균일하게 표시되도록 함
            for t in np.linspace(segment_start, segment_end, 50):
                current_times.append(t)
                current_humidity_data.append(next_humidity)
                
            # 새 세그먼트 추가
            if len(current_times) > 0:
                time_segments.append(current_times)
                humidity_segments.append(current_humidity_data)
        
        # 이전 구간이 0이 아니고 현재 구간이 0인 경우 (값->0)
        # 이 경우 이전 습도 세그먼트에서 이미 처리됨, 여기서는 아무것도 안함
        
        # 이전 구간도 0이고 현재 구간도 0인 경우 (0->0)
        # 아무것도 추가하지 않음
    
    # Create graph
    fig, ax1 = plt.subplots(figsize=(12, 7))
    
    # Set up the first axis for Temperature
    ax1.set_ylabel('Temperature(°C)', color='red', fontsize=12)
    ax1.tick_params(axis='y', labelcolor='red')
    ax1.grid(True, linestyle='--', alpha=0.7)
    
    # Set up the second axis for Humidity
    ax2 = ax1.twinx()
    ax2.set_ylabel('Humidity(%) RH', color='blue', fontsize=12)
    ax2.tick_params(axis='y', labelcolor='blue')
    
    # Set x-axis range
    ax1.set_xlim(0, cumulative_display_time[-1])
    
    # Remove x-axis ticks and labels
    ax1.set_xticks([])  # Remove x-axis ticks
    ax1.set_xlabel('')  # Remove x-axis label
    
    # Y축 범위 조정 - 온도
    min_temp = min(min(temp_points) if temp_points else 0, 0)
    max_temp = max(temp_points) if temp_points else 0
    
    # 온도 Y축 범위 설정
    y_min_temp = min_temp * 1.1  # 최소값에 여유 공간 추가
    if y_min_temp > 0:  # 모든 값이 양수인 경우
        y_min_temp = 0
    
    if max_temp > 100:
        y_max_temp = max(150, max_temp * 1.1)  # At least 150 or 110% of the maximum value
    else:
        y_max_temp = 100
    
    ax1.set_ylim(y_min_temp, y_max_temp)
    
    # 습도 Y축 범위 설정 (항상 0-100%)
    ax2.set_ylim(0, 100)
    
    # Plot temperature on ax1
    temp_line = ax1.plot(display_times, temp_points, 'ro', markersize=4, label='Temperature(°C) Point')[0]
    temp_detailed = ax1.plot(detailed_times, detailed_temps, 'r-', linewidth=2.5, label='Temperature(°C) Line')[0]
    
    # Plot humidity on ax2 (0이 아닌 값만 표시)
    # 포인트 마커는 0이 아닌 습도 값만 표시
    non_zero_humidity_points = []
    non_zero_display_times = []
    for i, h in enumerate(humidity_points):
        if h != 0:
            non_zero_humidity_points.append(h)
            non_zero_display_times.append(display_times[i])
    
    # 습도 포인트 그리기 (0이 아닌 값만)
    if len(non_zero_humidity_points) > 0:
        humid_line = ax2.plot(non_zero_display_times, non_zero_humidity_points, 'bo', markersize=4, label='Humidity(%) RH Point')[0]
    else:
        # 습도 포인트가 없는 경우 더미 라인 생성 (레전드용)
        humid_line = ax2.plot([], [], 'bo', markersize=4, label='Humidity(%) RH Point')[0]
    
    # 습도 선 그리기 (각 세그먼트별로)
    if len(humidity_segments) > 0:
        # 각 세그먼트마다 별도의 선으로 그림
        for i, (times, humidity_data) in enumerate(zip(time_segments, humidity_segments)):
            if i == 0:
                # 점선 패턴 변경 - 더 촘촘하게 (1.5, 1.5)는 1.5픽셀 실선, 1.5픽셀 공백
                # 첫 번째 세그먼트에만 레이블 추가
                humid_detailed = ax2.plot(times, humidity_data, 'b--', linewidth=2.5, 
                                           dashes=[1.5, 1.5], label='Humidity(%) RH Line')[0]
            else:
                # 나머지 세그먼트는 레이블 없이 그림
                ax2.plot(times, humidity_data, 'b--', linewidth=2.5, dashes=[1.5, 1.5])
    else:
        # 습도 선이 없는 경우 더미 라인 생성 (레전드용)
        humid_detailed = ax2.plot([], [], 'b--', linewidth=2.5, 
                                    dashes=[1.5, 1.5], label='Humidity(%) RH Line')[0]
    
    # 각 구간 경계(0 제외)에 x축에 붙은 회색 수직 바 추가
    for i, t in enumerate(cumulative_display_time):
        if t > 0:  # Don't draw line at start point (0)
            ax1.axvline(x=t, color='black', linestyle='--', alpha=0.5, ymin=-0.1, ymax=1.0)
            ax1.plot([t, t], [0, -0.04], color='gray', linewidth=2, transform=ax1.get_xaxis_transform(), clip_on=False)
    
    # Add arrows under x-axis for each segment
    arrow_style = dict(arrowstyle='<->', color='black', linewidth=1)
    
    # Add arrows for each segment
    for i in range(len(segment_times)):
        start = cumulative_display_time[i]    # Current segment start
        end = cumulative_display_time[i+1]    # Current segment end
        
        # Draw vertical line before arrow
        y_pos = -0.02  # y position for all arrows
        ax1.plot([start, start], [y_pos, y_pos - 0.01], color='black', linewidth=1, transform=ax1.get_xaxis_transform())
        
        # Draw arrow (all arrows at same y position)
        ax1.annotate('', 
                    xy=(start, y_pos), 
                    xytext=(end, y_pos),
                    xycoords=('data', 'axes fraction'),
                    textcoords=('data', 'axes fraction'),
                    arrowprops=arrow_style)
        
        # Display segment time and truncation info
        if truncated_segments[i]:
            # For truncated segments, show real time with note
            time_str = f'{segment_times[i]}h (last 1h only)'  # Keep decimal points
        else:
            # For normal segments, show time with decimal points if needed
            duration = segment_times[i]
            time_str = f'{duration}h'  # Always show as entered, including decimal points
        
        # Display time text under arrow
        ax1.text((start + end) / 2, y_pos - 0.02,
                time_str,
                horizontalalignment='center',
                verticalalignment='top',
                transform=ax1.get_xaxis_transform(),
                fontsize=10)
    
    # Adjust margins (to make room for arrows)
    plt.subplots_adjust(bottom=0.15)
    
    # Set graph title
    plt.title('Temperature-Humidity Profile (Segments over 90h show last 1h only)', fontsize=14)
    
    # Combine legends from both axes
    lines = [temp_line, temp_detailed, humid_line, humid_detailed]
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='upper left')
    
    plt.tight_layout()
    return fig

def get_image_download_link(fig, filename="temperature_profile.png", text="Download Graph"):
    if fig is None:
        return ""
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=300)
    buf.seek(0)
    b64 = base64.b64encode(buf.read()).decode()
    return f'<a href="data:image/png;base64,{b64}" download="{filename}">{text}</a>'

def main():
    set_matplotlib_font()
    st.title("Temperature-Humidity Profile Generator - Segment Truncation Version")

    if 'profile_df' not in st.session_state:
        st.session_state.profile_df = pd.DataFrame(columns=['Segment Time', 'Temperature(°C)', 'Humidity(%)'])

    st.subheader("Add New Segment")
    col1, col2, col3 = st.columns([2, 2, 2])
    with col1:
        seg_time = st.text_input("Segment Time", value="1.0", key="seg_time")
    with col2:
        temp = st.text_input("Temperature(°C)", value="25", key="temp")
    with col3:
        humid = st.text_input("Humidity(%)", value="0", key="humid")

    if st.button("Add"):
        try:
            seg_time_val = float(seg_time)
            temp_val = float(temp)
            humid_val = float(humid)
            if seg_time_val <= 0:
                st.warning("Segment time must be greater than 0.")
            elif not (-50 <= temp_val <= 200):
                st.warning("Temperature must be between -50°C and 200°C.")
            elif not (0 <= humid_val <= 100):
                st.warning("Humidity must be between 0% and 100%.")
            else:
                new_row = pd.DataFrame([{'Segment Time': seg_time_val, 'Temperature(°C)': temp_val, 'Humidity(%)': humid_val}])
                # 빈 DataFrame인 경우 직접 할당, 아닌 경우 concat 사용
                if st.session_state.profile_df.empty:
                    st.session_state.profile_df = new_row
                else:
                    st.session_state.profile_df = pd.concat([st.session_state.profile_df, new_row], ignore_index=True, copy=False)
                st.rerun()
        except ValueError:
            st.warning("Please enter numeric values.")

    st.subheader("Profile Points (Editable)")
    
    # 데이터 준비 - 자동으로 순번 컬럼 추가
    if not st.session_state.profile_df.empty:
        display_df = st.session_state.profile_df.copy()
        # 번호 열 추가 (1부터 시작하는 자동 번호)
        display_df.insert(0, 'No.', range(1, len(display_df) + 1))
    else:
        display_df = st.session_state.profile_df.copy()
        display_df['No.'] = []
    
    # 편집 가능한 데이터 에디터 표시
    edited_df = st.data_editor(
        display_df,
        key="profile_editor",
        use_container_width=True,
        num_rows="dynamic",
        column_config={
            "No.": st.column_config.NumberColumn(
                "No.",
                help="자동 순번",
                disabled=True,  # 편집 불가능하게 설정
                format="%d",
            )
        },
        hide_index=True,  # 기본 인덱스 숨기기
    )
    
    # 편집된 데이터 처리 ('No.' 컬럼 제거 후 session_state에 저장)
    if not edited_df.equals(display_df):
        # 'No.' 열 제거 후 저장
        edited_df_without_no = edited_df.drop(columns=['No.'])
        
        # 마지막 행의 Segment Time이 None인지 확인하고 처리
        if len(edited_df_without_no) > 0:
            last_row_segment_time = edited_df_without_no.iloc[-1]['Segment Time']
            
            if pd.isna(last_row_segment_time) or last_row_segment_time is None:
                # 마지막 행 삭제
                edited_df_without_no = edited_df_without_no.iloc[:-1]
                st.info("Last row with empty Segment Time has been removed.")
        
        # 업데이트된 df와 원래 df 비교
        if not edited_df_without_no.equals(st.session_state.profile_df):
            st.session_state.profile_df = edited_df_without_no
            st.rerun()
        
    # Check for segments over 90 hours
    if not st.session_state.profile_df.empty:
        long_segments = st.session_state.profile_df[st.session_state.profile_df['Segment Time'] >= 90]
        if not long_segments.empty:
            st.info(f"{len(long_segments)} segment(s) over 90 hours will display only the last 1 hour.")
        
        # Also check for zero humidity segments
        zero_humidity_segments = st.session_state.profile_df[st.session_state.profile_df['Humidity(%)'] == 0]
        if not zero_humidity_segments.empty:
            st.info(f"{len(zero_humidity_segments)} segment(s) have 0% humidity and will not be displayed in the humidity graph.")

    # Auto-generate graph
    if len(st.session_state.profile_df) >= 2:
        df_with_segment = st.session_state.profile_df.copy()
        
        # Ensure all values are numeric and remove rows with NaN values in Segment Time
        df_with_segment['Segment Time'] = pd.to_numeric(df_with_segment['Segment Time'], errors='coerce')
        df_with_segment['Temperature(°C)'] = pd.to_numeric(df_with_segment['Temperature(°C)'], errors='coerce')
        df_with_segment['Humidity(%)'] = pd.to_numeric(df_with_segment['Humidity(%)'], errors='coerce')
        
        # Remove rows with NaN in Segment Time (additional check)
        df_with_segment = df_with_segment.dropna(subset=['Segment Time'])

        if df_with_segment[['Segment Time', 'Temperature(°C)', 'Humidity(%)']].isnull().any().any():
            st.warning("Non-numeric values detected. Please correct before generating the graph.")
        elif len(df_with_segment) < 2:
            st.warning("At least 2 points with valid data are required to generate a graph.")
        else:
            fig = create_stepped_profile_graph(df_with_segment)
            if fig:
                st.pyplot(fig)
                st.markdown(get_image_download_link(fig), unsafe_allow_html=True)
    else:
        st.warning("At least 2 points are required to generate a graph.")

if __name__ == "__main__":
    main()
