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

def create_stepped_profile_graph(df, x_axis_unit='hours'):
    if len(df) < 1:
        st.error("At least 1 point is required.")
        return None

    set_matplotlib_font()

    # Check segments to truncate (segments with 1000+ hours)
    segment_times = df['Segment Time'].tolist()
    truncated_segments = [time >= 1000 for time in segment_times]
    
    # 시간 단위 변환 (시간 -> 분)
    if x_axis_unit == 'minutes':
        segment_times = [time * 60 for time in segment_times]  # 시간을 분으로 변환
    
    # 최소 영역 보장 (상대적 비율을 고려하여 균형잡힌 표시)
    threshold_time = 1.0 if x_axis_unit == 'hours' else 60  # 1시간 또는 60분
    min_display_time = threshold_time * 2  # 최소 표시 시간 (2시간 또는 120분)
    
    # 전체 시간 범위 계산
    total_time = sum(segment_times)
    max_time = max(segment_times)
    
    # 상대적 최소 표시 시간 계산 (전체 시간의 일정 비율)
    relative_min_time = max(min_display_time, total_time * 0.05)  # 전체 시간의 5% 또는 최소 표시 시간 중 큰 값
    
    # 0.5시간 구간의 표시 크기 계산 (기준값)
    base_0_5_time = 0.5 if x_axis_unit == 'hours' else 30  # 0.5시간 또는 30분
    expanded_0_5_time = base_0_5_time * 2  # 0.5시간을 두배로 확장한 크기
    reference_size = max(expanded_0_5_time, relative_min_time)  # 0.5시간 구간의 실제 표시 크기
    
    # 10시간 이하 기준 설정 (분 단위일 때는 600분)
    max_equal_size = 10 if x_axis_unit == 'hours' else 600  # 10시간 또는 600분
    
    adjusted_segment_times = []
    for time in segment_times:
        if time <= max_equal_size:  # 10시간(600분) 이하인 경우 모두 0.5시간(30분) 구간과 같은 크기로 표시
            adjusted_segment_times.append(reference_size)
        else:
            # 10시간(600분) 초과인 경우 최소 표시 시간보다 크게 보장
            adjusted_segment_times.append(max(time, min_display_time))
    
    # Calculate cumulative display time
    cumulative_display_time = [0]  # Start at 0
    current_time = 0
    
    # Check if we need proportional scaling (원본 시간 기준)
    original_segment_times = df['Segment Time'].tolist()
    has_1000plus = any(time >= 1000 for time in original_segment_times)
    has_30_to_1000 = any(30 < time < 1000 for time in original_segment_times)
    
    # Calculate display time for each segment
    if has_1000plus and has_30_to_1000:
        # Both types exist: scale proportionally within 30h limit
        max_time = max(original_segment_times)
        for i, original_time in enumerate(original_segment_times):
            if original_time >= 30:  # Scale segments >= 30 hours proportionally
                scaled_time = (original_time / max_time) * 30
                # 시간 단위 변환 적용
                if x_axis_unit == 'minutes':
                    scaled_time *= 60
                current_time += scaled_time
            else:
                # 최소 영역 보장 적용
                current_time += adjusted_segment_times[i]
            cumulative_display_time.append(current_time)
    else:
        # Original logic: only 1000+ hours compressed to 30 hours
        for i, original_time in enumerate(original_segment_times):
            if original_time >= 1000:
                compressed_time = 30  # Always show as 30 hours for segments >= 1000h
                # 시간 단위 변환 적용
                if x_axis_unit == 'minutes':
                    compressed_time *= 60
                current_time += compressed_time
            else:
                # 최소 영역 보장 적용
                current_time += adjusted_segment_times[i]
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
    
    # Process first segment
    segment_start = cumulative_display_time[0]
    segment_end = cumulative_display_time[1]
    
    for t in np.linspace(segment_start, segment_end, 20):
        detailed_times.append(t)
        detailed_temps.append(first_temp)
    
    # 첫 번째 구간 습도 데이터 생성
    current_times = [display_times[0]]
    current_humidity_data = [humidity_points[0]]
    
    if not humidity_zeros[0]:  # 첫 구간이 0이 아니면
        segment_start = cumulative_display_time[0]
        segment_end = cumulative_display_time[1]
        for t in np.linspace(segment_start, segment_end, 50):
            current_times.append(t)
            current_humidity_data.append(first_humidity)
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
        
        # Generate temperature data for this segment
        for t in np.linspace(segment_start, segment_end, 20):
            # Calculate ratio for temperature interpolation
            ratio = (t - segment_start) / (segment_end - segment_start) if segment_end > segment_start else 0
            
            detailed_times.append(t)
            detailed_temps.append(current_temp + ratio * (next_temp - current_temp))
        
        # 현재 구간의 습도 처리
        current_times = [display_times[i]]
        current_humidity_data = [humidity_points[i]]
        
        # 이전 구간과 현재 구간 모두 0이 아닌 경우 (값->값)
        if not humidity_zeros[i-1] and not humidity_zeros[i]:
            for t in np.linspace(segment_start, segment_end, 50):
                ratio = (t - segment_start) / (segment_end - segment_start) if segment_end > segment_start else 0
                h_value = current_humidity + ratio * (next_humidity - current_humidity)
                current_times.append(t)
                current_humidity_data.append(h_value)
            
            # 구간 데이터 저장
            if len(current_times) > 0:
                time_segments.append(current_times)
                humidity_segments.append(current_humidity_data)
        
        # 이전 구간이 0, 현재 구간이 0이 아닌 경우 (0->값)
        elif humidity_zeros[i-1] and not humidity_zeros[i]:
            # 새 세그먼트 시작 (현재 구간 시작점부터 끝까지 다음 습도값으로 고정)
            current_times = [display_times[i]]
            current_humidity_data = [next_humidity]
            for t in np.linspace(segment_start, segment_end, 50):
                current_times.append(t)
                current_humidity_data.append(next_humidity)
            if len(current_times) > 0:
                time_segments.append(current_times)
                humidity_segments.append(current_humidity_data)
        
        # 이전 구간이 0이 아니고 현재 구간이 0인 경우 (값->0)
        # 이 경우 이전 습도 세그먼트에서 이미 처리됨, 여기서는 아무것도 안함
        
        # 이전 구간도 0이고 현재 구간도 0인 경우 (0->0)
        # 아무것도 추가하지 않음
    
    # Create graph with adjusted figure size to accommodate legend
    fig, ax1 = plt.subplots(figsize=(14, 7))  # Increased width for legend space
    
    # Set up the first axis for Temperature
    ax1.set_ylabel('Temperature(°C)', color='red', fontsize=12)
    ax1.tick_params(axis='y', labelcolor='red')
    ax1.grid(True, linestyle='--', alpha=0.7)
    
    # Set up the second axis for Humidity
    ax2 = ax1.twinx()
    ax2.set_ylabel('Humidity(% RH)', color='blue', fontsize=12)
    ax2.tick_params(axis='y', labelcolor='blue')
    
    # Set x-axis range
    ax1.set_xlim(0, cumulative_display_time[-1])
    
    # Remove x-axis ticks and labels
    ax1.set_xticks([])  # Remove x-axis ticks
    ax1.set_xlabel('')  # Remove x-axis label
    
    # Add time unit label at the right end of x-axis
    time_unit_label = 'Time (h)' if x_axis_unit == 'hours' else 'Time (min)'
    ax1.text(cumulative_display_time[-1], -0.06, time_unit_label, horizontalalignment='right', 
             verticalalignment='top', transform=ax1.get_xaxis_transform(), fontsize=10)
    
    # Y축 범위 조정 - 온도
    min_temp = min(min(temp_points) if temp_points else 0, 0)
    max_temp = max(temp_points) if temp_points else 0
    
    # 온도 Y축 범위 설정 - 120 이상도 가능하도록 수정
    y_min_temp = min_temp * 1.1  # 최소값에 여유 공간 추가
    if y_min_temp > 0:  # 모든 값이 양수인 경우
        y_min_temp = 0

    # 최대값이 120보다 큰 경우 동적으로 조정
    if max_temp > 120:
        y_max_temp = max_temp * 1.1  # 최대값에 10% 여유 공간 추가
    else:
        y_max_temp = 120  # 기본값 120
    
    ax1.set_ylim(y_min_temp, y_max_temp)
    
    # 습도 Y축 범위 설정 (항상 0-100%)
    ax2.set_ylim(0, 100)
    
    # Plot temperature on ax1
    # temp_line = ax1.plot(display_times, temp_points, 'ro', markersize=4, label='Temperature(°C) Point')[0]  # 포인트 표시 제거
    temp_detailed = ax1.plot(detailed_times, detailed_temps, color='tomato', linewidth=1.0, label='Temperature(°C) Line')[0]
    
    # Plot humidity on ax2 (0이 아닌 값만 표시)
    # 포인트 마커는 0이 아닌 습도 값만 표시
    # 포인트 마커 표시 제거
    
    # 습도 선 그리기 (각 세그먼트별로)
    if len(humidity_segments) > 0:
        # 각 세그먼트마다 별도의 선으로 그림
        for i, (times, humidity_data) in enumerate(zip(time_segments, humidity_segments)):
            if i == 0:
                humid_detailed = ax2.plot(times, humidity_data, color='blue', linestyle='-', linewidth=0.5, label='Humidity(% RH Line)')[0]
            else:
                ax2.plot(times, humidity_data, color='blue', linestyle='-', linewidth=0.5)
    else:
        humid_detailed = ax2.plot([], [], color='blue', linestyle='-', linewidth=0.5, label='Humidity(% RH Line)')[0]
    
    # 각 구간 경계(0 포함)에 x축에 붙은 회색 수직 바 추가
    for i, t in enumerate(cumulative_display_time):
        # 모든 t (0 포함)에 대해 바를 그림
        ax1.axvline(x=t, color='black', linestyle=(0, (1, 1)), alpha=0.6, ymin=-0.1, ymax=1.0)
        ax1.plot([t, t], [0, -0.04], color='gray', linewidth=2, transform=ax1.get_xaxis_transform(), clip_on=False)
    
    # Add arrows under x-axis for each segment
    arrow_style = dict(arrowstyle='<->', color='black', linewidth=1)
    
    # Add arrows for each segment with actual time labels
    for i in range(len(original_segment_times)):
        start = cumulative_display_time[i]
        end = cumulative_display_time[i+1]
        y_pos = -0.02
        ax1.plot([start, start], [y_pos, y_pos - 0.01], color='black', linewidth=1, transform=ax1.get_xaxis_transform())
        ax1.annotate('', xy=(start, y_pos), xytext=(end, y_pos), xycoords=('data', 'axes fraction'), textcoords=('data', 'axes fraction'), arrowprops=arrow_style)
        actual_duration = original_segment_times[i]
        # 0인 구간은 텍스트 표시하지 않음
        if actual_duration != 0:
            # 시간 단위에 따라 표시 형식 조정
            if x_axis_unit == 'minutes':
                # 분 단위로 표시
                if actual_duration < 1:
                    time_str = f'{actual_duration * 60:.0f}m'
                else:
                    time_str = f'{int(actual_duration * 60)}m'
            else:
                # 시간 단위로 표시
                if actual_duration < 1:
                    time_str = f'{actual_duration:.1f}h'
                else:
                    time_str = f'{int(actual_duration)}h'
            ax1.text((start + end) / 2, y_pos - 0.02, time_str, horizontalalignment='center', verticalalignment='top', transform=ax1.get_xaxis_transform(), fontsize=10)
    
    # Update graph title to reflect compression
    plt.title('Temperature-Humidity Profile', fontsize=14)
    
    # Combine legends from both axes and place outside the plot area on the upper right
    lines = [temp_detailed, humid_detailed]
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='upper left', bbox_to_anchor=(1.05, 1.0))
    
    # Adjust layout to prevent legend from being cut off
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.15, right=0.85)  # Make room for legend and arrows
    
    return fig

def get_image_download_link(fig, filename="temperature_profile.png", text="Download Graph"):
    if fig is None:
        return ""
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=300, bbox_inches='tight')  # Added bbox_inches='tight'
    buf.seek(0)
    b64 = base64.b64encode(buf.read()).decode()
    return f'<a href="data:image/png;base64,{b64}" download="{filename}">{text}</a>'

def main():
    set_matplotlib_font()
    st.title("Temperature-Humidity Profile Generator - With Row Deletion")

    # 세션 상태 초기화
    if 'profile_df' not in st.session_state:
        st.session_state.profile_df = pd.DataFrame(
            [{'No.': 1, 'Segment Time': '', 'Temperature(°C)': '', 'Humidity(%)': ''}]
        )
    
    # X축 단위 선택 상태 초기화
    if 'x_axis_unit' not in st.session_state:
        st.session_state.x_axis_unit = 'hours'
    
    # 체크박스 선택 상태 초기화
    if 'selected_rows' not in st.session_state:
        st.session_state.selected_rows = set()

    # X축 단위 선택 UI
    st.subheader("X-axis Unit Selection")
    x_axis_option = st.radio(
        "Choose X-axis unit:",
        ["Hours (시간)", "Minutes (분)"],
        index=0 if st.session_state.x_axis_unit == 'hours' else 1,
        horizontal=True
    )
    
    # 선택된 단위를 세션 상태에 저장
    if x_axis_option == "Hours (시간)":
        st.session_state.x_axis_unit = 'hours'
    else:
        st.session_state.x_axis_unit = 'minutes'

    st.subheader("Profile Points (Editable)")
    
    # 항상 컬럼 순서 맞추기
    display_df = st.session_state.profile_df.copy()
    display_df = display_df.reindex(columns=['No.', 'Segment Time', 'Temperature(°C)', 'Humidity(%)'])

    # 체크박스와 행 삭제 버튼을 위한 레이아웃
    col1, col2 = st.columns([1, 4])
    
    with col1:
        st.write("**Select**")
        checkboxes = []
        for i in range(len(display_df)):
            checkbox_key = f"checkbox_{i}"
            checkbox = st.checkbox("", key=checkbox_key, value=i in st.session_state.selected_rows)
            checkboxes.append(checkbox)
            if checkbox:
                st.session_state.selected_rows.add(i)
            elif i in st.session_state.selected_rows:
                st.session_state.selected_rows.remove(i)
    
    with col2:
        # 편집 가능한 데이터 에디터 표시
        edited_df = st.data_editor(
            display_df,
            key="profile_editor",
            use_container_width=True,
            num_rows="dynamic",
            column_config={
                "No.": st.column_config.NumberColumn(
                    "No.",
                    help="자동 순번 (수정 불가)",
                    format="%d",
                    disabled=True,
                ),
            },
            hide_index=True,
        )

    # 선택된 행 삭제 버튼
    if st.button("🗑️ Delete Selected Rows", type="secondary"):
        if st.session_state.selected_rows:
            # 현재 테이블 행 수 확인
            current_row_count = len(st.session_state.profile_df)
            selected_count = len(st.session_state.selected_rows)
            
            # 모든 행을 삭제하려는 경우 방지
            if selected_count >= current_row_count:
                st.error("Cannot delete all rows. At least one row must remain in the table.")
            else:
                # 선택된 행들을 삭제 (역순으로 정렬하여 인덱스 문제 방지)
                rows_to_delete = sorted(st.session_state.selected_rows, reverse=True)
                
                for row_idx in rows_to_delete:
                    if row_idx < len(st.session_state.profile_df):
                        st.session_state.profile_df = st.session_state.profile_df.drop(
                            st.session_state.profile_df.index[row_idx]
                        ).reset_index(drop=True)
                
                # No. 컬럼 재정렬
                if len(st.session_state.profile_df) > 0:
                    st.session_state.profile_df['No.'] = range(1, len(st.session_state.profile_df) + 1)
                
                # 선택 상태 초기화
                st.session_state.selected_rows = set()
                
                st.success(f"{len(rows_to_delete)} row(s) deleted successfully!")
                st.rerun()
        else:
            st.warning("No rows selected for deletion.")

    # 편집된 데이터 처리
    if not edited_df.equals(display_df):
        # 마지막 행의 Segment Time이 None, NaN, '' 인지 확인하고 처리
        if len(edited_df) > 0:
            last_row_segment_time = edited_df.iloc[-1]['Segment Time']
            if pd.isna(last_row_segment_time) or last_row_segment_time is None or last_row_segment_time == '':
                edited_df = edited_df.iloc[:-1]
                st.info("Last row with empty Segment Time has been removed.")

        # No 컬럼이 int로 변환 불가한 값이 있으면 NaN으로 처리
        if len(edited_df) > 0:
            no_col = pd.to_numeric(edited_df['No.'], errors='coerce')
            # No 컬럼이 NaN이거나, 오름차순이 아니면 자동으로 재정렬
            if (
                'No.' not in edited_df.columns
                or no_col.isnull().any()
                or not np.array_equal(
                    no_col.dropna().sort_values().values,
                    np.arange(1, len(no_col.dropna()) + 1)
                )
            ):
                edited_df['No.'] = range(1, len(edited_df) + 1)

        # 컬럼 순서 맞추기
        edited_df = edited_df.reindex(columns=['No.', 'Segment Time', 'Temperature(°C)', 'Humidity(%)'])

        st.session_state.profile_df = edited_df
        
        # 선택 상태 업데이트 (삭제된 행들 제거)
        st.session_state.selected_rows = {i for i in st.session_state.selected_rows if i < len(edited_df)}
        
        st.rerun()

    # 전체 선택/해제 버튼
    col_select1, col_select2 = st.columns(2)
    with col_select1:
        if st.button("✅ Select All"):
            st.session_state.selected_rows = set(range(len(st.session_state.profile_df)))
            st.rerun()
    
    with col_select2:
        if st.button("❌ Deselect All"):
            st.session_state.selected_rows = set()
            st.rerun()

    # Check for segments over 1000 hours
    if not st.session_state.profile_df.empty:
        # 'Segment Time'을 숫자로 변환 (에러시 NaN)
        profile_df_numeric = st.session_state.profile_df.copy()
        profile_df_numeric['Segment Time'] = pd.to_numeric(profile_df_numeric['Segment Time'], errors='coerce')
        
        # Check for segments >= 1000 hours
        long_segments = profile_df_numeric[profile_df_numeric['Segment Time'] >= 1000]
        if not long_segments.empty:
            st.info(f"{len(long_segments)} segment(s) are ≥1000 hours and will be displayed as 30 hours.")
        
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
            fig = create_stepped_profile_graph(df_with_segment, st.session_state.x_axis_unit)
            if fig:
                st.pyplot(fig)
                st.markdown(get_image_download_link(fig), unsafe_allow_html=True)
    else:
        st.warning("At least 2 points are required to generate a graph.")

if __name__ == "__main__":
    main()
