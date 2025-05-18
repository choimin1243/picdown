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
    
    # Generate detailed time, temperature, and humidity data
    detailed_times = []
    detailed_temps = []
    detailed_humidity = []
    
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
            detailed_humidity.append(first_humidity)  # First segment has constant value
    else:
        # Less than 90 hours - include the whole segment
        segment_start = cumulative_display_time[0]
        segment_end = cumulative_display_time[1]
        
        for t in np.linspace(segment_start, segment_end, 20):
            detailed_times.append(t)
            detailed_temps.append(first_temp)
            detailed_humidity.append(first_humidity)
    
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
                detailed_humidity.append(current_humidity + real_ratio * (next_humidity - current_humidity))
        else:
            # Less than 90 hour segment - normal processing
            for t in np.linspace(segment_start, segment_end, 20)[1:]:
                ratio = (t - segment_start) / (segment_end - segment_start) if segment_end > segment_start else 0
                
                detailed_times.append(t)
                detailed_temps.append(current_temp + ratio * (next_temp - current_temp))
                detailed_humidity.append(current_humidity + ratio * (next_humidity - current_humidity))
    
    # Create graph
    fig, ax1 = plt.subplots(figsize=(12, 7))
    ax1.set_ylabel('Temperature(°C) and Humidity(%)', fontsize=12)
    ax1.tick_params(axis='y')
    ax1.grid(True, linestyle='--', alpha=0.7)
    
    # Set x-axis range
    ax1.set_xlim(0, cumulative_display_time[-1])
    
    # Remove x-axis ticks and labels
    ax1.set_xticks([])  # Remove x-axis ticks
    ax1.set_xlabel('')  # Remove x-axis label
    
    # Plot temperature/humidity graphs
    display_times = [cumulative_display_time[i+1] for i in range(len(temp_points))]
    ax1.plot(display_times, temp_points, 'ro', markersize=4, label='Temperature(°C) Point')
    ax1.plot(detailed_times, detailed_temps, 'r-', linewidth=2.5, label='Temperature(°C) Line')
    ax1.plot(display_times, humidity_points, 'bo', markersize=4, label='Humidity(%) Point')
    ax1.plot(detailed_times, detailed_humidity, 'b--', linewidth=2.5, label='Humidity(%) Line')
    
    # Set y-axis range based on temperature and humidity
    max_temp = max(temp_points) if temp_points else 0
    max_humidity = max(humidity_points) if humidity_points else 0
    
    # Determine the appropriate y-axis limit
    if max_temp > 100 or max_humidity > 100:
        y_max = max(150, max_temp * 1.1, max_humidity * 1.1)  # At least 150 or 110% of the maximum value
        ax1.set_ylim(0, y_max)
    else:
        ax1.set_ylim(0, 100)
    
    # Add segment divider lines
    for i, t in enumerate(cumulative_display_time):
        if t > 0:  # Don't draw line at start point (0)
            if i < len(truncated_segments) and truncated_segments[i-1]:
                # Thicker line after truncated segments
                ax1.axvline(x=t, color='gray', linestyle='-', linewidth=1.5, alpha=0.7)
            else:
                ax1.axvline(x=t, color='gray', linestyle='--', alpha=0.5)
    
    # Add arrows under x-axis for each segment
    arrow_style = dict(arrowstyle='<->', color='black', linewidth=1)
    
    # Add arrows for each segment
    for i in range(len(segment_times)):
        start = cumulative_display_time[i]    # Current segment start
        end = cumulative_display_time[i+1]    # Current segment end
        
        # Draw arrow (all arrows at same y position)
        y_pos = -0.02  # y position for all arrows
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
    
    # Add wave patterns for truncated segments
    for i in range(len(segment_times)):
        if truncated_segments[i]:
            x_pos = cumulative_display_time[i]
            y_range = ax1.get_ylim()
            y_min, y_max = y_range
            y_step = (y_max - y_min) / 10
            
            # Add multiple wave patterns to create grid-like appearance
            for j in range(1, 10):
                y_pos = y_min + j * y_step
                # Bold wave pattern
                ax1.text(x_pos + 0.05, y_pos, "//", fontsize=14, color='gray', alpha=0.7)
                
            # Add vertical box to indicate wave area
            rect = plt.Rectangle((x_pos, y_min), 0.15, y_max-y_min, 
                                fill=True, color='lightgray', alpha=0.1,
                                linewidth=0, zorder=0)
            ax1.add_patch(rect)
    
    # Adjust margins (to make room for arrows)
    plt.subplots_adjust(bottom=0.15)
    
    # Set graph title
    plt.title('Temperature-Humidity Profile (Segments over 90h show last 1h only)', fontsize=14)
    
    # Add legend
    ax1.legend(loc='upper left')
    
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
                st.session_state.profile_df = pd.concat([st.session_state.profile_df, new_row], ignore_index=True)
                st.rerun()
        except ValueError:
            st.warning("Please enter numeric values.")

    st.subheader("Profile Points (Editable)")
    edited_df = st.data_editor(
        st.session_state.profile_df,
        key="profile_editor",
        use_container_width=True,
        num_rows="dynamic"
    )

    # Check for data changes and update
    if not edited_df.equals(st.session_state.profile_df):
        st.session_state.profile_df = edited_df
        st.rerun()
        
    # Check for segments over 90 hours
    if not st.session_state.profile_df.empty:
        long_segments = st.session_state.profile_df[st.session_state.profile_df['Segment Time'] >= 90]
        if not long_segments.empty:
            st.info(f"{len(long_segments)} segment(s) over 90 hours will display only the last 1 hour.")

    # Auto-generate graph
    if len(st.session_state.profile_df) >= 2:
        df_with_segment = st.session_state.profile_df.copy()
        df_with_segment.insert(0, 'Segment Number', range(1, len(df_with_segment) + 1))
        df_with_segment['Segment Time'] = pd.to_numeric(df_with_segment['Segment Time'], errors='coerce')
        df_with_segment['Temperature(°C)'] = pd.to_numeric(df_with_segment['Temperature(°C)'], errors='coerce')
        df_with_segment['Humidity(%)'] = pd.to_numeric(df_with_segment['Humidity(%)'], errors='coerce')

        if df_with_segment[['Segment Time', 'Temperature(°C)', 'Humidity(%)']].isnull().any().any():
            st.warning("Non-numeric values detected. Please correct before generating the graph.")
        else:
            fig = create_stepped_profile_graph(df_with_segment)
            if fig:
                st.pyplot(fig)
                st.markdown(get_image_download_link(fig), unsafe_allow_html=True)
    else:
        st.warning("At least 2 points are required to generate a graph.")

if __name__ == "__main__":
    main()
