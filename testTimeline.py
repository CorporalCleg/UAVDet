import streamlit as st
from streamlit_timeline import st_timeline

st.set_page_config(layout="wide")

items = [
   {
      "id": 1,
      "content": "Start Video",
      "start": 0
    },
    {
      "id": 2,
      "content": "1 Seconds",
      "start": 1000
    },
    {
      "id": 3,
      "content": "2 Seconds",
      "start": 2000
    },
    {
      "id": 4,
      "content": "3 Seconds",
      "start": 4000
    }
]

options = {
    #"start": 0,  # Начальное время
    #"end": 4000000,    # Конечное время (4 секунды)
    "min": 0,
    "max": 7200000,
    #"zoomMin": 1000,  # Минимальный интервал для зума (миллисекунды)
    #"zoomMax": 60000,  # Максимальный интервал для зума (миллисекунды)
    "timeAxis": {"scale": "minute", "step": 1},

    "moment": {
        "format": "HH:mm:ss",
        "locale": "en",
        "timezone": "UTC"
    },
    
    "format": {
        "minorLabels": {
            "second": "s",
            "minute": "mm:ss",
            "hour": "HH:mm:ss"
        },
        "majorLabels": {
            "second": "s",
            "minute": "HH:mm:ss",
            "hour": "HH:mm:ss"
        }
    }
}

timeline = st_timeline(items, groups=[], options=options, height="300px")
st.subheader("Selected item")
st.write(timeline)
#st.subheader("Selected item")
#st.write(obj)

# Отображение видео
#video_file = open('video.mp4', 'rb')
#video_bytes = video_file.read()
#video_container = st.empty()
#video_container.video(video_bytes, start_time=10)
#time.sleep(10)
#video_container.video(video_bytes, start_time=50)
