"""실행: python -m streamlit run app.py"""
import json
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import streamlit as st

from src.common import config, load_model, path, write_image
from src.predict import analyze, predict_image

st.set_page_config(page_title="해양 쓰레기 AI",layout="wide")
st.title("해양 쓰레기 식별 AI 연구 대시보드")
cfg=config()
menu=st.sidebar.radio("메뉴",["AI 이미지 분석","동영상 분석","AI 성능 비교","실험 결과","데이터셋 분석"])
variant=st.sidebar.selectbox("모델",["A","B"])
weights=st.sidebar.text_input("학습 가중치 경로",cfg["weights"][variant])
conf=st.sidebar.slider("탐지 신뢰도 임계값",0.01,0.99,float(cfg["evaluation"]["conf"]),0.01)
st.sidebar.caption("낮은 신뢰도나 탐지 0건만으로 객체가 없다고 판단하지 않습니다.")


@st.cache_resource
def cached_model(filename,mtime):
    return load_model(filename)


def show_csv(filename):
    p=path(filename)
    if p.is_file():
        df=pd.read_csv(p)
        st.dataframe(df,hide_index=True)
        st.download_button("CSV 다운로드",p.read_bytes(),file_name=p.name,mime="text/csv",key=str(p))
        return df
    st.info(f"아직 결과가 없습니다: {p}")
    return None


def gallery(folder):
    folder=path(folder)
    if folder.is_dir():
        for p in sorted(folder.glob("*.png")):
            st.image(str(p),caption=p.stem)


try:
    if menu=="AI 이미지 분석":
        uploaded=st.file_uploader("수중 이미지",type=["jpg","jpeg","png","webp"])
        if uploaded is not None:
            st.image(uploaded,caption="원본")
            if st.button("이미지 탐지"):
                filename=path(weights)
                if not filename.is_file():
                    st.error("학습 가중치가 없습니다. 데이터 준비 → A/B 학습 후 best.pt를 지정하세요.")
                else:
                    image=cv2.imdecode(np.frombuffer(uploaded.getvalue(),np.uint8),cv2.IMREAD_COLOR)
                    if image is None:
                        raise ValueError("이미지 읽기 실패")
                    out=path(f"results/dashboard/{time.time_ns()}")
                    out.mkdir(parents=True)
                    source=out/"input.png"
                    write_image(source,image)
                    model=cached_model(str(filename),filename.stat().st_mtime_ns)
                    with st.spinner("탐지 중"):
                        shown,boxes,speed=predict_image(model,source,cfg,conf)
                    write_image(out/"prediction.jpg",shown)
                    st.image(shown,channels="BGR",caption="탐지 결과")
                    names=model.names
                    df=pd.DataFrame([{ "class":names[int(row[5])],"confidence":float(row[4]),"x1":float(row[0]),"y1":float(row[1]),"x2":float(row[2]),"y2":float(row[3])} for row in boxes])
                    df.to_csv(out/"predictions.csv",index=False,encoding="utf-8-sig")
                    st.dataframe(df)
                    st.caption(f"순수 추론: {speed['inference']:.1f} ms. 정답 라벨이 없으므로 정확도는 계산하지 않습니다.")
                    st.download_button("결과 이미지 저장",(out/"prediction.jpg").read_bytes(),file_name="prediction.jpg",mime="image/jpeg")
    elif menu=="동영상 분석":
        uploaded=st.file_uploader("영상",type=["mp4","avi","mov","mkv"])
        st.caption("웹캠은 python -m src.webcam으로 실행합니다. 영상 분석 시간은 길이에 따라 달라집니다.")
        if uploaded is not None and st.button("동영상 탐지"):
            out=path(f"results/dashboard/{time.time_ns()}")
            source=path(f"results/uploads/{time.time_ns()}{Path(uploaded.name).suffix.lower()}")
            source.parent.mkdir(parents=True,exist_ok=True)
            source.write_bytes(uploaded.getvalue())
            with st.spinner("전체 영상 분석 중"):
                video=analyze(str(source),weights,out,cfg,conf)
            st.success(f"결과 저장: {out}")
            st.video(str(video))
            st.caption("브라우저가 mp4v를 재생하지 못하면 다운로드 후 VLC/영상 플레이어로 확인하세요.")
            st.download_button("탐지 영상 다운로드",video.read_bytes(),file_name="prediction.mp4",mime="video/mp4")
            show_csv(str(out/"frames.csv"))
    elif menu=="AI 성능 비교":
        folder=st.text_input("비교 결과 폴더","results/comparison")
        df=show_csv(str(path(folder)/"comparison.csv"))
        if df is not None:
            st.bar_chart(df.set_index("run")[["precision","recall","map50","map50_95"]])
            meta=path(folder)/"comparison.json"
            if meta.is_file():
                for warning in json.loads(meta.read_text(encoding="utf-8")).get("warnings",[]):
                    st.warning(warning)
            gallery(folder)
    elif menu=="실험 결과":
        folder=st.text_input("실험 결과 폴더","results/experiments")
        df=show_csv(str(path(folder)/"conditions.csv"))
        if df is not None:
            st.caption("정답 라벨이 없는 조건의 정확도 지표는 빈칸입니다. 개수/오분류는 프레임 단위로 집계됩니다.")
            gallery(folder)
    else:
        folder=st.text_input("데이터 분석 폴더",str(path(cfg["data"]).parent/"analysis"))
        counts=show_csv(str(path(folder)/"split_counts.csv"))
        if counts is not None:
            st.metric("전체 이미지 수",int(counts["images"].sum()))
        show_csv(str(path(folder)/"class_counts.csv"))
        gallery(folder)
        st.caption("변환/검증 후 분포가 표시됩니다. 객체 라벨 누락 여부는 원본과 샘플 이미지를 육안 검토하세요.")
except (ValueError,FileNotFoundError,OSError,ImportError) as e:
    st.error(str(e))
