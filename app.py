
import streamlit as st
import requests
import cv2
import numpy as np


ROBOFLOW_API_KEY = "59iRIetC6t9YrCGT2OOz"
MODEL_ID = "brain-tumor-noh6o/2"
ROBOFLOW_URL = f"https://detect.roboflow.com/{MODEL_ID}?api_key={ROBOFLOW_API_KEY}"

API_BASE = "http://localhost:8000"  


st.set_page_config(page_title="Brain Tumor Assistant", layout="wide")
st.title("Brain Tumor Assistant")


for key in ["conv_id", "report", "findings", "annotated", "raw_image", "qa_history"]:
    if key not in st.session_state:
        st.session_state[key] = None
if st.session_state.qa_history is None:
    st.session_state.qa_history = []


def draw_predictions(image_array, predictions):
    img = image_array.copy()
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    for pred in predictions.get("predictions", []):
        x, y, w, h = int(pred["x"]), int(pred["y"]), int(pred["width"]), int(pred["height"])
        conf = round(pred["confidence"] * 100, 1)
        cls = pred["class"]
        x1, y1 = x - w // 2, y - h // 2
        x2, y2 = x + w // 2, y + h // 2
        cv2.rectangle(img, (x1, y1), (x2, y2), (255, 255, 0), 2)
        cv2.putText(img, f"{cls} ({conf}%)", (x1, y1 - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
    return img

def extract_findings(pred):
    return pred.get("class", "brain tumor").title()


col1, col2 = st.columns([1, 1])

with col1:
    uploaded_file = st.file_uploader("📤 Upload brain MRI", type=["jpg","jpeg","png"])
    if uploaded_file:
        file_bytes = uploaded_file.read()
        arr = np.frombuffer(file_bytes, np.uint8)
        img_cv = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img_cv is None:
            st.error("Could not decode image.")
            st.stop()
        st.session_state.raw_image = img_cv

       
        with st.spinner("🔍 Detecting tumor (Roboflow)..."):
            try:
                resp = requests.post(ROBOFLOW_URL, files={"file": file_bytes}, timeout=60)
                result = resp.json()
            except Exception as e:
                st.error(f"Roboflow error: {e}")
                st.stop()

        if "predictions" in result and result["predictions"]:
            annotated = draw_predictions(img_cv, result)
            st.session_state.annotated = annotated
            st.session_state.findings = extract_findings(result["predictions"][0])
            st.image(annotated, caption="✅ Tumor Detected (Annotated)", use_column_width=True)
        else:
            st.session_state.annotated = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
            st.session_state.findings = "No obvious tumor"
            st.warning("⚠️ No tumor detected.")
            st.image(st.session_state.annotated, caption="MRI (No tumor detected)", use_column_width=True)

with col2:
    st.subheader("📋 Radiology Report (Interactive)")


    if st.button("📝 Generate Initial Report"):
        if st.session_state.annotated is None:
            st.error("No annotated image yet.")
        else:
            _, buf = cv2.imencode(".jpg", st.session_state.annotated)
            ann_bytes = buf.tobytes()
            try:
                r = requests.post(
                    f"{API_BASE}/generate_report",
                    files={"image": ("annotated.jpg", ann_bytes, "image/jpeg")},
                    data={"findings": st.session_state.findings},
                    timeout=120
                )
                r.raise_for_status()
                data = r.json()
                st.session_state.conv_id = data.get("conversation_id")
                st.session_state.report = data.get("report", "")
                st.success("Report generated. You can refine or ask questions below.")
            except Exception as e:
                st.error(f"API error: {e}")

    if st.session_state.report:
        st.text_area("Current Report", value=st.session_state.report, height=300, key="report_box")

    
    refine_input = st.text_input("✏️ Refine the report:", key="refine_input")
    if st.button("Send Refinement"):
        if refine_input and st.session_state.conv_id:
            try:
                r = requests.post(
                    f"{API_BASE}/chat",
                    data={"conversation_id": st.session_state.conv_id, "message": refine_input},
                    timeout=120
                )
                r.raise_for_status()
                st.session_state.report = r.json().get("reply", "")
            except Exception as e:
                st.error(f"API error: {e}")

    
    qa_input = st.text_input("💬 Ask a question about this report:", key="qa_input")
    if st.button("Ask Question"):
        if qa_input and st.session_state.conv_id and st.session_state.annotated is not None:
            try:
                r = requests.post(
                    f"{API_BASE}/report_qa",
                    data={
                        "conversation_id": st.session_state.conv_id,
                        "question": qa_input
                    },
                    timeout=120
                )
                r.raise_for_status()
                answer = r.json().get("answer", "")
                st.session_state.qa_history.append({"question": qa_input, "answer": answer})
            except Exception as e:
                st.error(f"API error: {e}")

    
    if st.session_state.qa_history:
        st.subheader("💡 Q&A about Report")
        for i, qa in enumerate(st.session_state.qa_history, 1):
            st.markdown(f"**Q{i}:** {qa['question']}")
            st.markdown(f"**A{i}:** {qa['answer']}")
            st.markdown("---")
