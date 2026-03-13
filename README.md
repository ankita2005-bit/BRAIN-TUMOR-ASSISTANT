# BRAIN-TUMOR-ASSISTANT
MedScan AI

AI-Powered Diagnostic Assistant for Brain Tumor Detection and Reporting

Overview

MedScan AI is an advanced diagnostic support system designed for radiologists and healthcare professionals.
It leverages state-of-the-art computer vision and medical language modeling to assist in the detection, classification, and interpretation of brain tumors from MRI scans.

The system automates tumor localization and generates a structured, clinically relevant report using RaDialog LLM, while an integrated LLaVA-based assistant enables case-specific discussions and differential insights — enhancing the radiologist’s decision-making workflow.

Key Features

Automated Tumor Detection: Identifies tumor regions in MRI scans using a Roboflow-trained CV model integrated with OpenCV preprocessing.

Tumor Type Classification: Differentiates among key tumor categories such as Glioma, Meningioma, and Pituitary.

AI-Generated Radiology Report: Employs RaDialog LLM to produce structured summaries including findings, impressions, and clinical notes aligned with standard radiology report formatting.

Interactive Case Discussion: LLaVA + RaDialog LLM enable natural-language interaction for query-based interpretation, clinical reasoning, and report explanation.

Streamlined Interface: Built on Streamlit for intuitive usage within diagnostic workflows.

Tech Stack
Frontend-Streamlit
Backend-FastAPI
AI/ML Models-Roboflow, RaDialog LLM, LLaVA
Libraries-OpenCV, NumPy
Language-Python
