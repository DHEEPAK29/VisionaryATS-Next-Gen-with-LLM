import torch
from transformers import LlamaForCausalLM, LlamaTokenizer, pipeline
from sklearn.cluster import KMeans
import pandas as pd
import numpy as np

# Load LLaMA Model (e.g., Llama-2-7b)
model_name = "meta/llama-2-7b"  # Replace with the actual model path or Hugging Face name
tokenizer = LlamaTokenizer.from_pretrained(model_name)
model = LlamaForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16)

bucket = []

# categories for classification
categories = ["Cardiology", "Neurology", "Oncology", "Orthopedics", "Pediatrics", "General"]

# Sample patient reports for analysis
patient_reports = [
    "Patient has been experiencing chest pain and shortness of breath. ECG test indicates irregular heart rhythm.",
    "A child has a fever and persistent cough. No other symptoms observed. Chest X-ray showed mild lung congestion.",
    "The patient complains of severe headaches and blurred vision. MRI scan indicates a possible brain tumor.",
    "The patient has joint pain, difficulty moving, and swelling in the knees. X-rays suggest possible arthritis.",
    "Patient has a high fever, body aches, and cough. No history of chronic disease. Suspected viral infection."
]

#Database connection : MongodB for Doctor Info fetch 
doctors = [
    {"name": "Dr. Smith", "specialty": "Cardiology"},
    {"name": "Dr. Johnson", "specialty": "Neurology"},
    {"name": "Dr. White", "specialty": "Oncology"},
    {"name": "Dr. Brown", "specialty": "Orthopedics"},
    {"name": "Dr. Lee", "specialty": "Pediatrics"},
    {"name": "Dr. Taylor", "specialty": "General"}
]

def classify_report_with_llama(report_text):
    inputs = tokenizer(report_text, return_tensors="pt")
    # Generate logits for classification
    with torch.no_grad():
        outputs = model(**inputs)
    # Use a simple heuristic to determine the category based on the highest probability
    logits = outputs.logits
    prediction_idx = torch.argmax(logits, dim=-1).item()
    return categories[prediction_idx]

# Function to prioritize report based on urgency
def prioritize_report(report_text):
    high_priority_keywords = ["severe", "urgent", "critical", "acute", "emergency"]
    priority_score = sum(keyword in report_text.lower() for keyword in high_priority_keywords)
    return priority_score

# Function to map doctors to patient reports based on category
def map_doctor_to_report(category):
    for doctor in doctors:
        if doctor["specialty"].lower() == category.lower():
            return doctor["name"]
    return "No suitable doctor found"

# Analyze patient reports
def analyze_patient_reports(reports):
    analysis_results = []
    
    for report in reports:
        category = classify_report_with_llama(report)
        priority = prioritize_report(report)
        assigned_doctor = map_doctor_to_report(category)
        
        analysis_results.append({
            "report": report,
            "category": category,
            "priority": priority,
            "assigned_doctor": assigned_doctor
        })
    
    return analysis_results

# Process reports and print results
results = analyze_patient_reports(patient_reports)
for result in results:
    print(f"Report: {result['report']}")
    print(f"Category: {result['category']}")
    print(f"Priority: {result['priority']}")
    print(f"Assigned Doctor: {result['assigned_doctor']}")
    print("-" * 60)

