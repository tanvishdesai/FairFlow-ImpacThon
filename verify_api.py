
import requests
import json

url = "http://localhost:8000/api/predict"

payload = {
    "features": {
        "Age": 28,
        "Gender": "Female",
        "Education_Level": "Masters",
        "Experience_Years": 5,
        "Skill_Score": 85,
        "Interview_Score": 80,
        "Job_Role_Applied": "Software Engineer",
        "Expected_Salary": 120000,
        "Technical_Test_Score": 85,
        "Aptitude_Test_Score": 85,
        "Communication_Score": 80,
        "Certifications_Count": 2,
        "Previous_Companies": 1,
        "Location": "Urban"
    }
}

try:
    response = requests.post(url, json=payload)
    print(f"Status Code: {response.status_code}")
    print("Response JSON:")
    print(json.dumps(response.json(), indent=2))
except Exception as e:
    print(f"Error: {e}")
