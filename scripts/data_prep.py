import pandas as pd

def data_prep(data):

    # Rename Columns
    data.columns = [
        "university_admission_year",
        "gender",
        "age",
        "hs_grad_year",
        "program",
        "semester",
        "scholarship",
        "transportation",
        "study_hours",
        "study_seatings",
        "learning_mode",
        "smartphone",
        "computer",
        "social_media_hours",
        "english_proficiency",
        "attendance",
        "probation",
        "suspension",
        "teacher_consultation",
        "skills",
        "skill_hours",
        "interest_area",
        "relationship_status",
        "co_curriculars",
        "living_situation",
        "health_issues",
        "semester_gpa",
        "disabilities",
        "cumulative_gpa",
        "credits",
        "family_income",
    ]
    
    # Fill NA entry in skills column
    data["skills"] = data["skills"].fillna("None")

    # Drop duplicates
    data = data.drop_duplicates()

    return data
