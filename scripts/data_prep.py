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

    # Convert Attendance to numeric
    data['attendance'] = pd.to_numeric(data['attendance'].str.replace('%', ''), errors='coerce')

    # Define valid categories
    valid_skills = ['Programming', 'Web development', 'Networking', 'Cyber security', 
                    'Artificial Intelligence', 'Machine Learning', 'Software Development']

    valid_interests = ['Software', 'Hardware', 'Data Science', 'UI/UX', 
                    'NETWORKING', 'Machine Learning', 'Artificial Intelligence']

    # Apply 'Other' conversion
    data['skills'] = data['skills'].apply(lambda x: x if x in valid_skills else 'Other')
    data['interest_area'] = data['interest_area'].apply(lambda x: x if x in valid_interests else 'Other')

    # Encoding categorical variables
    categorical_vars = ['english_proficiency', 'health_issues', 'program']
    binary_vars = ['scholarship', 'gender', 'transportation', 'disabilities']

    # Added dtype=int to ensure 1/0 instead of True/False
    data = pd.get_dummies(
        data, 
        columns=categorical_vars + binary_vars + ['skills', 'interest_area'], 
        drop_first=True,
        dtype=int)
    
    # Drop unnecessary columns
    data = data.drop(columns=["university_admission_year", "hs_grad_year", "semester", 'probation', 'suspension',
                              'teacher_consultation', 'smartphone', 'computer', 'relationship_status', 
                              'co_curriculars', 'living_situation', 'learning_mode'])
    
    # Clean final columns
    data.columns = data.columns.str.lower().str.replace(' ', '_')

    return data


# Run data prep function and save cleaned dataset
prepared_data = data_prep(pd.read_csv("data/Students_Performance_dataset.csv"))
prepared_data.to_csv("data/prepared_student_data.csv", index=False)
