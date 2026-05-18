from flask import Flask, render_template, request, redirect, url_for
import sqlite3
import pandas as pd
import joblib

app = Flask(__name__)

clf_model = joblib.load("model.pkl")
reg_model = joblib.load("reg_model.pkl")
feature_columns = joblib.load("feature_columns.pkl")


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/login', methods=['GET', 'POST'])
def login():

    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']

        conn = sqlite3.connect("users.db")
        cursor = conn.cursor()

        cursor.execute(
            "SELECT * FROM users WHERE email=? AND password=?",
            (email, password)
        )

        user = cursor.fetchone()
        conn.close()

        if user:
            return redirect(url_for('predict_page'))
        else:
            return render_template(
                'login.html',
                error="Invalid email or password"
            )

    return render_template('login.html')


@app.route('/signup', methods=['GET', 'POST'])
def signup():

    if request.method == 'POST':
        name = request.form['name']
        email = request.form['email']
        password = request.form['password']

        conn = sqlite3.connect("users.db")
        cursor = conn.cursor()

        try:
            cursor.execute(
                "INSERT INTO users (name, email, password) VALUES (?, ?, ?)",
                (name, email, password)
            )

            conn.commit()
            conn.close()

            return redirect(url_for('login'))

        except sqlite3.IntegrityError:
            conn.close()

            return render_template(
                'signup.html',
                error="Email already exists"
            )

    return render_template('signup.html')


@app.route('/predict-page')
def predict_page():
    return render_template('predict.html')


@app.route('/predict', methods=['POST'])
def predict():
    tenth = float(request.form['tenth'])
    twelfth = float(request.form['twelfth'])
    cgpa = float(request.form['cgpa'])
    branch = request.form['branch']
    backlogs = int(request.form['backlogs'])

    coding = float(request.form['coding'])
    communication = float(request.form['communication'])
    aptitude = float(request.form['aptitude'])

    projects = int(request.form['projects'])
    internships = int(request.form['internships'])
    hackathons = int(request.form['hackathons'])
    certifications = int(request.form['certifications'])

    branch_map = {
        "CSE": 0,
        "IT": 1,
        "ECE": 2,
        "EEE": 3,
        "MECH": 4
    }

    input_data = pd.DataFrame({
        "tenth_percentage": [tenth],
        "twelfth_percentage": [twelfth],
        "cgpa": [cgpa],
        "branch": [branch_map.get(branch, 0)],
        "backlogs": [backlogs],
        "coding_skill_rating": [coding],
        "communication_skill_rating": [communication],
        "aptitude_skill_rating": [aptitude],
        "projects_completed": [projects],
        "internships_completed": [internships],
        "hackathons_participated": [hackathons],
        "certifications_count": [certifications],
        "gender": [1],
        "city_tier": [2],
        "study_hours_per_day": [4],
        "sleep_hours": [7],
        "stress_level": [5],
        "internet_access": [1],
        "attendance_percentage": [80],
        "extracurricular_involvement": [1],
        "part_time_job": [0],
        "family_income_level": [1]
    })

    for col in feature_columns:
        if col not in input_data.columns:
            input_data[col] = 0

    input_data = input_data[feature_columns]

    placement = clf_model.predict(input_data)[0]

    salary = 0
    weak_sectors = []
    reasons = []
    suggestions = []

    if cgpa < 7:
        weak_sectors.append("Academics")
        reasons.append("Low CGPA")
        suggestions.append("Improve academic performance and technical basics")

    if coding < 60:
        weak_sectors.append("Coding")
        reasons.append("Weak coding skills")
        suggestions.append("Practice DSA, Python, and problem solving daily")

    if communication < 60:
        weak_sectors.append("Communication")
        reasons.append("Weak communication skills")
        suggestions.append("Practice HR questions, group discussions, and mock interviews")

    if aptitude < 60:
        weak_sectors.append("Aptitude")
        reasons.append("Low aptitude score")
        suggestions.append("Practice quantitative aptitude, reasoning, and verbal ability")

    if internships == 0:
        weak_sectors.append("Industry Exposure")
        reasons.append("No internship experience")
        suggestions.append("Complete at least one internship")

    if projects < 2:
        weak_sectors.append("Projects")
        reasons.append("Low project experience")
        suggestions.append("Build real-world projects in AI, ML, web development, or cloud")

    if certifications == 0:
        weak_sectors.append("Certifications")
        reasons.append("No certifications")
        suggestions.append("Complete certifications in Python, AI/ML, cloud, or data science")

    if backlogs > 0:
        weak_sectors.append("Backlogs")
        reasons.append("Active backlogs")
        suggestions.append("Clear backlogs as early as possible")

    if hackathons == 0:
        weak_sectors.append("Hackathons")
        reasons.append("No hackathon participation")
        suggestions.append("Participate in hackathons to improve practical skills")

    if placement == 1:
        status = "PLACED"
        salary = round(reg_model.predict(input_data)[0], 2)

        if not weak_sectors:
            weak_sectors = ["Strong Profile"]
            reasons = ["Good academic, technical, and skill profile"]
            suggestions = ["Continue improving coding and interview preparation"]
    else:
        status = "NOT PLACED"
        salary = 0

    if cgpa >= 8 and coding >= 80:
        companies = ["Google", "Microsoft", "Amazon", "Adobe"]
    elif cgpa >= 7 and coding >= 60:
        companies = ["TCS", "Infosys", "Accenture", "Wipro"]
    else:
        companies = ["Startups", "Internship Programs", "Training-based Companies"]

    return render_template(
        "result.html",
        status=status,
        salary=salary,
        weak_sectors=weak_sectors,
        reasons=reasons,
        suggestions=suggestions,
        companies=companies
    )


if __name__ == '__main__':
    app.run(debug=True)