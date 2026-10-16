from flask import Flask, render_template, request
import pickle
import numpy as np
import webbrowser
import threading

app = Flask(__name__)

model = pickle.load(open("rf_model.pkl", "rb"))    # تحميل موديل Random Forest
encoder = pickle.load(open("encoder.pkl", "rb"))   # تحميل الـ LabelEncoder

FEATURES = [
    "sleep_hours",
    "study_hours",
    "screen_time",
    "attendance",
    "anxiety_level",
    "academic_pressure",
    "social_support",
    "physical_activity"
]


def get_advice(level):
    if level == "Low":
        return "Your stress level looks good. Keep your healthy routine and continue balancing study, sleep, and activities."
    if level == "Medium":
        return "Try to improve your sleep, reduce screen time, and organize your study schedule into smaller daily tasks."
    return "Your stress level is high. Take breaks, sleep better, ask for support, and reduce academic pressure when possible."


def smart_recommendations(values):
    sleep, study, screen, attendance, anxiety, pressure, support, activity = values

    tips = []

    if sleep < 6:
        tips.append("Sleep hours are low. Try to sleep 7 to 8 hours daily.")

    if study > 6:
        tips.append("Study hours are very high. Take short breaks between study sessions.")

    if screen > 6:
        tips.append("Screen time is high. Reduce phone and laptop usage before sleeping.")

    if attendance < 70:
        tips.append("Attendance is low. Try to attend more classes to reduce academic stress.")

    if anxiety > 7:
        tips.append("Anxiety level is high. Practice breathing exercises or relaxation techniques.")

    if pressure > 7:
        tips.append("Academic pressure is high. Divide tasks into smaller daily goals.")

    if support < 5:
        tips.append("Social support is low. Try to talk with friends, family, or your academic advisor.")

    if activity < 3:
        tips.append("Physical activity is low. Add walking or light exercise to your weekly routine.")

    if len(tips) == 0:
        tips.append("Your inputs look balanced. Keep maintaining your healthy routine.")

    return tips


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    values = []
    for feature in FEATURES:
        values.append(float(request.form[feature]))

    sleep, study, screen, attendance, anxiety, pressure, support, activity = values

    # نفس الـ Feature Engineering اللي اتعمل في التدريب
    screen_study_ratio = screen / (study + 1)
    pressure_index = anxiety + pressure
    health_score = sleep + activity
    support_attendance_score = support + (attendance / 10)

    data = np.array([[
        sleep,
        study,
        screen,
        attendance,
        anxiety,
        pressure,
        support,
        activity,
        screen_study_ratio,
        pressure_index,
        health_score,
        support_attendance_score
    ]])

    prediction = model.predict(data)[0]
    stress_level = encoder.inverse_transform([prediction])[0]

    probability = model.predict_proba(data)[0]
    confidence = round(float(max(probability)) * 100, 2)

    advice = get_advice(stress_level)
    extra_tips = smart_recommendations(values)

    return render_template(
        "result.html",
        stress_level=stress_level,
        confidence=confidence,
        advice=advice,
        values=values,
        extra_tips=extra_tips
    )


def open_browser():
    webbrowser.open_new("http://127.0.0.1:5000/")


if __name__ == "__main__":
    threading.Timer(1.5, open_browser).start()
    app.run(debug=True)
