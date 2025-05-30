from flask import Flask, render_template, request, jsonify, session,redirect, url_for
import numpy as np
import pandas as pd
import os
from sklearn.preprocessing import LabelEncoder, StandardScaler
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import joblib
from datetime import datetime
from flask import flash
import requests
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
app = Flask(__name__)
RECAPTCHA_SECRET_KEY = '6Lc5QUkrAAAAAPP6Gf5qCNUotCFyLRTNrdEoiOg1'
RECAPTCHA_SITE_KEY='6Lc5QUkrAAAAAJP_kWYlAo3eF81fuVmoXkp2QFFm'
# === Load models ===
heart_disease_model = joblib.load('heart_disease_model.pkl')
xray_model = load_model('new_xray_model.h5')
ecg_model = load_model('ecg_heartbeat_model.h5')

# === Upload folder ===
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# === Label encoders ===
def simple_encode(value, options, default=0):
    return options.index(value) if value in options else default

# Define default values for unknowns
DEFAULTS = {
    'sex': 'F',
    'chest_pain': 'ATA',
    'resting_ecg': 'Normal',
    'exercise_angina': 'N',
    'st_slope': 'Flat'
}

ENCODINGS = {
    'sex': ['F', 'M'],
    'chest_pain': ['TA', 'ATA', 'NAP', 'ASY'],
    'resting_ecg': ['Normal', 'ST', 'LVH'],
    'exercise_angina': ['N', 'Y'],
    'st_slope': ['Up', 'Flat', 'Down']
}

XRAY_CLASSES = [
    'Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration',
    'Mass', 'Nodule', 'Pneumonia', 'Pneumothorax',
    'Consolidation', 'Edema', 'Emphysema', 'Fibrosis',
    'Pleural_Thickening', 'Hernia'
]

ECG_CLASSES = {
    0: "Normal Heart Rhythm",
    1: "Atrial Fibrillation (AFib)",
    2: "Premature Ventricular Contraction (PVC)",
    3: "Ventricular Tachycardia (VT)",
    4: "Myocardial Infarction (MI)"
}

SENDER_EMAIL = "zainabsheik.2210@gmail.com"
SENDER_PASSWORD = "xcbbpnmodernetag"
RECEIVER_EMAIL = "zainabsheik.2210@gmail.com"

@app.route('/', methods=['GET', 'POST'])
def home():
    message = ''
    if request.method == 'POST':
        name = request.form.get('name')
        user_email = request.form.get('email')
        user_message = request.form.get('message')
        recaptcha_response = request.form.get('g-recaptcha-response')

        # Verify reCAPTCHA
        recaptcha_check = requests.post(
            'https://www.google.com/recaptcha/api/siteverify',
            data={
                'secret': RECAPTCHA_SECRET_KEY,
                'response': recaptcha_response
            }
        )
        result = recaptcha_check.json()

        if result.get('success'):
            try:
                msg = MIMEMultipart()
                msg['From'] = SENDER_EMAIL
                msg['To'] = RECEIVER_EMAIL
                msg['Reply-To'] = user_email
                msg['Subject'] = f"Message from {name}"

                body = f"Name: {name}\nEmail: {user_email}\n\nMessage:\n{user_message}"
                msg.attach(MIMEText(body, 'plain'))

                with smtplib.SMTP_SSL('smtp.gmail.com', 465) as server:
                    server.login(SENDER_EMAIL, SENDER_PASSWORD)
                    server.sendmail(SENDER_EMAIL, RECEIVER_EMAIL, msg.as_string())

                message = "Form submitted successfully! Email sent."
            except Exception as e:
                print("EMAIL ERROR:", e)
                message = f"Failed to send email: {e}"
        else:
            message = "reCAPTCHA verification failed. Please try again."

    return render_template('index.html', message=message, site_key=RECAPTCHA_SITE_KEY)





@app.route('/diagnosis')
def diagnosis():
    return render_template('diagnosis.html')

@app.route('/diagnose', methods=['POST'])
def diagnose():

    data = request.form
    files = request.files
    if request.method == "POST":
        name = request.form.get("name")
        age = request.form.get("age")

        # Store values in session
        session["name"] = name
        session["age"] = age


    def safe_int(value, default=0):
        try:
            return int(value)
        except (ValueError, TypeError):
            return default

    def safe_float(value, default=0.0):
        try:
            return float(value)
        except (ValueError, TypeError):
            return default

    # === HEART DISEASE ===
    try:
        age = safe_int(data.get('age'), 0)
        resting_bp = safe_int(data.get('resting_bp'), 120)
        cholesterol = safe_int(data.get('cholesterol'), 190)
        fasting_bs = safe_int(data.get('fasting_bs'), 0)
        max_hr = safe_int(data.get('max_hr'), 150)
        oldpeak = safe_float(data.get('oldpeak'), 0.0)

        sex = data.get('sex', 'Unknown')
        chest_pain = data.get('chest_pain', 'Unknown')
        resting_ecg = data.get('resting_ecg', 'Unknown')
        exercise_angina = data.get('exercise_angina', 'Unknown')
        st_slope = data.get('st_slope', 'Unknown')

        # Apply default if unknown
        sex = sex if sex != 'Unknown' else DEFAULTS['sex']
        chest_pain = chest_pain if chest_pain != 'Unknown' else DEFAULTS['chest_pain']
        resting_ecg = resting_ecg if resting_ecg != 'Unknown' else DEFAULTS['resting_ecg']
        exercise_angina = exercise_angina if exercise_angina != 'Unknown' else DEFAULTS['exercise_angina']
        st_slope = st_slope if st_slope != 'Unknown' else DEFAULTS['st_slope']

        # Encode categories
        sex_encoded = simple_encode(sex, ENCODINGS['sex'])
        chest_pain_encoded = simple_encode(chest_pain, ENCODINGS['chest_pain'])
        resting_ecg_encoded = simple_encode(resting_ecg, ENCODINGS['resting_ecg'])
        exercise_angina_encoded = simple_encode(exercise_angina, ENCODINGS['exercise_angina'])
        st_slope_encoded = simple_encode(st_slope, ENCODINGS['st_slope'])

        input_tabular = np.array([[age, sex_encoded, chest_pain_encoded, resting_bp, cholesterol, fasting_bs,
                                   resting_ecg_encoded, max_hr, exercise_angina_encoded, oldpeak, st_slope_encoded]])

        heart_prediction = int(heart_disease_model.predict(input_tabular)[0])
        heart_result = "Heart Disease Detected" if heart_prediction == 1 else "No Heart Disease"
    except Exception as e:
        heart_result = f"Heart disease prediction error: {str(e)}"

    # === X-RAY ===
    try:
        xray_file = files.get('xray_upload')
        if xray_file:
            path = os.path.join(UPLOAD_FOLDER, xray_file.filename)
            xray_file.save(path)

            img = image.load_img(path, target_size=(224, 224))
            img_array = image.img_to_array(img) / 255.0
            img_array = np.expand_dims(img_array, axis=0)

            preds = xray_model.predict(img_array)[0]
            top_preds = [(XRAY_CLASSES[i], float(preds[i])) for i in range(len(preds))]

            # Always show all predictions:
            xray_result = [f"{label}: {score:.2f}" for label, score in top_preds]

            # Check if any score is significant (> 0.3):
            significant = any(score > 0.85 for _, score in top_preds)

            # Pass 'significant' flag to your template via context:
            context = {
                'xray': xray_result,
                'significant': significant,
                # add other context variables...
            }

        else:
            context = {
                'xray': ["No X-ray uploaded"],
                'significant': False,
            }

    except Exception as e:
        context = {
            'xray': [f"X-ray prediction error: {str(e)}"],
            'significant': False,
        }



    # === ECG ===
    try:
        ecg_file = files.get('ecg_upload')
        if ecg_file:
            df = pd.read_csv(ecg_file)
            signal = df.values
            reshaped = np.reshape(signal, (-1, 187, 1))

            scaler = StandardScaler()
            scaled = scaler.fit_transform(reshaped.reshape(-1, 1)).reshape(reshaped.shape)

            prediction = ecg_model.predict(scaled)
            predicted_class = np.argmax(prediction, axis=1)[0]
            ecg_result = ECG_CLASSES.get(predicted_class, "Unknown ECG pattern")
        else:
            ecg_result = "No ECG file uploaded"
    except Exception as e:
        ecg_result = f"ECG prediction error: {str(e)}"

    # Store the results in session for access by the report route
    session['heart_result'] = heart_result
    session['xray_result'] = xray_result
    session['ecg_result'] = ecg_result
    return redirect(url_for('report'))

@app.route('/report')
def report():
    # Fetch results from session
    patient_name = session.get("name", "N/A")
    patient_age = session.get("age", "N/A")
    heart_disease = session.get('heart_result', 'No heart disease data available')
    ecg = session.get('ecg_result', 'No ECG data available')
    xray = session.get('xray_result', ['No X-ray data available'])

    # Generate suggestions based on results (simple example)
    suggestions = []

    # Heart disease suggestions
    if isinstance(heart_disease, str) and "Heart Disease Detected" in heart_disease:
        suggestions.append("Consult a cardiologist for a detailed examination.")
        suggestions.append("Adopt a heart-healthy diet and exercise regularly.")
    elif isinstance(heart_disease, str) and "No Heart Disease" in heart_disease:
        suggestions.append("Maintain a healthy lifestyle to prevent heart disease.")
    
    # ECG suggestions
    if ecg and ecg != "No ECG file uploaded" and "Normal" not in ecg:
        suggestions.append(f"Follow up on ECG findings: {ecg}.")
    
    # X-ray suggestions
    if xray and isinstance(xray, list):
        abnormal_findings = [f for f in xray if "No major abnormalities" not in f and "No X-ray uploaded" not in f]
        if abnormal_findings:
            suggestions.append("Review chest X-ray abnormalities with a pulmonologist or radiologist.")
        else:
            suggestions.append("Chest X-ray shows no significant abnormalities.")

    # Current date/year for the report header/footer
    current_date = datetime.now().strftime("%B %d, %Y")
    current_year = datetime.now().year

    return render_template(
        'report.html',
        patient_name=patient_name, 
        patient_age=patient_age,
        heart_disease=heart_disease,
        ecg=ecg,
        xray=xray,
        suggestions=suggestions,
        current_date=current_date,
        current_year=current_year
    )

app.secret_key = 'F2ukCAi9kk'
if __name__ == '__main__':
      # Required for session to work
    app.run(debug=True)
