from flask import Flask, render_template, request, redirect, url_for, flash, session, send_file
from flask_mail import Mail, Message
from werkzeug.utils import secure_filename
import cv2
import numpy as np
import tensorflow as tf
import qrcode
import io
from models import db, User, EmergencyRequest, bcrypt
import os  # For file handling

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your_secret_key'  # Change to a random string, e.g., 'my_secret_key_123'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///health.db'
app.config['MAIL_SERVER'] = 'smtp.gmail.com'  # For email notifications (optional)
app.config['MAIL_USERNAME'] = 'your_email@gmail.com'  # Replace with your Gmail
app.config['MAIL_PASSWORD'] = 'your_password'  # Replace with your password or app password
app.config['UPLOAD_FOLDER'] = 'uploads/'

db.init_app(app)
mail = Mail(app)

# Load AI Model
model = tf.keras.models.load_model('blood_group_model.h5')
le = np.load('label_encoder.npy', allow_pickle=True)

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        email = request.form['email']
        existing_user = User.query.filter_by(email=email).first()
        if existing_user:
            flash('Email already registered. Please use a different email or log in.')
            return redirect(url_for('register'))
        password = bcrypt.generate_password_hash(request.form['password']).decode('utf-8')
        user = User(email=email, password=password, name=request.form.get('name'), age=request.form.get('age'), contact=request.form.get('contact'), medical_conditions=request.form.get('medical_conditions'), allergies=request.form.get('allergies'))
        db.session.add(user)
        db.session.commit()
        flash('Registered successfully!')
        return redirect(url_for('home'))
    return render_template('register.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        user = User.query.filter_by(email=request.form['email']).first()
        if user and bcrypt.check_password_hash(user.password, request.form['password']):
            session['user_id'] = user.id
            return redirect(url_for('dashboard'))
        flash('Invalid credentials')
    return render_template('login.html')

@app.route('/upload', methods=['GET', 'POST'])
def upload():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    if request.method == 'POST':
        file = request.files['fingerprint']
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Preprocess and predict
        img = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (128, 128)) / 255.0
        img = np.expand_dims(img, axis=[0, -1])
        prediction = model.predict(img)
        predicted_class = np.argmax(prediction)
        blood_group = le[predicted_class]
        confidence = np.max(prediction) * 100
        
        # Removed auto-save: No longer updating user.blood_group here
        # Just return the prediction to the result page
        
        return render_template('result.html', blood_group=blood_group, confidence=confidence)
    return render_template('upload.html')

@app.route('/dashboard', methods=['GET', 'POST'])
def dashboard():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    user = User.query.get(session['user_id'])
    if request.method == 'POST':
        # Update user profile from form
        user.name = request.form.get('name')
        user.age = request.form.get('age')
        user.contact = request.form.get('contact')
        user.medical_conditions = request.form.get('medical_conditions')
        user.allergies = request.form.get('allergies')
        user.blood_group = request.form.get('blood_group')
        db.session.commit()
        flash('Profile updated successfully!')
        return redirect(url_for('dashboard'))  # Redirect to reload with data
    return render_template('dashboard.html', user=user)



@app.route('/qr_code')
def qr_code():
    user = User.query.get(session['user_id'])
    qr_data = f"Name: {user.name}\nBlood Group: {user.blood_group}\nContact: {user.contact}"
    qr = qrcode.QRCode()
    qr.add_data(qr_data)
    img = qr.make_image()
    buf = io.BytesIO()
    img.save(buf, format='PNG')
    buf.seek(0)
    return send_file(buf, mimetype='image/png', as_attachment=True, download_name='health_card.png')

@app.route('/emergency', methods=['GET', 'POST'])
def emergency():
    if request.method == 'POST':
        req = EmergencyRequest(user_id=session['user_id'], blood_group_needed=request.form['blood_group'],
                               location=request.form['location'], description=request.form['description'])
        db.session.add(req)
        db.session.commit()
        # Email notification (optional)
        msg = Message('Emergency Blood Request', sender='your_email@gmail.com', recipients=['admin@example.com'])
        msg.body = f"New request: {req.description}"
        mail.send(msg)
        flash('Request posted!')
    requests = EmergencyRequest.query.all()
    return render_template('emergency.html', requests=requests)



if __name__ == '__main__':
    with app.app_context():
        db.create_all()
    app.run(debug=True)