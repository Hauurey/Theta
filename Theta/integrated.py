import os
import threading
import queue
import cv2
import numpy as np
from datetime import datetime, timedelta
from flask import Flask, render_template, Response, request, redirect, url_for, session, flash, jsonify
from email.message import EmailMessage
from PIL import Image
import smtplib
from flask_socketio import SocketIO
import sys

app = Flask(__name__)
socketio = SocketIO(app)
app.secret_key = 'your_secret_key'  # Set your secret key for session management

# Email configuration
Sender_Email = "202110035@fit.edu.ph"
Reciever_Email = "anonuevo.harry.s@gmail.com"
Password = "vdca prti xerc wsba"  # Use app-specific password if using Gmail

image_queue = queue.Queue()
new_image_event = threading.Event()

last_print_times = {"motion": datetime.min, "face": datetime.min, "unknown_face": datetime.min}
print_interval = timedelta(seconds=10)
log_dir = 'logs'
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

log_file_path = None
log_file = None

def restart_flask_app():
    python = sys.executable
    os.execl(python, python, *sys.argv)

def get_log_file_path():
    current_date = datetime.now().strftime('%m-%d-%Y')
    return os.path.join(log_dir, f'{current_date}.txt')

def log_message(message):
    global log_file, log_file_path
    current_date = datetime.now().strftime('%m-%d-%Y')
    new_log_file_path = get_log_file_path()
    if new_log_file_path != log_file_path:
        if log_file:
            log_file.close()
        log_file_path = new_log_file_path
        log_file = open(log_file_path, 'a')
    current_time = datetime.now().strftime('%H:%M')
    log_entry = f"[{current_time}]: {message}\n"
    log_file.write(log_entry)
    log_file.flush()
    socketio.emit('log_update', log_entry)

def send_email_with_image(image_path):
    newMessage = EmailMessage()
    newMessage['Subject'] = "Unknown Faces Image"
    newMessage['From'] = Sender_Email
    newMessage['To'] = Reciever_Email
    newMessage.set_content('Let me know what you think. Image attached!')

    with open(image_path, 'rb') as f:
        image_data = f.read()
        image = Image.open(f)
        image_type = image.format.lower()
        image_name = os.path.basename(image_path)

    newMessage.add_attachment(image_data, maintype='image', subtype=image_type, filename=image_name)

    with smtplib.SMTP_SSL('smtp.gmail.com', 465) as smtp:
        smtp.login(Sender_Email, Password)
        smtp.send_message(newMessage)

def handle_image_tasks():
    image_counter = 0  # Initialize the image counter
    while True:
        face_img, date_folder, current_time = image_queue.get()
        if face_img is None:
            break
        image_counter += 1  # Increment the counter
        if image_counter % 10 == 0:  # Only process every 5th image
            try:
                # Save the unknown face
                face_filename = os.path.join(date_folder, f"unknown_face_{current_time}.jpg")
                cv2.imwrite(face_filename, face_img)
                # Wait for any new image event to be set
                new_image_event.wait(2)
                new_image_event.clear()
                # Check if the queue is empty, meaning this is the latest image
                if image_queue.empty():
                    # Send email with the image
                    send_email_with_image(face_filename)
            except Exception as e:
                print(f"Failed to process image task: {e}")
            finally:
                image_queue.task_done()

unknown_faces_dir = 'unknown_faces'
if not os.path.exists(unknown_faces_dir):
    os.makedirs(unknown_faces_dir)

data_dir = 'data'
if not os.path.exists(data_dir):
    os.makedirs(data_dir)

id_name_dict = {}
id_name_file_path = os.path.join(data_dir, "id_name.txt")
if os.path.exists(id_name_file_path):
    with open(id_name_file_path, "r") as f:
        for line in f:
            id, name = line.strip().split(',')
            id_name_dict[int(id)] = name
else:
    log_message("id_name.txt not found, starting with an empty id-name dictionary")

video = cv2.VideoCapture(0)
backSub = cv2.createBackgroundSubtractorMOG2()
facedetect = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
recognizer = cv2.face.LBPHFaceRecognizer_create()

trainer_file_path = r'C:\Users\user\Desktop\Theta\Trainer.yml'
if os.path.exists(trainer_file_path):
    recognizer.read(trainer_file_path)
else:
    log_message("Trainer.yml not found, starting without pre-trained recognizer")

worker_thread = threading.Thread(target=handle_image_tasks)
worker_thread.start()

collecting_faces = False
collect_id = None
collect_name = None
collect_count = 0
max_collect_count = 100

def generate_frames():
    global video, collecting_faces, collect_id, collect_name, collect_count
    while True:
        ret, frame = video.read()
        if not ret:
            break

        fgMask = backSub.apply(frame)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = facedetect.detectMultiScale(gray, 1.3, 5)
        nonZeroCount = cv2.countNonZero(fgMask)

        if nonZeroCount > 5000:
            current_time = datetime.now()
            if current_time - last_print_times["motion"] > print_interval:
                log_message("Motion Detected")
                last_print_times["motion"] = current_time

        for (x, y, w, h) in faces:
            if collecting_faces and collect_count < max_collect_count:
                face_img = gray[y:y+h, x:x+w]
                dataset_dir = r'C:\Users\user\Desktop\Theta\datasets'
                if not os.path.exists(dataset_dir):
                    os.makedirs(dataset_dir)
                collect_count += 1
                cv2.imwrite(os.path.join(dataset_dir, f'User.{collect_id}.{collect_count}.jpg'), face_img)
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                if collect_count >= max_collect_count:
                    with open(id_name_file_path, "a") as file:
                        file.write(f"{collect_id},{collect_name}\n")
                    log_message(f"Finished collecting faces for ID: {collect_id}, Name: {collect_name}")
                    collecting_faces = False
                    collect_id = None
                    collect_name = None
                    collect_count = 0

            else:
                if os.path.exists(trainer_file_path):
                    serial, conf = recognizer.predict(gray[y:y+h, x:x+w])
                    text = f"Unknown - Confidence: {round(100 - conf, 2)}%"
                    if conf <= 80:
                        name = id_name_dict.get(serial, "Unknown")
                        text = f"{name} - Confidence: {round(100 - conf, 2)}%"
                        current_time = datetime.now()
                        if current_time - last_print_times["face"] > print_interval:
                            log_message(f"Face Detected: {name}")
                            last_print_times["face"] = current_time
                    else:
                        current_date = datetime.now().strftime('%Y-%m-%d')
                        date_folder = os.path.join(unknown_faces_dir, current_date)
                        if not os.path.exists(date_folder):
                            os.makedirs(date_folder)

                        face_img = frame[y:y+h, x:x+w]
                        image_queue.put((face_img, date_folder, current_time.strftime('%m-%d-%Y_%H-%M')))
                        new_image_event.set()
                        current_time = datetime.now()
                        if current_time - last_print_times["unknown_face"] > print_interval:
                            log_message("Face Detected: Unknown")
                            last_print_times["unknown_face"] = current_time

                    cv2.putText(frame, text, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (50, 50, 255), 1)
                else:
                    cv2.putText(frame, "Recognizer model not trained yet", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)

        frame = cv2.resize(frame, (640, 480))
        fgMask = cv2.resize(fgMask, (640, 480))

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    if 'logged_in' in session:
        registered_faces_count = len(id_name_dict)
        registered_faces = id_name_dict.items()
        return render_template('dashboard.html', registered_faces_count=registered_faces_count, registered_faces=registered_faces)
    else:
        return redirect(url_for('login'))

    # Function to check if the provided email and password match any pair in the text file
def check_credentials(email, password):
    credentials_file_path = 'credentials.txt'  # Path to your credentials text file
    if os.path.exists(credentials_file_path):
        with open(credentials_file_path, 'r') as file:
            for line in file:
                stored_email, stored_password = line.strip().split(',')
                if email == stored_email and password == stored_password:
                    return True
    return False

@app.route('/login', methods=['GET', 'POST'])
def login():
    if 'logged_in' in session:
        return redirect(url_for('dashboard'))
    
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']
        
        # Check if the provided email and password match any pair in the text file
        if check_credentials(email, password):
            session['logged_in'] = True
            flash('Login successful!', 'success')
            return redirect(url_for('index'))
        else:
            flash('Invalid email or password. Please try again.', 'error')

    return render_template('login.html')

@app.route('/logout')
def logout():
    session.pop('logged_in', None)
    flash('Logged out successfully!', 'success')
    return redirect(url_for('login'))

@app.route('/collect_faces', methods=['POST'])
def collect_faces():
    global collecting_faces, collect_id, collect_name
    collect_id = int(request.form['id'])
    collect_name = request.form['name']
    collecting_faces = True
    return redirect(url_for('index'))

@app.route('/train_faces', methods=['POST'])
def train_faces():
    data_dir = r'C:\Users\user\Desktop\Theta\datasets'
    face_samples = []
    ids = []

    # Collect all face samples and their corresponding IDs
    for root, dirs, files in os.walk(data_dir):
        for file in files:
            if file.endswith('.jpg'):
                image_path = os.path.join(root, file)
                img = Image.open(image_path).convert('L')
                img_numpy = np.array(img, 'uint8')
                id = int(os.path.split(image_path)[-1].split(".")[1])
                face_samples.append(img_numpy)
                ids.append(id)

    ids = np.array(ids)
    
    # Train the recognizer with the collected face samples and IDs
    recognizer.train(face_samples, ids)
    
    # Overwrite the existing Trainer.yml file with the new trained model
    recognizer.write(trainer_file_path)
    log_message("Training completed successfully")
    
    # Reload the trained model
    recognizer.read(trainer_file_path)
    restart_flask_app()
    return redirect(url_for('index'))

@app.route('/delete_user', methods=['POST'])
def delete_user():
    user_id = int(request.form['id'])
    user_name = id_name_dict.pop(user_id, None)
    
    if user_name:
        # Remove user from id_name.txt
        with open(id_name_file_path, "r") as f:
            lines = f.readlines()
        with open(id_name_file_path, "w") as f:
            for line in lines:
                if line.strip().split(',')[0] != str(user_id):
                    f.write(line)
        
        # Remove user's dataset directory
        dataset_dir = r'C:\Users\user\Desktop\Theta\datasets'
        for root, dirs, files in os.walk(dataset_dir):
            for file in files:
                if file.startswith(f'User.{user_id}.'):
                    os.remove(os.path.join(root, file))
        
        log_message(f"Deleted user ID: {user_id}, Name: {user_name}")
    else:
        log_message(f"Attempted to delete non-existing user ID: {user_id}")

    return redirect(url_for('index'))

@app.route('/get_logs')
def get_logs():
    logs_dir = r'C:\Users\user\Desktop\Theta\logs'  # Update the directory path
    if not os.path.exists(logs_dir):
        os.makedirs(logs_dir)
    log_files = [file for file in os.listdir(logs_dir) if file.endswith('.txt')]
    return jsonify(log_files)

@app.route('/get_logs/<filename>')
def get_logs_content(filename):
    logs_dir = 'logs'  # Update the directory path
    file_path = os.path.join(logs_dir, filename)
    
    try:
        with open(file_path, 'r') as file:
            content = file.read()
        return content
    except FileNotFoundError:
        return "File not found"

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/dashboard')
def dashboard():
    if 'logged_in' in session:
        registered_faces_count = len(id_name_dict)
        registered_faces = id_name_dict.items()
        return render_template('dashboard.html', registered_faces_count=registered_faces_count, registered_faces=registered_faces)
    else:
        return redirect(url_for('login'))
    
    
@app.route('/change_password', methods=['POST'])
def change_password():
    # Extract email, old password, and new password from the request
    email = request.json.get('email')
    old_password = request.json.get('oldPassword')
    new_password = request.json.get('newPassword')

    # Read the credentials from the file
    credentials_file_path = 'credentials.txt'  # Path to your credentials text file
    credentials = {}

    with open(credentials_file_path, 'r') as file:
        for line in file:
            stored_email, stored_password = line.strip().split(',')
            credentials[stored_email] = stored_password

    # Check if the provided email exists in the credentials dictionary
    if email in credentials and credentials[email] == old_password:
        # Update the password in the credentials dictionary
        credentials[email] = new_password

        # Write the updated credentials back to the file
        with open(credentials_file_path, 'w') as file:
            for email, password in credentials.items():
                file.write(f"{email},{password}\n")

        return jsonify({'success': True, 'message': 'Password changed successfully.'}), 200
    else:
        # Return an error response if the provided email or old password is incorrect
        return jsonify({'success': False, 'message': 'Incorrect email or old password.'}), 400

if __name__ == '__main__':
    try:
        socketio.run(app, debug=True, allow_unsafe_werkzeug=True)
    finally:
        if log_file is not None:
            log_file.close()
        image_queue.put((None, None, None))
        worker_thread.join()
        video.release()
        cv2.destroyAllWindows()