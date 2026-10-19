import tkinter as tk
from tkinter import ttk, messagebox
import cv2
import os
import sqlite3
import numpy as np
from PIL import Image, ImageTk, ImageDraw
import datetime
import time
import threading
import random
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import queue
import winsound
from collections import defaultdict

# Configuration  
BASE_DIR = os.path.dirname(os.path.abspath(_file_))
TRAINING_IMAGE_DIR = os.path.join(BASE_DIR, "TrainingImage")
STUDENT_DB_PATH = os.path.join(BASE_DIR, "StudentDetails", "student_database.db")
TRAINNER_PATH = os.path.join(TRAINING_IMAGE_DIR, "Trainner.yml")
HAARCASCADE_PATH = os.path.join(BASE_DIR, "haarcascade_frontalface_default.xml")
ATTENDANCE_DIR = os.path.join(BASE_DIR, "Attendance")
UNKNOWN_IMAGES_DIR = os.path.join(BASE_DIR, "ImagesUnknown")
SOUNDS_DIR = os.path.join(BASE_DIR, "Sounds")
REPORTS_DIR = os.path.join(BASE_DIR, "Reports")

# Create directories if they don't exist
os.makedirs(TRAINING_IMAGE_DIR, exist_ok=True)
os.makedirs(ATTENDANCE_DIR, exist_ok=True)
os.makedirs(UNKNOWN_IMAGES_DIR, exist_ok=True)
os.makedirs(SOUNDS_DIR, exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)
os.makedirs(os.path.dirname(STUDENT_DB_PATH), exist_ok=True)

class ThreadSafeDatabase:
    def _init_(self):
        self.db_queue = queue.Queue()
        self.result_queue = queue.Queue()
        self.worker_thread = threading.Thread(target=self._db_worker, daemon=True)
        self.worker_thread.start()
    
    def _db_worker(self):
        """Worker thread that handles all database operations"""
        conn = sqlite3.connect(STUDENT_DB_PATH)
        conn.execute("PRAGMA journal_mode=WAL")  # Enable WAL mode for better concurrency
        cursor = conn.cursor()
        
        # Create tables if they don't exist
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS students (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                registration_date TEXT,
                last_attendance TEXT
            )
        ''')
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS attendance (
                id TEXT,
                name TEXT,
                date TEXT,
                time TEXT,
                FOREIGN KEY(id) REFERENCES students(id)
            )
        ''')
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS admin (
                username TEXT PRIMARY KEY,
                password TEXT NOT NULL
            )
        ''')
        
        # Insert default admin account if none exists
        cursor.execute('SELECT 1 FROM admin')
        if not cursor.fetchone():
            cursor.execute('''
                INSERT INTO admin (username, password)
                VALUES (?, ?)
            ''', ('admin', 'admin123'))  # Default credentials
        
        conn.commit()
        
        while True:
            task = self.db_queue.get()
            if task == 'STOP':
                conn.close()
                break
                
            method, args, kwargs = task
            try:
                if method == 'add_student':
                    result = self._add_student(cursor, *args, **kwargs)
                elif method == 'mark_attendance':
                    result = self._mark_attendance(cursor, *args, **kwargs)
                elif method == 'get_student_name':
                    result = self._get_student_name(cursor, *args, **kwargs)
                elif method == 'get_attendance_records':
                    result = self._get_attendance_records(cursor, *args, **kwargs)
                elif method == 'get_all_students':
                    result = self._get_all_students(cursor, *args, **kwargs)
                elif method == 'verify_admin':
                    result = self._verify_admin(cursor, *args, **kwargs)
                elif method == 'change_admin_password':
                    result = self._change_admin_password(cursor, *args, **kwargs)
                elif method == 'get_student_by_id':
                    result = self._get_student_by_id(cursor, *args, **kwargs)
                else:
                    result = ValueError(f"Unknown method: {method}")
                
                conn.commit()
                self.result_queue.put(result)
            except Exception as e:
                self.result_queue.put(e)
            finally:
                self.db_queue.task_done()
    
    def execute(self, method, *args, **kwargs):
        """Execute a database method in the worker thread"""
        self.db_queue.put((method, args, kwargs))
        result = self.result_queue.get()
        if isinstance(result, Exception):
            raise result
        return result
    
    def _add_student(self, cursor, student_id, name):
        """Add a new student to the database"""
        registration_date = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        cursor.execute('''
            INSERT OR IGNORE INTO students (id, name, registration_date) 
            VALUES (?, ?, ?)
        ''', (student_id, name, registration_date))
        return cursor.rowcount > 0
    
    def _mark_attendance(self, cursor, student_id, name):
        """Mark attendance for a student"""
        date = datetime.datetime.now().strftime('%Y-%m-%d')
        time_str = datetime.datetime.now().strftime('%H:%M:%S')
        
        # Check if already marked today
        cursor.execute('''
            SELECT 1 FROM attendance 
            WHERE id = ? AND date = ?
        ''', (student_id, date))
        if cursor.fetchone() is None:
            cursor.execute('''
                INSERT INTO attendance (id, name, date, time)
                VALUES (?, ?, ?, ?)
            ''', (student_id, name, date, time_str))
            
            # Update last attendance in students table
            cursor.execute('''
                UPDATE students 
                SET last_attendance = ?
                WHERE id = ?
            ''', (f"{date} {time_str}", student_id))
            
            return True
        return False
    
    def _get_student_name(self, cursor, student_id):
        """Get student name by ID"""
        cursor.execute('SELECT name FROM students WHERE id = ?', (student_id,))
        result = cursor.fetchone()
        return result[0] if result else None
    
    def _get_student_by_id(self, cursor, student_id):
        """Get student record by ID"""
        cursor.execute('SELECT id, name FROM students WHERE id = ?', (student_id,))
        return cursor.fetchone()
    
    def _get_attendance_records(self, cursor, date=None):
        """Get attendance records for a specific date or all dates"""
        if date:
            cursor.execute('''
                SELECT id, name, date, time FROM attendance 
                WHERE date = ?
                ORDER BY time DESC
            ''', (date,))
        else:
            cursor.execute('''
                SELECT id, name, date, time FROM attendance 
                ORDER BY date DESC, time DESC
            ''')
        return cursor.fetchall()
    
    def _get_all_students(self, cursor):
        """Get all registered students"""
        cursor.execute('SELECT id, name FROM students ORDER BY name')
        return cursor.fetchall()
    
    def _verify_admin(self, cursor, username, password):
        """Verify admin credentials"""
        cursor.execute('''
            SELECT 1 FROM admin 
            WHERE username = ? AND password = ?
        ''', (username, password))
        return cursor.fetchone() is not None
    
    def _change_admin_password(self, cursor, username, new_password):
        """Change admin password"""
        cursor.execute('''
            UPDATE admin 
            SET password = ?
            WHERE username = ?
        ''', (new_password, username))
        return cursor.rowcount > 0
    
    def close(self):
        """Cleanly shutdown the database worker"""
        self.db_queue.put('STOP')
        self.worker_thread.join()

class SpiderWebBackground(tk.Canvas):
    def _init_(self, master, **kwargs):
        super()._init_(master, **kwargs)
        self.configure(bg='#0a0a0a', highlightthickness=0)
        self.web_center_x = self.winfo_reqwidth() // 2
        self.web_center_y = self.winfo_reqheight() // 2
        self.web_lines = []
        self.web_circles = []
        self.spider = None
        self.spider_position = [50, 750]
        self.spider_direction = [1, -1]  # x, y direction
        self.animate_spider = False
        self.particles = []
        self.draw_web()
        self.create_spider()
        self.create_particles(30)
        
    def draw_web(self):
        """Draw the spider web pattern"""
        # Clear existing web
        for line in self.web_lines:
            self.delete(line)
        for circle in self.web_circles:
            self.delete(circle)
            
        self.web_lines = []
        self.web_circles = []
        
        # Draw radial lines (spider web strands)
        for angle in range(0, 360, 30):
            end_x = self.web_center_x + 900 * np.cos(np.radians(angle))
            end_y = self.web_center_y + 900 * np.sin(np.radians(angle))
            line = self.create_line(
                self.web_center_x, self.web_center_y, end_x, end_y, 
                fill='#8b0000', width=2, dash=(3, 3), tags="web"
            )
            self.web_lines.append(line)
        
        # Draw concentric circles (web spirals)
        for r in range(100, 800, 100):
            circle = self.create_oval(
                self.web_center_x-r, self.web_center_y-r, 
                self.web_center_x+r, self.web_center_y+r,
                outline='#8b0000', width=1, dash=(5, 5), tags="web"
            )
            self.web_circles.append(circle)
    
    def create_spider(self):
        """Create a simple spider graphic"""
        try:
            # Try to load spider image if available
            spider_img = Image.open("spider.png").resize((100, 100))
            self.spider_img = ImageTk.PhotoImage(spider_img)
            self.spider = self.create_image(
                self.spider_position[0], self.spider_position[1], 
                image=self.spider_img, anchor="nw", tags="spider"
            )
        except:
            # Fallback to drawing a simple spider
            spider_img = Image.new('RGBA', (100, 100), (0, 0, 0, 0))
            draw = ImageDraw.Draw(spider_img)
            # Body
            draw.ellipse((40, 40, 60, 60), fill='black')
            # Head
            draw.ellipse((45, 30, 55, 40), fill='black')
            # Legs
            for angle in range(0, 360, 45):
                start_x = 50 + 10 * np.cos(np.radians(angle))
                start_y = 50 + 10 * np.sin(np.radians(angle))
                end_x = 50 + 30 * np.cos(np.radians(angle))
                end_y = 50 + 30 * np.sin(np.radians(angle))
                draw.line((start_x, start_y, end_x, end_y), fill='black', width=3)
            
            self.spider_img = ImageTk.PhotoImage(spider_img)
            self.spider = self.create_image(
                self.spider_position[0], self.spider_position[1], 
                image=self.spider_img, anchor="nw", tags="spider"
            )
    
    def create_particles(self, count):
        """Create floating particles for background effect"""
        for _ in range(count):
            x = random.randint(0, self.winfo_reqwidth())
            y = random.randint(0, self.winfo_reqheight())
            size = random.randint(1, 3)
            particle = self.create_oval(
                x, y, x+size, y+size, 
                fill='#8b0000', outline='', tags="particle"
            )
            self.particles.append({
                'id': particle,
                'x': x,
                'y': y,
                'dx': random.uniform(-0.5, 0.5),
                'dy': random.uniform(-0.5, 0.5)
            })
    
    def animate(self):
        """Animate the spider and particles"""
        if self.animate_spider:
            # Move spider
            self.spider_position[0] += self.spider_direction[0]
            self.spider_position[1] += self.spider_direction[1]
            
            # Bounce off edges
            if self.spider_position[0] <= 0 or self.spider_position[0] >= self.winfo_reqwidth() - 100:
                self.spider_direction[0] *= -1
                self.play_sound("bounce.wav")
            if self.spider_position[1] <= 0 or self.spider_position[1] >= self.winfo_reqheight() - 100:
                self.spider_direction[1] *= -1
                self.play_sound("bounce.wav")
            
            self.coords(self.spider, self.spider_position[0], self.spider_position[1])
        
        # Move particles
        for particle in self.particles:
            particle['x'] += particle['dx']
            particle['y'] += particle['dy']
            
            # Wrap around screen edges
            if particle['x'] < 0:
                particle['x'] = self.winfo_reqwidth()
            elif particle['x'] > self.winfo_reqwidth():
                particle['x'] = 0
            if particle['y'] < 0:
                particle['y'] = self.winfo_reqheight()
            elif particle['y'] > self.winfo_reqheight():
                particle['y'] = 0
                
            self.coords(particle['id'], particle['x'], particle['y'], 
                       particle['x']+3, particle['y']+3)
        
        self.after(30, self.animate)
    
    def play_sound(self, sound_file):
        """Play a sound effect if available"""
        sound_path = os.path.join(SOUNDS_DIR, sound_file)
        if os.path.exists(sound_path):
            try:
                winsound.PlaySound(sound_path, winsound.SND_ASYNC)
            except:
                pass

class AttendanceSystem:
    def _init_(self, window):
        self.window = window
        self.window.title("Student Attendance System")
        self.window.geometry("1600x800")
        self.window.state('zoomed')  # Start maximized
        
        # Initialize thread-safe database
        self.db = ThreadSafeDatabase()
        self.db_lock = threading.Lock()
        
        # Admin state
        self.admin_logged_in = False
        self.admin_username = "admin"  # Default admin username
        
        # Create spider web background
        self.canvas = SpiderWebBackground(self.window, width=1600, height=800)
        self.canvas.pack(fill="both", expand=True)
        
        # Start background animation
        self.canvas.animate_spider = True
        self.canvas.animate()
        
        # Setup UI
        self.setup_ui()
        
        # Sound effects
        self.load_sounds()
        
        # Current state
        self.camera_active = False
        self.capture_thread = None
        self.recognition_thread = None
        self.class_in_session = False
        self.class_end_time = None
        self.attendance_sessions = []
        
    def load_sounds(self):
        """Ensure sound files exist or create placeholders"""
        sounds = {
            "success.wav": 440,  # Frequency for simple tones
            "error.wav": 220,
            "capture.wav": 880,
            "bounce.wav": 660
        }
        
        for sound_file, frequency in sounds.items():
            sound_path = os.path.join(SOUNDS_DIR, sound_file)
            if not os.path.exists(sound_path):
                try:
                    duration = 300  # milliseconds
                    winsound.Beep(frequency, duration)
                except:
                    pass
    
    def setup_ui(self):
        """Create all UI elements"""
        self.create_main_title()
        self.create_input_section()
        self.create_action_buttons()
        self.create_attendance_display()
        self.create_admin_panel()
        self.create_status_bar()
        
    def create_main_title(self):
        """Create the main title with spider-web style"""
        title_font = ('Arial Black', 40, 'bold')
        self.canvas.create_text(
            800, 50, 
            text="🕷 STUDENT ATTENDANCE SYSTEM 🕸", 
            fill="#ff0000", font=title_font, anchor="center"
        )
        
        # College information
        self.canvas.create_text(
            1350, 700, 
            text="GEC College", 
            fill="#8b0000", font=('Arial', 20, 'bold italic')
        )
    
    def create_input_section(self):
        """Create input fields for student registration"""
        # Input frame
        input_frame = tk.Frame(self.window, bg='#1a1a1a', bd=2, relief='groove')
        input_frame.place(x=100, y=120, width=500, height=200)
        
        # Labels
        tk.Label(
            input_frame, text="STUDENT REGISTRATION", 
            bg='#1a1a1a', fg="#ff0000", font=('Arial', 16, 'bold')
        ).pack(pady=(10, 5))
        
        tk.Label(
            input_frame, text="College ID:", 
            bg='#1a1a1a', fg="white", font=('Arial', 12)
        ).pack(anchor='w', padx=20, pady=5)
        
        # ID Entry with black text
        self.id_entry = ttk.Entry(
            input_frame, font=('Arial', 12),
            style='BlackText.TEntry'
        )
        self.id_entry.pack(fill='x', padx=20, pady=5)
        
        tk.Label(
            input_frame, text="Full Name:", 
            bg='#1a1a1a', fg="white", font=('Arial', 12)
        ).pack(anchor='w', padx=20, pady=5)
        
        # Name Entry with black text
        self.name_entry = ttk.Entry(
            input_frame, font=('Arial', 12),
            style='BlackText.TEntry'
        )
        self.name_entry.pack(fill='x', padx=20, pady=5)
        
        # Configure styles
        style = ttk.Style()
        style.configure('BlackText.TEntry', 
                      fieldbackground='#2a2a2a', 
                      foreground='black',  # Black text color
                      insertbackground='black')  # Black cursor color
    
    def create_action_buttons(self):
        """Create main action buttons"""
        # Action frame
        action_frame = tk.Frame(self.window, bg='#1a1a1a', bd=2, relief='groove')
        action_frame.place(x=650, y=120, width=300, height=200)
        
        tk.Label(
            action_frame, text="ACTIONS", 
            bg='#1a1a1a', fg="#ff0000", font=('Arial', 16, 'bold')
        ).pack(pady=(10, 5))
        
        # Buttons
        button_style = {'bg': '#8b0000', 'fg': 'white', 'activebackground': '#ff0000',
                      'font': ('Arial', 12, 'bold'), 'bd': 0, 'padx': 10, 'pady': 5}
        
        tk.Button(
            action_frame, text="📷 CAPTURE IMAGES", 
            command=self.take_images, **button_style
        ).pack(fill='x', padx=20, pady=5)
        
        tk.Button(
            action_frame, text="⚙ TRAIN MODEL", 
            command=self.train_images, **button_style
        ).pack(fill='x', padx=20, pady=5)
        
        tk.Button(
            action_frame, text="✅ MARK ATTENDANCE", 
            command=self.toggle_class_session, **button_style
        ).pack(fill='x', padx=20, pady=5)
    
    def toggle_class_session(self):
        """Start or stop a class session for attendance"""
        if self.class_in_session:
            # End the current class session
            self.class_in_session = False
            self.class_end_time = None
            self.update_status("Class session ended")
            self.canvas.play_sound("success.wav")
        else:
            # Start a new class session
            duration = 5  # Default duration in minutes
            self.class_in_session = True
            self.class_end_time = datetime.datetime.now() + datetime.timedelta(minutes=duration)
            self.attendance_sessions = []
            
            # Start the attendance monitoring thread
            self.recognition_thread = threading.Thread(
                target=self._monitor_class_attendance,
                args=(duration,),
                daemon=True
            )
            self.recognition_thread.start()
            
            self.update_status(f"Class session started for {duration} minutes")
            self.canvas.play_sound("success.wav")
    
    def _monitor_class_attendance(self, duration_minutes):
        """Monitor attendance throughout the class session"""
        recognizer = cv2.face.LBPHFaceRecognizer_create()
        recognizer.read(TRAINNER_PATH)
        
        face_cascade = cv2.CascadeClassifier(HAARCASCADE_PATH)
        if face_cascade.empty():
            self.update_status("Error: Could not load face detection model")
            return
            
        cam = cv2.VideoCapture(0)
        if not cam.isOpened():
            self.update_status("Error: Could not open camera")
            return
            
        end_time = time.time() + (duration_minutes * 60)
        last_capture_time = 0
        capture_interval = random.randint(10, 30)  # Random interval between 10-30 seconds
        
        while self.class_in_session and time.time() < end_time:
            current_time = time.time()
            
            # Check if it's time for another capture
            if current_time - last_capture_time >= capture_interval:
                last_capture_time = current_time
                capture_interval = random.randint(10, 30)  # New random interval
                
                # Capture and process frame
                ret, frame = cam.read()
                if not ret:
                    continue
                    
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = face_cascade.detectMultiScale(gray, 1.2, 5)
                
                current_attendance = set()  # Track students in this capture
                
                for (x, y, w, h) in faces:
                    # Recognize face
                    id, confidence = recognizer.predict(gray[y:y+h, x:x+w])
                    
                    if confidence < 50:  # Confidence threshold
                        with self.db_lock:
                            student = self.db.execute('get_student_by_id', str(id))
                        
                        if student:
                            student_id, student_name = student
                            current_attendance.add(student_id)
                            
                            # Mark attendance if not already marked in this session
                            if student_id not in self.attendance_sessions:
                                with self.db_lock:
                                    marked = self.db.execute('mark_attendance', student_id, student_name)
                                
                                if marked:
                                    self.attendance_sessions.append(student_id)
                                    self.window.after(0, self.refresh_attendance)
                                    self.window.after(0, self.update_stats)
                                    self.canvas.play_sound("capture.wav")
                
                # Update status with current attendance count
                self.update_status(
                    f"Class in session: {len(self.attendance_sessions)} students marked. "
                    f"Next capture in {capture_interval}s"
                )
                
                # Display the captured frame with recognition results
                for (x, y, w, h) in faces:
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                
                cv2.imshow('Class Attendance Monitoring', frame)
                cv2.waitKey(1000)  # Show for 1 second
                cv2.destroyAllWindows()
            
            time.sleep(1)  # Check every second
            
        # Clean up
        cam.release()
        cv2.destroyAllWindows()
        
        # Update status
        if self.class_in_session:  # If we exited due to time
            self.class_in_session = False
            self.class_end_time = None
            self.update_status(
                f"Class session completed. {len(self.attendance_sessions)} students attended."
            )
            self.canvas.play_sound("success.wav")
    
    def create_attendance_display(self):
        """Create the attendance display area"""
        # Attendance frame
        attendance_frame = tk.Frame(self.window, bg='#1a1a1a', bd=2, relief='groove')
        attendance_frame.place(x=100, y=350, width=850, height=400)
        
        tk.Label(
            attendance_frame, text="ATTENDANCE RECORDS", 
            bg='#1a1a1a', fg="#ff0000", font=('Arial', 16, 'bold')
        ).pack(pady=(10, 5))
        
        # Date selector
        date_frame = tk.Frame(attendance_frame, bg='#1a1a1a')
        date_frame.pack(fill='x', padx=10, pady=5)
        
        tk.Label(
            date_frame, text="Select Date:", 
            bg='#1a1a1a', fg="white", font=('Arial', 10)
        ).pack(side='left', padx=5)
        
        self.date_var = tk.StringVar(value=datetime.datetime.now().strftime('%Y-%m-%d'))
        self.date_entry = ttk.Entry(
            date_frame, textvariable=self.date_var, 
            font=('Arial', 10), width=12,
            style='BlackText.TEntry'  # Using the same black text style
        )
        self.date_entry.pack(side='left', padx=5)
        
        tk.Button(
            date_frame, text="🔍", 
            command=self.refresh_attendance, bg='#8b0000', fg='white',
            font=('Arial', 10), bd=0
        ).pack(side='left', padx=5)
        
        tk.Button(
            date_frame, text="📊 Generate Report", 
            command=self.generate_report, bg='#8b0000', fg='white',
            font=('Arial', 10), bd=0
        ).pack(side='right', padx=5)
        
        # Attendance treeview
        self.attendance_tree = ttk.Treeview(
            attendance_frame, 
            columns=('id', 'name', 'date', 'time'), 
            show='headings', height=15
        )
        
        # Configure columns
        self.attendance_tree.heading('id', text='ID')
        self.attendance_tree.heading('name', text='Name')
        self.attendance_tree.heading('date', text='Date')
        self.attendance_tree.heading('time', text='Time')
        
        self.attendance_tree.column('id', width=100, anchor='center')
        self.attendance_tree.column('name', width=200, anchor='center')
        self.attendance_tree.column('date', width=100, anchor='center')
        self.attendance_tree.column('time', width=100, anchor='center')
        
        # Add scrollbar
        scrollbar = ttk.Scrollbar(
            attendance_frame, orient='vertical', 
            command=self.attendance_tree.yview
        )
        self.attendance_tree.configure(yscrollcommand=scrollbar.set)
        
        self.attendance_tree.pack(side='left', fill='both', expand=True)
        scrollbar.pack(side='right', fill='y')
        
        # Load initial attendance data
        self.refresh_attendance()
    
    def create_admin_panel(self):
        """Create admin controls panel with login protection"""
        # Admin frame
        admin_frame = tk.Frame(self.window, bg='#1a1a1a', bd=2, relief='groove')
        admin_frame.place(x=1000, y=120, width=500, height=400)
        
        tk.Label(
            admin_frame, text="ADMIN PANEL", 
            bg='#1a1a1a', fg="#ff0000", font=('Arial', 16, 'bold')
        ).pack(pady=(10, 5))
        
        # Username entry
        tk.Label(
            admin_frame, text="Username:", 
            bg='#1a1a1a', fg="white", font=('Arial', 10)
        ).pack(anchor='w', padx=20, pady=2)
        
        self.admin_user_var = tk.StringVar(value="admin")
        ttk.Entry(
            admin_frame, 
            textvariable=self.admin_user_var, 
            font=('Arial', 10), 
            style='BlackText.TEntry'
        ).pack(fill='x', padx=20, pady=2)
        
        # Password entry
        tk.Label(
            admin_frame, text="Password:", 
            bg='#1a1a1a', fg="white", font=('Arial', 10)
        ).pack(anchor='w', padx=20, pady=2)
        
        self.admin_pass_var = tk.StringVar()
        ttk.Entry(
            admin_frame, 
            textvariable=self.admin_pass_var, 
            font=('Arial', 10), 
            show='*', 
            style='BlackText.TEntry'
        ).pack(fill='x', padx=20, pady=2)
        
        # Login/Logout buttons
        login_frame = tk.Frame(admin_frame, bg='#1a1a1a')
        login_frame.pack(fill='x', padx=20, pady=5)
        
        self.login_btn = tk.Button(
            login_frame, text="🔓 Login", 
            command=self.admin_login,
            bg='#8b0000', fg='white', font=('Arial', 10, 'bold'),
            bd=0, padx=5, pady=2
        )
        self.login_btn.pack(side='left', expand=True)
        
        self.logout_btn = tk.Button(
            login_frame, text="🔒 Logout", 
            command=self.admin_logout,
            bg='#8b0000', fg='white', font=('Arial', 10, 'bold'),
            bd=0, padx=5, pady=2, state='disabled'
        )
        self.logout_btn.pack(side='right', expand=True)
        
        # Change password frame
        pass_frame = tk.Frame(admin_frame, bg='#1a1a1a')
        pass_frame.pack(fill='x', padx=20, pady=5)
        
        tk.Label(
            pass_frame, text="New Password:", 
            bg='#1a1a1a', fg="white", font=('Arial', 10)
        ).pack(anchor='w', padx=5, pady=2)
        
        self.new_pass_var = tk.StringVar()
        ttk.Entry(
            pass_frame, 
            textvariable=self.new_pass_var, 
            font=('Arial', 10), 
            show='*', 
            style='BlackText.TEntry'
        ).pack(side='left', fill='x', expand=True, padx=5)
        
        self.change_pass_btn = tk.Button(
            pass_frame, text="🔄 Change", 
            command=self.change_admin_password,
            bg='#8b0000', fg='white', font=('Arial', 10, 'bold'),
            bd=0, padx=5, pady=2, state='disabled'
        )
        self.change_pass_btn.pack(side='right', padx=5)
        
        # Admin status
        self.admin_status_label = tk.Label(
            admin_frame, text="Status: Not Logged In",
            bg='#1a1a1a', fg="#ff0000", font=('Arial', 10)
        )
        self.admin_status_label.pack(fill='x', padx=20, pady=5)
        
        # Admin functions (initially disabled)
        button_style = {
            'bg': '#8b0000', 'fg': 'white', 'activebackground': '#ff0000',
            'font': ('Arial', 10, 'bold'), 'bd': 0, 'state': 'disabled'
        }
        
        self.student_list_btn = tk.Button(
            admin_frame, text="📋 Student List", 
            command=self.show_student_list, **button_style
        )
        self.student_list_btn.pack(fill='x', padx=20, pady=5)
        
        self.refresh_data_btn = tk.Button(
            admin_frame, text="🔄 Refresh Data", 
            command=self.refresh_all_data, **button_style
        )
        self.refresh_data_btn.pack(fill='x', padx=20, pady=5)
        
        self.reset_system_btn = tk.Button(
            admin_frame, text="⚠ Reset System", 
            command=self.reset_system, **button_style
        )
        self.reset_system_btn.pack(fill='x', padx=20, pady=5)
        
        # Exit button (always available)
        tk.Button(
            admin_frame, text="🚪 Exit", 
            command=self.quit_window,
            bg='#8b0000', fg='white', font=('Arial', 10, 'bold'),
            bd=0, padx=5, pady=2
        ).pack(fill='x', padx=20, pady=5)
    
    def admin_login(self):
        """Verify admin credentials and enable features"""
        username = self.admin_user_var.get()
        password = self.admin_pass_var.get()
        
        try:
            with self.db_lock:
                valid = self.db.execute('verify_admin', username, password)
            
            if valid:
                self.admin_logged_in = True
                self.admin_username = username
                self.update_admin_ui()
                messagebox.showinfo("Success", "Admin privileges activated!")
                self.canvas.play_sound("success.wav")
            else:
                messagebox.showerror("Error", "Invalid username or password!")
                self.canvas.play_sound("error.wav")
        except Exception as e:
            messagebox.showerror("Error", f"Login failed: {str(e)}")
            self.canvas.play_sound("error.wav")
    
    def admin_logout(self):
        """Disable admin features"""
        self.admin_logged_in = False
        self.update_admin_ui()
        messagebox.showinfo("Info", "Admin privileges deactivated")
        self.canvas.play_sound("success.wav")
    
    def change_admin_password(self):
        """Change admin password"""
        if not self.admin_logged_in:
            messagebox.showwarning("Access Denied", "Admin login required!")
            return
            
        new_password = self.new_pass_var.get()
        if not new_password:
            messagebox.showwarning("Error", "Please enter a new password!")
            return
            
        try:
            with self.db_lock:
                success = self.db.execute('change_admin_password', 
                                        self.admin_username, new_password)
            
            if success:
                messagebox.showinfo("Success", "Password changed successfully!")
                self.canvas.play_sound("success.wav")
                self.new_pass_var.set("")  # Clear password field
            else:
                messagebox.showerror("Error", "Failed to change password!")
                self.canvas.play_sound("error.wav")
        except Exception as e:
            messagebox.showerror("Error",