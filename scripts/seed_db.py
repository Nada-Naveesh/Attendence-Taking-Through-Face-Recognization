"""
Database seeding script to initialize sample student records and attendance history.
Generates synthetic 512-D normalized biometric vectors for testing.
"""

from datetime import datetime, timedelta
import numpy as np
from backend.db.session import SessionLocal, init_db
from backend.db.models import Student, Attendance


def seed_database():
    """Populate database with benchmark student profiles and attendance records."""
    init_db()
    db = SessionLocal()

    try:
        # Check if already seeded
        if db.query(Student).count() > 0:
            print("[INFO] Database already contains student records. Skipping seeding.")
            return

        print("[INFO] Seeding database with benchmark student profiles...")

        students_data = [
            ("Alan Turing", "CS-2024-001"),
            ("Ada Lovelace", "CS-2024-002"),
            ("Claude Shannon", "EE-2024-001"),
            ("Grace Hopper", "CS-2024-003"),
            ("Geoffrey Hinton", "AI-2024-001"),
            ("Yann LeCun", "AI-2024-002"),
            ("Yoshua Bengio", "AI-2024-003"),
            ("Fei-Fei Li", "AI-2024-004"),
        ]

        rng = np.random.RandomState(42)
        created_students = []

        for name, roll_no in students_data:
            # Generate normalized 512-D synthetic embedding vector
            vec = rng.randn(512).astype(np.float32)
            vec /= np.linalg.norm(vec)

            student = Student(
                name=name,
                roll_no=roll_no,
                created_at=datetime.utcnow() - timedelta(days=30)
            )
            student.set_embedding(vec)
            db.add(student)
            created_students.append(student)

        db.commit()
        for s in created_students:
            db.refresh(s)

        print(f"[SUCCESS] Enrolled {len(created_students)} students.")

        # Generate attendance records across past 14 days
        print("[INFO] Generating historical attendance records...")
        attendance_records = []
        base_time = datetime.utcnow()

        for day_offset in range(14, 0, -1):
            log_date = base_time - timedelta(days=day_offset)
            # Skip weekends (5=Saturday, 6=Sunday)
            if log_date.weekday() >= 5:
                continue

            for student in created_students:
                # 85% probability of attendance, higher for Turing/Lovelace
                prob = 0.95 if "Turing" in student.name or "Lovelace" in student.name else 0.80
                if rng.rand() < prob:
                    att = Attendance(
                        student_id=student.id,
                        timestamp=log_date.replace(hour=9, minute=int(rng.randint(0, 30)), second=int(rng.randint(0, 59))),
                        confidence=round(float(rng.uniform(0.82, 0.98)), 4),
                        liveness_passed=True
                    )
                    attendance_records.append(att)

        db.add_all(attendance_records)
        db.commit()
        print(f"[SUCCESS] Inserted {len(attendance_records)} attendance logs.")

    finally:
        db.close()


if __name__ == "__main__":
    seed_database()
