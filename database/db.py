import sqlite3
from config import DATABASE_NAME

def init_db():
    conn = sqlite3.connect(DATABASE_NAME)
    cursor = conn.cursor()

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS focus_data (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        imu_mean REAL,
        imu_std REAL,
        hr_mean REAL,
        hr_std REAL,
        prediction TEXT,
        confidence REAL,
        timestamp TEXT
    )
    """)

    conn.commit()
    conn.close()


def insert_prediction(data):
    conn = sqlite3.connect(DATABASE_NAME)
    cursor = conn.cursor()

    cursor.execute("""
    INSERT INTO focus_data
    (imu_mean, imu_std, hr_mean, hr_std, prediction, confidence, timestamp)
    VALUES (?, ?, ?, ?, ?, ?, ?)
    """, data)

    conn.commit()
    conn.close()


def get_history():
    conn = sqlite3.connect(DATABASE_NAME)
    cursor = conn.cursor()

    cursor.execute("SELECT * FROM focus_data ORDER BY id DESC")
    rows = cursor.fetchall()

    conn.close()
    return rows