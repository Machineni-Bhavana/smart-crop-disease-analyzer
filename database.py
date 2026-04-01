import sqlite3  # type: ignore
import pandas as pd  # type: ignore
from datetime import datetime
import os

DB_PATH = "data/predictions.db"


def create_connection():
    os.makedirs("data", exist_ok=True)
    conn = None
    try:
        conn = sqlite3.connect(DB_PATH)
    except sqlite3.Error as e:
        print(f"Error connecting to database: {e}")
    return conn


def init_db():
    conn = create_connection()
    if conn is not None:
        try:
            cursor = conn.cursor()
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS predictions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    date TEXT NOT NULL,
                    crop_type TEXT,
                    disease_name TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    is_blurry BOOLEAN NOT NULL,
                    latency REAL NOT NULL
                )
            """
            )
            conn.commit()
        except sqlite3.Error as e:
            print(f"Failed to initialize DB: {e}")
        finally:
            conn.close()


def log_prediction(crop_type, disease_name, confidence, is_blurry, latency):
    conn = create_connection()
    if conn is not None:
        try:
            cursor = conn.cursor()
            current_date = datetime.now().isoformat()
            cursor.execute(
                """
                INSERT INTO predictions(date, crop_type, disease_name, confidence, is_blurry, latency)
                VALUES(?,?,?,?,?,?)
            """,
                (current_date, crop_type, disease_name, confidence, is_blurry, latency),
            )
            conn.commit()
        except sqlite3.Error as e:
            print(f"Failed to log prediction: {e}")
        finally:
            conn.close()


def get_analytics_data():
    conn = create_connection()
    if conn is not None:
        try:
            df = pd.read_sql_query("SELECT * FROM predictions", conn)
            # Transform 'date' to datetime
            df["date"] = pd.to_datetime(df["date"])
            # Extract basic crop info if not specifically logged (assume first word is crop name for now)
            # E.g., 'Apple Black Rot' -> 'Apple'
            if (
                df.empty
                or all(pd.isna(x) for x in df["crop_type"])
                or all(x == "" for x in df["crop_type"])
            ):
                df["crop_type"] = df["disease_name"].apply(
                    lambda x: x.split(" ")[0] if " " in x else x
                )
            return df
        except Exception as e:
            print(f"Failed to retrieve data: {e}")
        finally:
            conn.close()
    return pd.DataFrame()
