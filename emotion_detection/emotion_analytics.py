import csv
import os
from datetime import datetime
from pathlib import Path


LOG_PATH = Path("emotion_log.csv")
LOG_COLUMNS = ["timestamp", "identity", "emotion"]


def log_emotion(emotion, identity="unknown", log_path=LOG_PATH):
    """Append one emotion prediction to the session analytics CSV."""
    log_path = Path(log_path)
    file_exists = log_path.exists()

    with log_path.open("a", newline="") as file:
        writer = csv.writer(file)

        if not file_exists or file.tell() == 0:
            writer.writerow(LOG_COLUMNS)

        writer.writerow(
            [
                datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                identity,
                emotion,
            ]
        )


def load_emotion_log(log_path=LOG_PATH):
    """Load logged emotion predictions into a pandas DataFrame."""
    import pandas as pd

    log_path = Path(log_path)

    if not log_path.exists():
        raise FileNotFoundError(f"No emotion log found at {log_path}")

    return pd.read_csv(log_path)


def show_analytics(log_path=LOG_PATH):
    """Print emotion counts and display a pie chart of the distribution."""
    try:
        os.environ.setdefault("MPLCONFIGDIR", str(Path(".cache/matplotlib")))
        Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

        import matplotlib.pyplot as plt

        emotion_log = load_emotion_log(log_path)

        if emotion_log.empty:
            print("Emotion analytics unavailable: log file is empty")
            return

        emotion_counts = emotion_log["emotion"].value_counts()

        print("\nEmotion Summary")
        print(emotion_counts)

        plt.figure(figsize=(6, 6))
        emotion_counts.plot.pie(autopct="%1.1f%%", startangle=90)
        plt.title("Emotion Distribution")
        plt.ylabel("")
        plt.tight_layout()
        plt.show()

    except Exception as error:
        print("Emotion analytics unavailable:", error)
