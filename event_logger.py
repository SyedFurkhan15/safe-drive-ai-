import csv
import os
from datetime import datetime


class EventLogger:
    def __init__(self, filename="logs.csv"):
        self.filename = filename

        # Create CSV file if it doesn't exist
        if not os.path.exists(self.filename):
            with open(self.filename, mode="w", newline="", encoding="utf-8") as file:
                writer = csv.writer(file)

                writer.writerow([
                    "Date",
                    "Time",
                    "Event Type",
                    "Severity",
                    "Risk Score",
                    "Additional Information"
                ])

    def log_event(
        self,
        event_type,
        severity="medium",
        risk_score=0,
        additional_info=""
    ):

        now = datetime.now()

        with open(self.filename, mode="a", newline="", encoding="utf-8") as file:
            writer = csv.writer(file)

            writer.writerow([
                now.strftime("%Y-%m-%d"),
                now.strftime("%H:%M:%S"),
                event_type,
                severity,
                risk_score,
                additional_info
            ])

        print(f"Logged: {event_type}")

    def get_recent_events(self, limit=10):

        if not os.path.exists(self.filename):
            return []

        with open(self.filename, mode="r", encoding="utf-8") as file:
            reader = list(csv.reader(file))

            if len(reader) <= 1:
                return []

            return reader[-limit:]

    def clear_logs(self):

        with open(self.filename, mode="w", newline="", encoding="utf-8") as file:
            writer = csv.writer(file)

            writer.writerow([
                "Date",
                "Time",
                "Event Type",
                "Severity",
                "Risk Score",
                "Additional Information"
            ])

        print("Logs cleared successfully.")


# =====================================================
# Example Usage
# =====================================================

if __name__ == "__main__":

    logger = EventLogger()

    logger.log_event(
        event_type="PHONE DETECTED",
        severity="high",
        risk_score=45,
        additional_info="Driver using mobile phone"
    )

    logger.log_event(
        event_type="DROWSINESS DETECTED",
        severity="critical",
        risk_score=90,
        additional_info="Eyes closed for more than 3 seconds"
    )

    logger.log_event(
        event_type="DISTRACTION DETECTED",
        severity="medium",
        risk_score=35,
        additional_info="Driver looking away from road"
    )

    logger.log_event(
        event_type="HIGH RISK",
        severity="critical",
        risk_score=95,
        additional_info="Multiple dangerous conditions detected"
    )

    print("\nRecent Events:\n")

    events = logger.get_recent_events()

    for event in events:
        print(event)