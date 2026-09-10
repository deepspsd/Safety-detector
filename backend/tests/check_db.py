import sqlite3
import os

db_paths = ['safety.db', 'backend/safety.db', 'backend/safety_detection.db']
for db_path in db_paths:
    if os.path.exists(db_path):
        print(f"=== {db_path} ===")
        conn = sqlite3.connect(db_path)
        cur = conn.cursor()
        cur.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = [r[0] for r in cur.fetchall()]
        print("Tables:", tables)
        for t in ['alerts', 'alert_cases', 'camera_snapshots', 'evidence', 'cameras', 'violations']:
            if t in tables:
                cur.execute(f"SELECT count(*) FROM {t}")
                print(f"  {t}:", cur.fetchone()[0])
    for t in ['alerts', 'alert_cases', 'camera_snapshots', 'evidence', 'cameras']:
        if t in tables:
            cur.execute(f"SELECT count(*) FROM {t}")
            print(f"{t}:", cur.fetchone()[0])
            if t == 'alerts':
                cur.execute("SELECT alert_type, count(*) FROM alerts GROUP BY alert_type")
                print("Alert types:", cur.fetchall())
else:
    print("No DB found")
