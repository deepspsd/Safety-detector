"""
Migration: Add thumbnail_b64 column to face_encodings table.
Run once: python migrate_thumbnail.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

from database import engine
from sqlalchemy import text

try:
    with engine.connect() as conn:
        conn.execute(text("ALTER TABLE face_encodings ADD COLUMN thumbnail_b64 TEXT"))
        conn.commit()
    print("[OK] thumbnail_b64 column added to face_encodings")
except Exception as e:
    if "duplicate column" in str(e).lower() or "already exists" in str(e).lower():
        print("[SKIP] Column already exists — nothing to do")
    else:
        print(f"[ERROR] {e}")
        raise
