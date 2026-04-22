"""
Quick diagnostic script — run this to find what's stopping the backend from starting.
Usage: venv\Scripts\python check_startup.py
"""
import sys
import traceback

print("=" * 60)
print("OccuSafe Backend Startup Diagnostics")
print("=" * 60)

errors = []

# Test 1: Database
print("\n[1] Testing database connection...")
try:
    from database import engine, create_tables, User, FaceEncoding
    with engine.connect() as conn:
        from sqlalchemy import text
        result = conn.execute(text("SELECT COUNT(*) FROM users"))
        count = result.scalar()
    print(f"    ✓ Database OK — {count} users found")
    # Check thumbnail_b64 column
    try:
        conn2 = engine.connect()
        conn2.execute(text("SELECT thumbnail_b64 FROM face_encodings LIMIT 1"))
        conn2.close()
        print("    ✓ thumbnail_b64 column exists")
    except Exception:
        print("    ! thumbnail_b64 column missing — running migration...")
        with engine.connect() as mc:
            mc.execute(text("ALTER TABLE face_encodings ADD COLUMN thumbnail_b64 TEXT"))
            mc.commit()
        print("    ✓ Migration applied")
except Exception as e:
    errors.append(f"Database: {e}")
    print(f"    ✗ FAILED: {e}")
    traceback.print_exc()

# Test 2: YOLO service
print("\n[2] Testing YOLO service...")
try:
    from services.yolo_service import load_model
    load_model()
    print("    ✓ YOLO service OK")
except Exception as e:
    errors.append(f"YOLO: {e}")
    print(f"    ✗ FAILED: {e}")

# Test 3: Face service
print("\n[3] Testing face service...")
try:
    from services import face_service
    print(f"    ✓ face_service OK (simulation={face_service._use_simulation})")
except Exception as e:
    errors.append(f"FaceService: {e}")
    print(f"    ✗ FAILED: {e}")
    traceback.print_exc()

# Test 4: Routers
print("\n[4] Testing all routers...")
routers = ["auth", "users", "alerts", "detection", "video", "faces", "cctv"]
for name in routers:
    try:
        mod = __import__(f"routers.{name}", fromlist=[name])
        print(f"    ✓ routers.{name}")
    except Exception as e:
        errors.append(f"Router {name}: {e}")
        print(f"    ✗ routers.{name} FAILED: {e}")
        traceback.print_exc()

# Test 5: FastAPI app
print("\n[5] Testing FastAPI app creation...")
try:
    from main import app
    print("    ✓ FastAPI app created OK")
except Exception as e:
    errors.append(f"App: {e}")
    print(f"    ✗ FAILED: {e}")
    traceback.print_exc()

# Summary
print("\n" + "=" * 60)
if errors:
    print(f"FAILED — {len(errors)} error(s):")
    for err in errors:
        print(f"  • {err}")
    print("\nFix the errors above, then restart the backend.")
else:
    print("ALL CHECKS PASSED ✓")
    print("Backend is ready to start with:")
    print("  venv\\Scripts\\python -m uvicorn main:app --reload --port 8000")
print("=" * 60)
