"""
Database Migration: Add Multi-Model Support
===========================================

Adds columns to support multi-model AI detection:
- cameras.enabled_models_json
- cameras.model_conf_overrides_json  
- zone_configs.zone_models_json

Run this script to update existing databases without data loss.
"""

import sqlite3
import sys
from pathlib import Path

# Add parent directory to path to import config
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import settings


def migrate():
    """Add multi-model support columns to cameras and zone_configs tables."""
    
    # Extract database path from DATABASE_URL
    if settings.DATABASE_URL.startswith("sqlite:///"):
        db_path = settings.DATABASE_URL.replace("sqlite:///", "")
        if db_path.startswith("./"):
            db_path = str(Path(__file__).parent.parent / db_path[2:])
    else:
        print("❌ Only SQLite databases are supported by this migration script")
        return False

    print(f"📂 Database: {db_path}")
    
    if not Path(db_path).exists():
        print("❌ Database file not found")
        return False

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    try:
        # Check if columns already exist
        cursor.execute("PRAGMA table_info(cameras)")
        camera_columns = [row[1] for row in cursor.fetchall()]
        
        cursor.execute("PRAGMA table_info(zone_configs)")
        zone_columns = [row[1] for row in cursor.fetchall()]

        migrations_applied = []

        # Add enabled_models_json to cameras table
        if "enabled_models_json" not in camera_columns:
            print("➕ Adding cameras.enabled_models_json...")
            cursor.execute("""
                ALTER TABLE cameras 
                ADD COLUMN enabled_models_json TEXT
            """)
            migrations_applied.append("cameras.enabled_models_json")
        else:
            print("✓ cameras.enabled_models_json already exists")

        # Add model_conf_overrides_json to cameras table
        if "model_conf_overrides_json" not in camera_columns:
            print("➕ Adding cameras.model_conf_overrides_json...")
            cursor.execute("""
                ALTER TABLE cameras 
                ADD COLUMN model_conf_overrides_json TEXT
            """)
            migrations_applied.append("cameras.model_conf_overrides_json")
        else:
            print("✓ cameras.model_conf_overrides_json already exists")

        # Add zone_models_json to zone_configs table
        if "zone_models_json" not in zone_columns:
            print("➕ Adding zone_configs.zone_models_json...")
            cursor.execute("""
                ALTER TABLE zone_configs 
                ADD COLUMN zone_models_json TEXT
            """)
            migrations_applied.append("zone_configs.zone_models_json")
        else:
            print("✓ zone_configs.zone_models_json already exists")

        # Commit changes
        conn.commit()
        
        if migrations_applied:
            print(f"\n✅ Migration complete! Applied {len(migrations_applied)} changes:")
            for change in migrations_applied:
                print(f"   - {change}")
        else:
            print("\n✅ Database already up to date - no changes needed")

        return True

    except Exception as e:
        print(f"\n❌ Migration failed: {e}")
        conn.rollback()
        return False
    
    finally:
        conn.close()


if __name__ == "__main__":
    print("="*70)
    print("Multi-Model Support Migration")
    print("="*70)
    print()
    
    success = migrate()
    
    print()
    print("="*70)
    if success:
        print("✅ Migration completed successfully")
    else:
        print("❌ Migration failed")
    print("="*70)
    
    sys.exit(0 if success else 1)
