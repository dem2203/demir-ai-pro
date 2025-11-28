"""
Database Migration Runner
Executes SQL migration files from database/migrations/
"""

import os
import sys
import glob

# -------------------------------------------------------------------
# Ensure project root is on PYTHONPATH so "config" can be imported
# -------------------------------------------------------------------
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from config import DATABASE_URL


def run_migrations() -> bool:
    """Execute all SQL migration files"""
    # Ensure psycopg2 is available
    try:
        import psycopg2
    except ImportError:
        print("❌ psycopg2 not installed. Installing...")
        os.system("pip install psycopg2-binary")
        import psycopg2  # type: ignore
    
    # Connect to database
    try:
        conn = psycopg2.connect(DATABASE_URL)  # type: ignore
        cursor = conn.cursor()
        print("✅ Connected to database")
    except Exception as e:
        print(f"❌ Connection failed: {e}")
        return False
    
    # Find all migration files
    migrations_dir = os.path.join(PROJECT_ROOT, "database", "migrations")
    migration_files = sorted(glob.glob(os.path.join(migrations_dir, "*.sql")))
    
    if not migration_files:
        print("⚠️  No migration files found")
        cursor.close()
        conn.close()
        return True
    
    # Execute each migration
    for migration_file in migration_files:
        try:
            with open(migration_file, "r", encoding="utf-8") as f:
                sql_content = f.read()
            
            print(f"📝 Running: {os.path.basename(migration_file)}")
            cursor.execute(sql_content)
            conn.commit()
            print(f"✅ Completed: {os.path.basename(migration_file)}")
        except Exception as e:
            conn.rollback()
            print(f"❌ Error in {migration_file}: {e}")
            cursor.close()
            conn.close()
            return False
    
    cursor.close()
    conn.close()
    print("\n✅ All migrations completed successfully!")
    return True


if __name__ == "__main__":
    success = run_migrations()
    sys.exit(0 if success else 1)
