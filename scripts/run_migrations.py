"""
Database Migration Runner
Executes SQL migration files from database/migrations/
"""

import os
import glob
from config import DATABASE_URL

def run_migrations():
    """Execute all SQL migration files"""
    try:
        import psycopg2
    except ImportError:
        print("❌ psycopg2 not installed. Installing...")
        os.system("pip install psycopg2-binary")
        import psycopg2
    
    # Connect to database
    try:
        conn = psycopg2.connect(DATABASE_URL)
        cursor = conn.cursor()
        print("✅ Connected to database")
    except Exception as e:
        print(f"❌ Connection failed: {e}")
        return False
    
    # Find all migration files
    migrations_dir = "database/migrations"
    migration_files = sorted(glob.glob(f"{migrations_dir}/*.sql"))
    
    if not migration_files:
        print("⚠️  No migration files found")
        return True
    
    # Execute each migration
    for migration_file in migration_files:
        try:
            with open(migration_file, "r") as f:
                sql_content = f.read()
            
            print(f"📝 Running: {os.path.basename(migration_file)}")
            cursor.execute(sql_content)
            conn.commit()
            print(f"✅ Completed: {os.path.basename(migration_file)}")
        except Exception as e:
            conn.rollback()
            print(f"❌ Error in {migration_file}: {e}")
            return False
    
    cursor.close()
    conn.close()
    print("\n✅ All migrations completed successfully!")
    return True


if __name__ == "__main__":
    success = run_migrations()
    exit(0 if success else 1)
