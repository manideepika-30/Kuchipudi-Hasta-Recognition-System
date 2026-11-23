"""
Script to create or promote an administrative user securely.
Usage:
    python backend/scripts/create_admin.py --email admin@kuchipudi.edu --password <secret_password> --name "System Administrator"
"""
import sys
import os
import argparse

# Add backend directory to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.database.session import SessionLocal, engine
from app.database.base import Base
from app.models.user import User, StudentProfile
from app.utils.security import get_password_hash

def create_or_promote_admin(email: str, password: str, name: str):
    Base.metadata.create_all(bind=engine)
    db = SessionLocal()
    try:
        user = db.query(User).filter(User.email == email.lower()).first()
        if user:
            print(f"User with email '{email}' already exists. Updating role to 'admin'...")
            user.role = "admin"
            user.hashed_password = get_password_hash(password)
            if user.profile:
                user.profile.full_name = name
            else:
                profile = StudentProfile(user_id=user.id, full_name=name)
                db.add(profile)
            db.commit()
            print(f"[SUCCESS] User '{email}' promoted to Administrator successfully.")
        else:
            print(f"Creating new Administrator account for '{email}'...")
            hashed_pwd = get_password_hash(password)
            admin_user = User(
                email=email.lower(),
                hashed_password=hashed_pwd,
                role="admin",
                is_active=True
            )
            db.add(admin_user)
            db.commit()
            db.refresh(admin_user)

            profile = StudentProfile(
                user_id=admin_user.id,
                full_name=name,
                preferred_language="en",
                experience_level="Administrator"
            )
            db.add(profile)
            db.commit()
            print(f"[SUCCESS] Admin user '{email}' created successfully (User ID: {admin_user.id}).")
    finally:
        db.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create or promote an admin account for Kuchipudi Platform.")
    parser.add_argument("--email", required=True, help="Admin email address")
    parser.add_argument("--password", required=True, help="Admin password (at least 6 characters)")
    parser.add_argument("--name", default="System Administrator", help="Admin display name")

    args = parser.parse_args()
    if len(args.password) < 6:
        print("[ERROR] Password must be at least 6 characters long.")
        sys.exit(1)

    create_or_promote_admin(args.email, args.password, args.name)
