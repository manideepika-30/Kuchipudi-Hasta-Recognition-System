"""
Automated Test Suite for STAGE 4: Authentic Kuchipudi Curriculum & Modules.
"""
import sys
import os
import json
import urllib.request

# Add backend directory to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.database.session import SessionLocal
from app.models.curriculum import Module, Lesson, StudentLessonProgress
from app.models.user import User

def run_stage4_tests():
    print("=" * 70)
    print("RUNNING STAGE 4: AUTHENTIC CURRICULUM VERIFICATION SUITE")
    print("=" * 70)

    db = SessionLocal()
    try:
        # 1. Module Count & Names
        print("\n[TEST 1] Verifying All 7 Canonical Modules...")
        modules = db.query(Module).order_by(Module.module_number).all()
        assert len(modules) == 7, f"Expected 7 modules, found {len(modules)}"
        
        expected_modules = {
            1: ("Introduction to Kuchipudi", "కూచిపూడి పరిచయం", 4),
            2: ("Asamyukta Hasthas", "అసంయుక్త హస్తాలు (28 ఏకహస్త ముద్రలు)", 28),
            3: ("Samyukta Hasthas", "సంయుక్త హస్తాలు (24 ద్విహస్త ముద్రలు)", 24),
            4: ("Shiro Bhedhas", "శిరో భేదాలు (9 శిరస్సు కదలికలు)", 9),
            5: ("Drishti Bhedhas", "దృష్టి భేదాలు (8 నేత్ర కదలికలు)", 8),
            6: ("Dvadasha Hasta Pranamas", "ద్వాదశ హస్త ప్రణామాలు (12 నమస్కార ముద్రలు)", 12),
            7: ("Sapta Talamulu", "సప్త తాళములు (7 ప్రధాన తాళాలు మరియు 35 తాళ వ్యవస్థ)", 7)
        }

        total_lessons = 0
        for m in modules:
            exp = expected_modules[m.module_number]
            assert m.title_en == exp[0], f"Title mismatch: {m.title_en} != {exp[0]}"
            assert len(m.lessons) == exp[2], f"Module {m.module_number} lessons count: {len(m.lessons)} != {exp[2]}"
            total_lessons += len(m.lessons)
            print(f"  ✓ Module {m.module_number}: {m.title_en} ({m.title_te}) — {len(m.lessons)} lessons")

        print(f"  -> Total Canonical Lessons: {total_lessons}")
        assert total_lessons == 92, f"Expected 92 lessons, found {total_lessons}"
        print("  -> [PASS] All 7 modules and 92 lessons verified in SQLite.")

        # 2. Verify Shlokas and PDF Image References
        print("\n[TEST 2] Verifying Shlokas, Translations & Authentic Images...")
        lessons_with_images = db.query(Lesson).filter(Lesson.image_url.isnot(None)).count()
        lessons_with_shlokas = db.query(Lesson).filter(Lesson.shloka.isnot(None)).count()
        print(f"  ✓ Lessons with authentic images: {lessons_with_images} / 92")
        print(f"  ✓ Lessons with canonical shlokas: {lessons_with_shlokas} / 92")
        assert lessons_with_images >= 61, f"Expected at least 61 lessons with images, found {lessons_with_images}"
        assert lessons_with_shlokas == 92, f"Expected 92 lessons with shlokas, found {lessons_with_shlokas}"
        print("  -> [PASS] Shlokas and image bindings verified.")

        # 3. Test Progress Persistence in SQLite
        print("\n[TEST 3] Testing Lesson Progress Toggle & Persistence...")
        test_student = db.query(User).filter(User.role == "student").first()
        assert test_student is not None, "No student found in DB"
        
        # Test toggling lesson 2 of module 2 (Tripathaka)
        les_target = db.query(Lesson).filter(Lesson.module_id == 2, Lesson.lesson_order == 2).first()
        assert les_target is not None

        # Clean prior progress if any
        db.query(StudentLessonProgress).filter(
            StudentLessonProgress.user_id == test_student.id,
            StudentLessonProgress.lesson_id == les_target.id
        ).delete()
        db.commit()

        # Record progress
        prog = StudentLessonProgress(
            user_id=test_student.id,
            module_id=2,
            lesson_id=les_target.id,
            is_completed=True
        )
        db.add(prog)
        db.commit()

        # Check persistence
        saved_prog = db.query(StudentLessonProgress).filter(
            StudentLessonProgress.user_id == test_student.id,
            StudentLessonProgress.lesson_id == les_target.id,
            StudentLessonProgress.is_completed == True
        ).first()
        assert saved_prog is not None
        print(f"  ✓ Persisted lesson completion for User {test_student.id}: {les_target.name_en} ({les_target.name_te})")
        print("  -> [PASS] Lesson progress successfully stored and retrieved from SQLite.")

        # 4. Verify Physical Image Files in frontend/public/assets
        print("\n[TEST 4] Verifying Physical Extracted Image Files...")
        all_lessons = db.query(Lesson).filter(Lesson.image_url.isnot(None)).all()
        missing_count = 0
        missing_files = []
        for l in all_lessons:
            if l.image_url and l.image_url.startswith("/assets/"):
                frontend_public = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "frontend", "public"))
                rel_path = os.path.join(frontend_public, l.image_url.lstrip("/"))
                if not os.path.exists(rel_path):
                    missing_count += 1
                    missing_files.append((l.id, l.name_en, l.image_url))
        assert missing_count == 0, f"Missing {missing_count} image files: {missing_files}"
        print(f"  ✓ All {len(all_lessons)} referenced image files physically exist on disk.")
        print("  -> [PASS] Image asset integrity confirmed.")

        print("\n" + "=" * 70)
        print("STAGE 4 CURRICULUM VERIFICATION PASSED (4/4)!")
        print("=" * 70)

    finally:
        db.close()

if __name__ == "__main__":
    run_stage4_tests()
