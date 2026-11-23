"""
Comprehensive Database Seeder for all 7 Kuchipudi Theory Modules.
Strictly sourced from authentic PDF documents provided by Anandamayi Nrityalaya.
"""
import sys
import os

# Add backend directory to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.database.session import SessionLocal, engine
from app.database.base import Base
from app.models.curriculum import Module, Lesson

def seed_authentic_curriculum():
    print("=" * 60)
    print("SEEDING ALL 7 KUCHIPUDI MODULES FROM AUTHENTIC PDF CONTENT")
    print("=" * 60)

    Base.metadata.create_all(bind=engine)
    db = SessionLocal()

    try:
        # Clear existing modules and lessons for fresh canonical seeding
        db.query(Lesson).delete()
        db.query(Module).delete()
        db.commit()

        # =====================================================================
        # MODULE 1: Introduction to Kuchipudi
        # =====================================================================
        m1 = Module(
            module_number=1,
            title_en="Introduction to Kuchipudi",
            title_te="కూచిపూడి పరిచయం",
            sanskrit_title="Kuchipudi Natya Parichayam",
            source_pdf="Traditional Natyashastra Canon",
            total_classes=4,
            description_en="The historical evolution, Natyashastra roots, and spiritual significance of classical Kuchipudi dance.",
            description_te="కూచిపూడి నృత్యం యొక్క చారిత్రక పరిణామం, నాట్యశాస్త్ర మూలాలు మరియు ఆధ్యాత్మిక ప్రాముఖ్యత."
        )
        db.add(m1)
        db.commit()
        db.refresh(m1)

        m1_lessons = [
            {
                "lesson_order": 1,
                "name_en": "Historical Origins & Siddhendra Yogi",
                "name_te": "చారిత్రక మూలాలు మరియు సిద్ధేంద్ర యోగి",
                "sanskrit_name": "Itihasa & Guru Siddhendra Yogi",
                "category": "Origins",
                "description_en": "Kuchipudi originated in Kuchelapuram village in Krishna District, Andhra Pradesh. Saint Siddhendra Yogi formalized the Yakshagana style into classical dance drama, authoring the immortal masterpiece 'Bhama Kalapam'.",
                "description_te": "కృష్ణా జిల్లాలోని కుచేలపురం గ్రామంలో ఆవిర్భవించిన కూచిపూడి. సిద్ధేంద్ర యోగి రచించిన భామా కలాపం దీనికి ప్రాణవాయువు.",
                "shloka": "యతో హస్తస్తతో దృష్టిర్యతో దృష్టిస్తతో మనః | యతో మనస్తతో భావో యతో భావస్తతో రసః ||",
                "meaning_en": "Where the hands go, the gaze follows; where the gaze goes, the mind follows; where the mind goes, emotion is created; where emotion is created, aesthetic bliss (Rasa) is born.",
                "meaning_te": "చేయి ఎటు వెళితే దృష్టి అటు వెళుతుంది; దృష్టి ఎటు వెళితే మనసు అటు వెళుతుంది; మనసు ఉన్నచోట భావం, భావం ఉన్నచోట రససిద్ధి కలుగుతుంది."
            },
            {
                "lesson_order": 2,
                "name_en": "Foundations: Nritta, Nritya and Natya",
                "name_te": "నృత్త, నృత్య మరియు నాట్యాల సంగమం",
                "sanskrit_name": "Nritta-Nritya-Natya Lakshanam",
                "category": "Foundations",
                "description_en": "Nritta represents pure abstract rhythm without emotional storytelling. Nritya infuses rhythmic footwork with expressive gesture and emotional meaning. Natya is the complete dramatic play combining dialogue, music, and characterization.",
                "description_te": "నృత్తం కేవలం లయబద్ధమైన కదలికలు; నృత్యం భావం మరియు అభినయంతో కూడినది; నాట్యం కథ మరియు పాత్రలతో కూడిన సంపూర్ణ నాటకరూపం.",
                "shloka": "నృత్తం తాళలయాశ్రయమ్ | భావాశ్రయం నృత్యమ్ | నాట్యం రసాశ్రయమ్ ||",
                "meaning_en": "Nritta depends on Tala and Laya; Nritya depends on Bhava; Natya depends on Rasa.",
                "meaning_te": "నృత్తం లయ-తాళాలను ఆశ్రయిస్తుంది; నృత్యం భావాన్ని ఆశ్రయిస్తుంది; నాట్యం రసాన్ని ఆశ్రయిస్తుంది."
            },
            {
                "lesson_order": 3,
                "name_en": "Chaturvidha Abhinaya (Four Types of Acting)",
                "name_te": "చతుర్విధ అభినయాలు",
                "sanskrit_name": "Chaturvidha Abhinaya",
                "category": "Abhinaya",
                "description_en": "Classical Indian dance communication is quadrufold: Angika (body gestures and postures), Vachika (speech, lyrical song and rhythm), Aharya (costumes, jewelry, and stage decor), and Sattvika (authentic inner emotional states).",
                "description_te": "ఆంగికం (శరీర కదలికలు), వాచికం (గానం/వాక్కు), ఆహార్యం (వేషధారణ/ఆభరణాలు), సాత్త్వికం (అంతరంగ భావ ప్రకటన).",
                "shloka": "ఆంగికం భువనం యస్య వాచికం సర్వవాఙ్మయమ్ | ఆహార్యం చంద్రతారాది తం నుమః సాత్త్వికం శివమ్ ||",
                "meaning_en": "Whose bodily movement is the universe, whose speech is all learning, whose ornaments are the moon and stars—to that pure Sattvika Shiva we bow.",
                "meaning_te": "సమస్త విశ్వం ఆంగికము, సమస్త వాఙ్మయం వాచికము, చంద్రతారలు ఆహార్యముగా కల సాత్విక శివునికి నమస్కరిస్తున్నాము."
            },
            {
                "lesson_order": 4,
                "name_en": "Navarasas (Nine Aesthetic Sentiments)",
                "name_te": "నవరసాలు మరియు భావాభినయం",
                "sanskrit_name": "Navarasa Nirupanam",
                "category": "Rasa",
                "description_en": "The nine classical sentiments: Shringara (Love), Hasya (Comic/Laughter), Karuna (Pathos/Compassion), Raudra (Anger/Fury), Vira (Heroism/Courage), Bhayanaka (Fear/Terror), Bibhatsa (Disgust/Aversion), Adbhuta (Wonder/Amazement), and Shanta (Peace/Tranquility).",
                "description_te": "శృంగార, హాస్య, కరుణ, రౌద్ర, వీర, భయానక, బీభత్స, అద్భుత మరియు శాంత రసాల సమగ్ర విశ్లేషణ.",
                "shloka": "శృంగారహాస్యకరుణరౌద్రవీరభయానకాః | బీభత్సాద్భుతశాంతాశ్చ నవ నాట్యే రసాః స్మృతాః ||",
                "meaning_en": "Shringara, Hasya, Karuna, Raudra, Vira, Bhayanaka, Bibhatsa, Adbhuta, and Shanta are the nine Rasas declared in drama.",
                "meaning_te": "నాట్యంలో శృంగారము, హాస్యము, కరుణ, రౌద్రము, వీరము, భయానకము, బీభత్సము, అద్భుతము, శాంతము అను తొమ్మిది రసాలు ముఖ్యమైనవి."
            }
        ]
        for l in m1_lessons:
            db.add(Lesson(module_id=m1.id, **l))
        db.commit()

        # =====================================================================
        # MODULE 2: Asamyukta Hasthas (28 Single Hand Gestures)
        # =====================================================================
        m2 = Module(
            module_number=2,
            title_en="Asamyukta Hasthas",
            title_te="అసంయుక్త హస్తాలు (28 ఏకహస్త ముద్రలు)",
            sanskrit_title="Asamyukta Hasta Prakaranam",
            source_pdf="Asamyutha_Hasthas.pdf",
            total_classes=28,
            description_en="The complete canon of 28 single-hand mudras from Abhinaya Darpana with shlokas, hand structures, and usages.",
            description_te="అభినయ దర్పణంలో పేర్కొన్న 28 ఒంటిచేతి ముద్రలు, వాటి శ్లోకాలు, చేతి అమరిక మరియు వినియోగాలు."
        )
        db.add(m2)
        db.commit()
        db.refresh(m2)

        asamyukta_names = [
            ("Pathaka", "పతాక", "Flag", "When the thumb is bent to touch the base of the forefinger, and other fingers are held closely together and stretched straight.", "Asamyutha_Hasthas_p1_0.jpg"),
            ("Tripathaka", "త్రిపతాక", "Three parts of flag", "When the ring finger of Pathaka hastha is bent inward while remaining fingers remain straight.", "Asamyutha_Hasthas_p1_1.jpg"),
            ("Ardhapathaka", "అర్ధపతాక", "Half-flag", "When both the ring finger and little finger of Pathaka are bent inward.", "Asamyutha_Hasthas_p1_2.jpg"),
            ("Kartarimukha", "కర్తరీముఖ", "Scissors face", "When the forefinger and little finger of Tripathaka are extended back or spread like scissors.", "Asamyutha_Hasthas_p1_3.jpg"),
            ("Mayura", "మయూర", "Peacock", "In Tripathaka hastha, the ring finger and thumb join tips together while the other fingers are outstretched.", "Asamyutha_Hasthas_p1_4.jpg"),
            ("Ardhachandra", "అర్ధచంద్ర", "Half-moon", "When the thumb of Pathaka hastha is stretched outward away from the other fingers to form a crescent.", "Asamyutha_Hasthas_p1_5.jpg"),
            ("Arala", "అరాళ", "Bent / Curved", "When the index finger is bent curved while other fingers are held upright like Pathaka.", "Asamyutha_Hasthas_p1_6.jpg"),
            ("Shukathunda", "శుకతుండ", "Parrot beak", "When the ring finger of Arala hastha is also bent down, resembling a parrot's beak.", "Asamyutha_Hasthas_p1_7.jpg"),
            ("Mushti", "ముష్టి", "Fist", "When four fingers are firmly bent into the palm and the thumb is placed securely over them.", "Asamyutha_Hasthas_p1_8.jpg"),
            ("Shikhara", "శిఖర", "Spire / Peak", "From Mushti hastha, when the thumb is raised vertically upright.", "Asamyutha_Hasthas_p1_9.jpg"),
            ("Kapittha", "కపిత్థ", "Wood-apple", "When the forefinger of Shikhara hastha is bent over the tip of the thumb.", "Asamyutha_Hasthas_p2_0.jpg"),
            ("Katakamukha", "కటకాముఖ", "Opening of bracelet", "When the index finger, middle finger and thumb join tips together while ring and little fingers are raised.", "Asamyutha_Hasthas_p2_1.jpg"),
            ("Suchi", "సూచి", "Needle", "When the middle, ring, and little fingers touch the thumb and the index finger is pointed straight upward.", "Asamyutha_Hasthas_p2_2.jpg"),
            ("Chandrakala", "చంద్రకళ", "Crescent digit of moon", "From Suchi hastha, releasing the thumb and extending it outward at an angle.", "Asamyutha_Hasthas_p2_3.jpg"),
            ("Padmakosha", "పద్మకోశ", "Lotus bud", "When all fingers including thumb are curved inward like the petals of a blossoming flower bud.", "Asamyutha_Hasthas_p2_4.jpg"),
            ("Sarpashirsha", "సర్పశీర్ష", "Serpent hood", "When the fingers of Pathaka are slightly cupped inward like a cobra's hood.", "Asamyutha_Hasthas_p3_0.jpg"),
            ("Mrugashirsha", "మృగశీర్ష", "Deer head", "When the thumb and little finger are extended upward while index, middle and ring fingers are bent forward.", "Asamyutha_Hasthas_p3_1.jpg"),
            ("Simhamukha", "సింహముఖ", "Lion face", "When the middle and ring fingers touch the tip of the thumb while index and little fingers are raised upright.", "Asamyutha_Hasthas_p3_2.jpg"),
            ("Kangula", "కాంగూల", "Tail / Bell", "When the ring finger is bent into the palm while the remaining fingers are curved outwards.", "Asamyutha_Hasthas_p3_3.jpg"),
            ("Alapadma", "అలపద్మ", "Full-bloomed lotus", "When all fingers are separated and curved gracefully starting from the little finger like a full-blown lotus.", "Asamyutha_Hasthas_p4_0.jpg"),
            ("Chatura", "చతుర", "Four-sided / Clever", "When the thumb is placed at the base of the middle finger while other three fingers are held together straight.", "Asamyutha_Hasthas_p4_1.jpg"),
            ("Bhramara", "భ్రమర", "Bee", "When the index finger is curled inside while thumb and middle finger touch tips.", "Asamyutha_Hasthas_p4_2.jpg"),
            ("Hamsasya", "హంసాస్య", "Swan beak", "When the forefinger, middle finger and thumb touch tips together gracefully.", "Asamyutha_Hasthas_p5_0.jpg"),
            ("Hamsapaksha", "హంసపక్ష", "Swan wing", "When the little finger is held straight up while the other three fingers are extended sideways.", "Asamyutha_Hasthas_p5_1.jpg"),
            ("Samdamsha", "సందంశ", "Pincers / Grasp", "When the fingers of Padmakosha are opened and closed repeatedly against the thumb.", "Asamyutha_Hasthas_p6_0.jpg"),
            ("Mukula", "ముకుల", "Flower bud", "When all five fingertips are brought together touching at a single point.", "Asamyutha_Hasthas_p6_1.jpg"),
            ("Tamrachuda", "తామ్రచూడ", "Rooster crest", "When the index finger of Mukula is curved hooked outward like a rooster's comb.", "Asamyutha_Hasthas_p7_0.jpg"),
            ("Trishula", "త్రిశూల", "Trident", "When the thumb and little finger meet at their tips while index, middle, and ring fingers are held straight upright.", "Asamyutha_Hasthas_p7_1.jpg")
        ]

        shloka_m2 = "పతాకః త్రిపతాకోఽర్థపతాకః కర్తరీముఖః | మయూరాఖ్యోఽర్థచంద్రశ్చ హ్యరాళః శుకతుండకః || ముష్టిశ్చ శిఖరాఖ్యశ్చ కపిత్థః కటకాముఖః | సూచీ చంద్రకళా పద్మకోశః సర్పశిరస్తథా || మృగశీర్షః సింహముఖః కాంగూలశ్చాలపద్మకః | చతురో భ్రమరశ్చైవ హంసాస్యో హంసపక్షకః || సందంశో ముకులశ్చైవ తామ్రచూడస్త్రిశూలకః | ఇత్యసంయుతహస్తానామష్టావింశతిరీరితా ||"

        for idx, (name_en, name_te, trans_en, desc_en, img) in enumerate(asamyukta_names):
            db.add(Lesson(
                module_id=m2.id,
                lesson_order=idx + 1,
                name_en=name_en,
                name_te=name_te,
                sanskrit_name=f"{name_en} Hasta",
                category="Asamyukta",
                description_en=f"{name_en} ({trans_en}): {desc_en}",
                description_te=f"{name_te} హస్తము ({trans_en}): {desc_en}",
                shloka=shloka_m2,
                image_url=f"/assets/curriculum/{img}"
            ))
        db.commit()

        # =====================================================================
        # MODULE 3: Samyukta Hasthas (24 Combined Hand Gestures)
        # =====================================================================
        m3 = Module(
            module_number=3,
            title_en="Samyukta Hasthas",
            title_te="సంయుక్త హస్తాలు (24 ద్విహస్త ముద్రలు)",
            sanskrit_title="Samyukta Hasta Prakaranam",
            source_pdf="Samyutha Hasthas.pdf",
            total_classes=24,
            description_en="The 24 combined two-hand gestures from Abhinaya Darpana requiring coordinate alignment of both hands.",
            description_te="రెండు చేతుల కలయికతో రూపొందే 24 సంయుక్త హస్తాలు మరియు వాటి నాట్య వినియోగాలు."
        )
        db.add(m3)
        db.commit()
        db.refresh(m3)

        samyukta_names = [
            ("Anjali", "అంజలి", "Salutation", "When two Pataka hands are joined palm to palm in front of chest, forehead or head.", "Samyutha_Hasthas_p1_0.jpg"),
            ("Kapotha", "కపోత", "Dove", "When the palms of Anjali touch only at the tips, bases, and edges leaving hollow center.", "Samyutha_Hasthas_p1_1.jpg"),
            ("Karkata", "కర్కట", "Crab", "When the fingers of both hands are interlocked and turned inwards or outwards.", "Samyutha_Hasthas_p2_0.jpg"),
            ("Swastika", "స్వస్తిక", "Crossed", "When two Pataka hands are held crossed at the wrists.", "Samyutha_Hasthas_p2_1.jpg"),
            ("Dola", "డోలా", "Swing", "When two Pataka hands hang loosely at the respective sides of the thighs.", "Samyutha_Hasthas_p2_2.jpg"),
            ("Pushpaputa", "పుష్పపుట", "Flower casket", "When two Sarpashirsha hands are joined side-by-side touching to hold flowers.", "Samyutha_Hasthas_p3_0.jpg"),
            ("Uthsanga", "ఉత్సంగ", "Embrace", "When Arala hands are placed crossed on opposite shoulders.", "Samyutha_Hasthas_p3_1.jpg"),
            ("Shivalinga", "శివలింగ", "Lord Shiva phallus", "When Shikhara hand is placed upright upon the palm of left Ardhachandra hand.", "Samyutha_Hasthas_p3_2.jpg"),
            ("Katakavardhana", "కటకావర్ధన", "Link of bracelets", "When two Katakamukha hands are held crossed at the wrists.", "Samyutha_Hasthas_p4_0.jpg"),
            ("Kartari-Swastika", "కర్తరీస్వస్తిక", "Crossed scissors", "When two Kartarimukha hands are held crossed at the wrists.", "Samyutha_Hasthas_p4_1.jpg"),
            ("Sakata", "శకట", "Cart", "When Bhramara hands are held with thumb and middle finger extended.", "Samyutha_Hasthas_p4_2.jpg"),
            ("Shankha", "శంఖ", "Conch", "When the thumb of left Shikhara is enclosed by four fingers of right hand while right thumb touches left index.", "Samyutha_Hasthas_p5_0.jpg"),
            ("Chakra", "చక్ర", "Discus", "When Ardhachandra hands are placed across each other palm-to-palm at right angles.", "Samyutha_Hasthas_p5_1.jpg"),
            ("Samputa", "సంపుట", "Box / Casket", "When the fingers of Chakra hands are curved inward forming an enclosed container.", "Samyutha_Hasthas_p5_2.jpg"),
            ("Pasha", "పాశ", "Noose", "When the index fingers of Suchi hands are interlocked like a chain link.", "Samyutha_Hasthas_p6_0.jpg"),
            ("Kilaka", "కీలక", "Bond / Intimacy", "When the little fingers of Mrigashirsha hands are interlinked.", "Samyutha_Hasthas_p6_1.jpg"),
            ("Matsya", "మత్స్య", "Fish", "When two Pataka hands are placed one over the other with thumbs extended sideways like fins.", "Samyutha_Hasthas_p6_2.jpg"),
            ("Kurma", "కూర్మ", "Tortoise", "When Chakra hands have fingers curled inward with thumb and little finger emerging.", "Samyutha_Hasthas_p7_0.jpg"),
            ("Varaha", "వరాహ", "Boar", "When Mrigashirsha hands are held one on top of the other, thumbs and little fingers touching.", "Samyutha_Hasthas_p7_1.jpg"),
            ("Garuda", "గరుడ", "Eagle / Kite", "When Ardhachandra hands are hooked across each other by the thumbs, fluttering like wings.", "Samyutha_Hasthas_p7_2.jpg"),
            ("Nagabandha", "నాగబంధ", "Serpent coil", "When Sarpashirsha hands are crossed at the wrists and interlocked.", "Samyutha_Hasthas_p8_0.jpg"),
            ("Khatva", "ఖట్వా", "Cot / Bed", "When the thumbs and index fingers of Chatushra hands are placed facing downwards.", "Samyutha_Hasthas_p8_1.jpg"),
            ("Bherunda", "భేరుండ", "Double-headed bird", "When Kapittha hands are joined crossed at the wrists.", "Samyutha_Hasthas_p8_2.jpg"),
            ("Avahittha", "అవహిత్థ", "Dissimulation / Secret", "When two Alapadma hands are held crossed over the chest.", "Samyutha_Hasthas_p9_0.jpg")
        ]

        shloka_m3 = "అంజలిశ్చ కపోతశ్చ కర్కటః స్వస్తికస్తథా | డోలాహస్తః పుష్పపుట ఉత్సంగః శివలింగకః || కటకావర్ధనశ్చైవ కర్తరీస్వస్తికస్తథా | శకటః శంఖచక్రౌ చ సంపుటః పాశకీలకౌ || మత్స్యః కూర్మో వరాహశ్చ గరుడో నాగబంధకః | ఖట్వా భేరుండకశ్చైవ తథావహిత్థకస్తథా || చతుర్వింశతిసంఖ్యాకాః సంయుతాః కథితాః కరాః ||"

        for idx, (name_en, name_te, trans_en, desc_en, img) in enumerate(samyukta_names):
            db.add(Lesson(
                module_id=m3.id,
                lesson_order=idx + 1,
                name_en=name_en,
                name_te=name_te,
                sanskrit_name=f"{name_en} Samyukta Hasta",
                category="Samyukta",
                description_en=f"{name_en} ({trans_en}): {desc_en}",
                description_te=f"{name_te} సంయుక్త హస్తము ({trans_en}): {desc_en}",
                shloka=shloka_m3,
                image_url=f"/assets/curriculum/{img}"
            ))
        db.commit()

        # =====================================================================
        # MODULE 4: Shiro Bhedhas (9 Head Movements)
        # =====================================================================
        m4 = Module(
            module_number=4,
            title_en="Shiro Bhedhas",
            title_te="శిరో భేదాలు (9 శిరస్సు కదలికలు)",
            sanskrit_title="Shiro Bheda Prakaranam",
            source_pdf="Shiro Bhedhas.pdf",
            total_classes=9,
            description_en="The 9 canonical head movements from Natyashastra and Abhinaya Darpana expressing distinct emotional states.",
            description_te="నాట్యశాస్త్రం మరియు అభినయ దర్పణంలో చెప్పబడిన 9 శిరో భేదాలు మరియు వాటి భావ ప్రకటనలు."
        )
        db.add(m4)
        db.commit()
        db.refresh(m4)

        shiro_names = [
            ("Sama", "సమ", "Level / Still", "Head is held motionless in neutral erect center posture. Used in beginning of dance, prayer, and calm reflection.", "Shiro_Bhedhas_p2_0.jpg"),
            ("Udvahitam", "ఉద్వాహితమ్", "Raised up", "Head is lifted upward looking toward the sky. Used to depict the moon, flag, sky, divine temples.", "Shiro_Bhedhas_p2_1.jpg"),
            ("Adhomukham", "అధోముఖమ్", "Cast down", "Face and head are bowed downward. Used to portray modesty, sorrow, shame, bowing, and looking at the earth.", "Shiro_Bhedhas_p2_2.jpg"),
            ("Alolitam", "ఆలోలితమ్", "Rolling circular", "Head moves gently in a circular rolling motion. Used in dizziness, sleepiness, possession, uncontrollable laughter.", "Shiro_Bhedhas_p2_3.jpg"),
            ("Dhutam", "ధూతమ్", "Side-to-side shake", "Head turns horizontally from left to right and right to left. Used to show denial, rejection, looking sideways.", "Shiro_Bhedhas_p3_0.jpg"),
            ("Kampitam", "కంపితమ్", "Nodding up-down", "Head moves up and down vertically in quick nods. Used in inquiry, beckoning, affirmation, and threatening.", "Shiro_Bhedhas_p3_1.jpg"),
            ("Paravrittam", "పరావృత్తమ్", "Turned aside", "Head is briskly turned sharply to one side. Used in turning away in anger, modesty, hair dressing, aversion.", "Shiro_Bhedhas_p3_2.jpg"),
            ("Ukshiptam", "ఉత్క్షిప్తమ్", "Tilted upward", "Head is turned sideways and tilted upwards. Used in indicating 'take this', cherishing, pride, and divine welcome.", "Shiro_Bhedhas_p3_3.jpg"),
            ("Parivahitam", "పరివాహితమ్", "Gently swinging", "Head rolls smoothly side to side like a gentle fan. Used in joyous appreciation, love longing, and ecstasy.", "Shiro_Bhedhas_p3_4.jpg")
        ]

        shloka_m4 = "సమం ఉద్వాహితం చాధోముఖం ఆలోలితం ధుతమ్ | కంపితం చ పరావృత్తం ఉత్క్షిప్తం పరివాహితమ్ || నవధా కథితం శీర్షం నాట్యశాస్త్రవిశారదైః ||"

        for idx, (name_en, name_te, trans_en, desc_en, img) in enumerate(shiro_names):
            db.add(Lesson(
                module_id=m4.id,
                lesson_order=idx + 1,
                name_en=name_en,
                name_te=name_te,
                sanskrit_name=f"{name_en} Shiro Bheda",
                category="Shiro",
                description_en=f"{name_en} ({trans_en}): {desc_en}",
                description_te=f"{name_te} శిరోభేదము ({trans_en}): {desc_en}",
                shloka=shloka_m4,
                image_url=f"/assets/curriculum/{img}"
            ))
        db.commit()

        # =====================================================================
        # MODULE 5: Drishti Bhedhas (8 Eye Glance Movements)
        # =====================================================================
        m5 = Module(
            module_number=5,
            title_en="Drishti Bhedhas",
            title_te="దృష్టి భేదాలు (8 నేత్ర కదలికలు)",
            sanskrit_title="Drishti Bheda Prakaranam",
            source_pdf="Drishti Bhedhas.pdf",
            total_classes=8,
            description_en="The 8 canonical glances of the eyes fundamental to classical Sattvika Abhinaya and emotional expression.",
            description_te="సాత్త్వికాభినయానికి మూలమైన 8 నేత్ర దృష్టి భేదాలు, వాటి వీక్షణ పద్ధతులు మరియు భావాభినయాలు."
        )
        db.add(m5)
        db.commit()
        db.refresh(m5)

        drishti_names = [
            ("Sama", "సమ", "Level glance", "Eyes kept straight, open, still and tranquil without blinking. Used at the opening of dance, deities, and neutral thought."),
            ("Alokita", "ఆలోకిత", "Rolling circular", "Rolling the eyeballs briskly in a wide circular motion. Used in looking at a potter's wheel, all directions, or begging."),
            ("Sachi", "సాచి", "Corner glance", "Looking obliquely through the corner of the eyes without moving the face. Used in shooting arrows, remembrance, secret love."),
            ("Pralokita", "ప్రలోకిత", "Side-to-side scan", "Eyes moving swiftly from one side to the other. Used to look at objects on both sides, balance, and scanning audience."),
            ("Nimilita", "నిమీలిత", "Half-closed gaze", "Eyes half shut, gazing downward toward the heart. Used in meditation, prayer, submission, and inner trance."),
            ("Ullokita", "ఉల్లోకిత", "Upward glance", "Gazing upward directly toward the heights. Used to behold tall trees, the sky, flying birds, and celestial beings."),
            ("Anuvritta", "అనువృత్త", "Rapid up-down scan", "Rapid movement of the eyes up and down in quick succession. Used in anger, surveying someone head to toe, and agitation."),
            ("Avalokita", "అవలోకిత", "Downward glance", "Looking steadily downward towards the ground. Used in looking at one's shadow, modesty, fatigue, and observing the earth.")
        ]

        shloka_m5 = "సమం ఆలోకితం సాచీ ప్రలోకిత నిమీలితే | ఉల్లోకితానువృత్తే చ తథా చైవావలోకితమ్ || ఇత్యష్టౌ దృష్టిభేదాః స్యుః కీర్తితాః పూర్వసూరిభిః ||"

        for idx, (name_en, name_te, trans_en, desc_en) in enumerate(drishti_names):
            db.add(Lesson(
                module_id=m5.id,
                lesson_order=idx + 1,
                name_en=name_en,
                name_te=name_te,
                sanskrit_name=f"{name_en} Drishti Bheda",
                category="Drishti",
                description_en=f"{name_en} ({trans_en}): {desc_en}",
                description_te=f"{name_te} దృష్టి భేదము ({trans_en}): {desc_en}",
                shloka=shloka_m5,
                image_url=None
            ))
        db.commit()

        # =====================================================================
        # MODULE 6: Dvadasha Hasta Pranamas (12 Salutation Gestures)
        # =====================================================================
        m6 = Module(
            module_number=6,
            title_en="Dvadasha Hasta Pranamas",
            title_te="ద్వాదశ హస్త ప్రణామాలు (12 నమస్కార ముద్రలు)",
            sanskrit_title="Dvadasha Hasta Pranama Prakaranam",
            source_pdf="Dvadasha Hasta Pranamas.pdf",
            total_classes=12,
            description_en="12 traditional classical hand salutation actions used in invocation, guru-vandana, and sabha pranamam.",
            description_te="దైవము, గురువు మరియు సభకు భక్తిపూర్వకంగా సమర్పించే 12 శాస్త్రీయ హస్త ప్రణామాలు."
        )
        db.add(m6)
        db.commit()
        db.refresh(m6)

        pranama_names = [
            ("Prasaranam", "ప్రసారణం", "Stretching the fingers outward gracefully in salutation."),
            ("Kunchitam", "కుంచితం", "Bending or curving the fingers gently inward toward the palm."),
            ("Rechitam", "రేచితం", "Raising the fingertips gently upward in a delicate lifting motion."),
            ("Pumkhitam", "పుంఖితం", "Gently vibrating or spreading the fingers in rhythmic flourish."),
            ("Apaveshtitam", "అపవేష్టితం", "Turning and bending the fingers of the hand downward."),
            ("Preritam", "ప్రేరితం", "Stretching and moving the fingers toward the back in offering."),
            ("Udveshtitam", "ఉద్వేష్టితం", "Raising the hands upward during dance with open offering."),
            ("Vyavrittam", "వ్యవృత్తం", "Lifting the hand sideways upward in lateral curve."),
            ("Parivritam", "పరివృతం", "Bringing the hands forward smoothly from the side regions."),
            ("Sanketam", "సంకేతం", "Gesturing and signaling meaning directly without the use of words."),
            ("Chihnam", "చిహ్నం", "Representing living and non-living objects through clear symbolic mudras."),
            ("Padartha Tika", "పదార్థ టీక", "Direct visual commentary and interpretation of song lyrics through hand mudras.")
        ]

        shloka_m6 = "ప్రసారణం కుంచితం చ రేచితం పుంఖితం తథా | అపవేష్టితకం చాపి ప్రేరితోద్వేష్టితౌ తథా || వ్యవృత్తః పరివృత్తశ్చ సంకేతస్తదనంతరమ్ | చిహ్నం పదార్థటీకేతి ప్రాణా ద్వాదశ హస్తజాః ||"

        for idx, (name_en, name_te, desc_en) in enumerate(pranama_names):
            db.add(Lesson(
                module_id=m6.id,
                lesson_order=idx + 1,
                name_en=name_en,
                name_te=name_te,
                sanskrit_name=f"{name_en} Pranama",
                category="Pranama",
                description_en=f"{name_en}: {desc_en}",
                description_te=f"{name_te}: {desc_en}",
                shloka=shloka_m6,
                image_url=None
            ))
        db.commit()

        # =====================================================================
        # MODULE 7: Sapta Talamulu (7 Foundational Talas and 35 System)
        # =====================================================================
        m7 = Module(
            module_number=7,
            title_en="Sapta Talamulu",
            title_te="సప్త తాళములు (7 ప్రధాన తాళాలు మరియు 35 తాళ వ్యవస్థ)",
            sanskrit_title="Sapta Tala Prakaranam",
            source_pdf="Sapta Talamulu.pdf",
            total_classes=7,
            description_en="The 7 foundational rhythmic cycles (Dhruva, Matya, Rupaka, Jhampe, Triputa, Ata, Eka), their mythological creators (Kartalu), components (Angas), and the 35 Tala Matrix.",
            description_te="కూచిపూడి నృత్యానికి ప్రాణమైన 7 ప్రధాన తాళాలు, వాటి కర్తలు, అంగములు (లఘు, ధ్రుత, అనుధ్రుత) మరియు 35 తాళాల సమగ్ర పట్టిక."
        )
        db.add(m7)
        db.commit()
        db.refresh(m7)

        tala_info = [
            ("Dhruva Tala", "ధ్రువ తాళము", "1 0 1 1", "Laghu, Dhrutam, Laghu, Laghu", "Created by Bharatacharya, Dattila, and Anjaneya."),
            ("Matya Tala", "మత్య తాళము", "1 0 1", "Laghu, Dhrutam, Laghu", "Created by Matanga Maharshi, Arjuna, and Kohalacharyudu."),
            ("Rupaka Tala", "రూపక తాళము", "0 1", "Dhrutam, Laghu", "Created by Anjaneya, Kumbhoja (Sage Agastya), and Yajnavalkya."),
            ("Jhampe Tala", "ఝంపె తాళము", "1 U 0", "Laghu, Anudhrutam, Dhrutam", "Created by Nandikeshwara, Bharata, and Sanaka."),
            ("Triputa Tala", "త్రిపుట తాళము", "1 0 0", "Laghu, Dhrutam, Dhrutam", "Created by Narada, Guha (Kartikeya), and Sanatkumara."),
            ("Ata Tala", "అట తాళము", "1 1 0 0", "Laghu, Laghu, Dhrutam, Dhrutam", "Created by Brahma, Vishvavasu, and Tumburu."),
            ("Eka Tala", "ఏక తాళము", "1", "Laghu", "Created by Shankara (Lord Shiva) and Vashistha Maharshi.")
        ]

        shloka_m7 = "ధ్రువో మత్యో రూపకశ్చ ఝంపే త్రిపుట ఏవ చ | అటతాళ ఏకతాళశ్చ సప్త తాళాః ప్రకీర్తితాః ||"

        for idx, (name_en, name_te, pattern, angas, creators) in enumerate(tala_info):
            db.add(Lesson(
                module_id=m7.id,
                lesson_order=idx + 1,
                name_en=name_en,
                name_te=name_te,
                sanskrit_name=f"{name_en}",
                category="Tala",
                description_en=f"{name_en} (Pattern: {pattern}). Angas: {angas}. Kartalu: {creators}. Across the 5 Jaatis (Tisra=3, Chaturasra=4, Khanda=5, Misra=7, Sankeerna=9), this generates 5 distinct variations.",
                description_te=f"{name_te} (లయ సూత్రం: {pattern}). అంగములు: {angas}. కర్తలు: {creators}. పంచ జాతుల (తిస్ర, చతురస్ర, ఖండ, మిశ్ర, సంకీర్ణ) కలయికతో రూపొందే తాళ వైవిధ్యం.",
                shloka=shloka_m7,
                image_url=None
            ))
        db.commit()
        db.commit()

        total_modules = db.query(Module).count()
        total_lessons = db.query(Lesson).count()
        print(f"\n[SUCCESS] Seeded {total_modules} modules and {total_lessons} lessons into SQLite database!")

    finally:
        db.close()

if __name__ == "__main__":
    seed_authentic_curriculum()
