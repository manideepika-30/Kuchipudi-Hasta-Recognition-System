"""
Authentic Quiz and Assessment Seeder for Kuchipudi Platform.
Strictly sourced from Anandamayi Nrityalaya canonical texts and curriculum.
"""
import sys
import os
import json

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from app.database.session import SessionLocal, engine
from app.database.base import Base
from app.models.quiz_practice import Quiz, Question, QuizResult

def seed_quizzes_data(db=None):
    close_at_end = False
    if db is None:
        Base.metadata.create_all(bind=engine)
        db = SessionLocal()
        close_at_end = True

    try:
        # Check if quizzes already seeded
        existing_count = db.query(Quiz).count()
        if existing_count >= 3:
            print(f"[INFO] Quizzes already seeded ({existing_count} quizzes exist). Skipping.")
            return

        # Clear existing quizzes & questions to allow clean reseed
        db.query(Question).delete()
        db.query(QuizResult).delete()
        db.query(Quiz).delete()
        db.commit()

        # =====================================================================
        # QUIZ 1: TYPE A — THEORY MCQ EXAMINATION
        # =====================================================================
        q1 = Quiz(
            module_id=1,
            quiz_type="theory_mcq",
            title_en="Comprehensive Kuchipudi Classical Theory Examination",
            title_te="కూచిపూడి సిద్ధాంత సమగ్ర మూల్యాంకనం (థియరీ పరీక్ష)",
            description_en="A rigorous assessment covering Kuchipudi origins, Siddhendra Yogi, Chaturvidha Abhinaya, Navarasas, and Sapta Talamulu.",
            description_te="కుచేలపురం మూలాలు, సిద్ధేంద్ర యోగి నాట్య సంప్రదాయం, చతుర్విధ అభినయాలు, నవరసాలు మరియు సప్త తాళములపై సమగ్ర పరీక్ష.",
            time_limit_minutes=20,
            passing_score_pct=70.0
        )
        db.add(q1)
        db.commit()
        db.refresh(q1)

        type_a_questions = [
            {
                "order_num": 1,
                "question_text_en": "In which village of Andhra Pradesh did classical Kuchipudi dance originate?",
                "question_text_te": "శాస్త్రీయ కూచిపూడి నాట్యం ఆంధ్రప్రదేశ్‌లోని ఏ గ్రామంలో ఆవిర్భవించింది?",
                "option_a_en": "Kalamkari Puram, Guntur",
                "option_a_te": "కలంకారీ పురం, గుంటూరు",
                "option_b_en": "Kuchelapuram (Kuchipudi), Krishna District",
                "option_b_te": "కుచేలపురం (కూచిపూడి గ్రామం), కృష్ణా జిల్లా",
                "option_c_en": "Lepakshi, Anantapur",
                "option_c_te": "లేపాక్షి, అనంతపురం",
                "option_d_en": "Amaravathi, Palnadu",
                "option_d_te": "అమరావతి, పల్నాడు",
                "correct_option": "B",
                "explanation_en": "Kuchipudi originated in the village of Kuchelapuram (now known as Kuchipudi) in Krishna District, Andhra Pradesh.",
                "explanation_te": "కూచిపూడి నృత్యం ఆంధ్రప్రదేశ్‌లోని కృష్ణా జిల్లా కుచేలపురం (ప్రస్తుతం కూచిపూడి) గ్రామంలో జన్మించింది."
            },
            {
                "order_num": 2,
                "question_text_en": "Who is revered as the patron saint and formalizer of the classical Kuchipudi dance drama tradition?",
                "question_text_te": "కూచిపూడి నృత్య నాటక సంప్రదాయాన్ని క్రమబద్ధీకరించిన ఆద్యుడు మరియు గురువు ఎవరు?",
                "option_a_en": "Saint Siddhendra Yogi",
                "option_a_te": "సిద్ధేంద్ర యోగి",
                "option_b_en": "Saint Tyagaraja",
                "option_b_te": "త్యాగరాజు",
                "option_c_en": "Purandara Dasa",
                "option_c_te": "పురందర దాసు",
                "option_d_en": "Annamacharya",
                "option_d_te": "అన్నమాచార్య",
                "correct_option": "A",
                "explanation_en": "Saint Siddhendra Yogi formalized the Yakshagana style into classical Kuchipudi Natya and authored the foundational masterpiece 'Bhama Kalapam'.",
                "explanation_te": "సిద్ధేంద్ర యోగి కూచిపూడి నృత్య నాటకాన్ని రూపొందించి, అమర కావ్యం 'భామా కలాపం' రచించారు."
            },
            {
                "order_num": 3,
                "question_text_en": "Which foundational Sanskrit shloka describes the fundamental relationship between hand gesture, gaze, mind, and aesthetic bliss (Rasa)?",
                "question_text_te": "హస్తం, దృష్టి, మనస్సు మరియు రససిద్ధి మధ్య సంబంధాన్ని వివరించే ప్రముఖ శ్లోకం ఏది?",
                "option_a_en": "Yato Hastastato Drishtir...",
                "option_a_te": "యతో హస్తస్తతో దృష్టిర్యతో దృష్టిస్తతో మనః...",
                "option_b_en": "Angikam Bhuvanam Yasya...",
                "option_b_te": "ఆంగికం భువనం యస్య...",
                "option_c_en": "Gurur Brahma Gurur Vishnu...",
                "option_c_te": "గురుర్ బ్రహ్మ గురుర్ విష్ణుః...",
                "option_d_en": "Shringara Hasya Karuna...",
                "option_d_te": "శృంగార హాస్య కరుణ...",
                "correct_option": "A",
                "explanation_en": "'Yato Hastastato Drishtir, Yato Drishtistato Manah, Yato Manastato Bhavo, Yato Bhavastato Rasah' establishes the aesthetic hierarchy of classical Indian dance.",
                "explanation_te": "'యతో హస్తస్తతో దృష్టిః...' శ్లోకం హస్త ముద్ర, నేత్ర దృష్టి, మనస్సు, భావం మరియు రసానుభూతి లయను తెలియజేస్తుంది."
            },
            {
                "order_num": 4,
                "question_text_en": "According to Natyashastra, what is pure, abstract rhythmic dance without storytelling called?",
                "question_text_te": "నాట్యశాస్త్రం ప్రకారం కథ లేదా భావ విన్యాసం లేని కేవల లయబద్ధ కదలికలను ఏమంటారు?",
                "option_a_en": "Natya",
                "option_a_te": "నాట్యం",
                "option_b_en": "Nritya",
                "option_b_te": "నృత్యం",
                "option_c_en": "Nritta",
                "option_c_te": "నృత్తం",
                "option_d_en": "Abhinaya",
                "option_d_te": "అభినయం",
                "correct_option": "C",
                "explanation_en": "Nritta is pure abstract rhythm and footwork. Nritya incorporates emotional expression (Bhava), while Natya is complete dramatic presentation.",
                "explanation_te": "నృత్తం కేవలం లయ-తాళాలతో కూడిన కదలిక; నృత్యంలో భావం ఉంటుంది; నాట్యంలో సంపూర్ణ కథానాటకం ఉంటుంది."
            },
            {
                "order_num": 5,
                "question_text_en": "Which of the Chaturvidha Abhinayas refers to expression through costume, jewelry, and stage makeup?",
                "question_text_te": "చతుర్విధ అభినయాలలో వేషధారణ, ఆభరణాలు మరియు అలంకరణను సూచించే అభినయం ఏది?",
                "option_a_en": "Angika Abhinaya",
                "option_a_te": "ఆంగిక అభినయం",
                "option_b_en": "Vachika Abhinaya",
                "option_b_te": "వాచిక అభినయం",
                "option_c_en": "Aharya Abhinaya",
                "option_c_te": "ఆహార్య అభినయం",
                "option_d_en": "Sattvika Abhinaya",
                "option_d_te": "సాత్త్విక అభినయం",
                "correct_option": "C",
                "explanation_en": "Aharya Abhinaya pertains to costumes, jewelry, facial paint, and decorative stage ornaments.",
                "explanation_te": "ఆహార్య అభినయం అనగా వేషధారణ, కిరీటాలు, నగల అలంకరణ మరియు రంగస్థల సౌందర్యం."
            },
            {
                "order_num": 6,
                "question_text_en": "How many canonical single-hand gestures (Asamyukta Hasthas) are detailed in the Abhinaya Darpana?",
                "question_text_te": "అభినయ దర్పణంలో పేర్కొనబడిన అసంయుక్త హస్తాల (ఏకహస్త ముద్రల) సంఖ్య ఎంత?",
                "option_a_en": "24",
                "option_a_te": "24",
                "option_b_en": "28",
                "option_b_te": "28",
                "option_c_en": "32",
                "option_c_te": "32",
                "option_d_en": "52",
                "option_d_te": "52",
                "correct_option": "B",
                "explanation_en": "The Abhinaya Darpana enumerates exactly 28 Asamyukta Hasthas (starting with Pathaka and concluding with Trishoola).",
                "explanation_te": "అభినయ దర్పణంలో పతాక నుండి త్రిశూల వరకు సరిగ్గా 28 అసంయుక్త హస్తాలు వర్ణించబడ్డాయి."
            },
            {
                "order_num": 7,
                "question_text_en": "How many head movements (Shiro Bhedas) and eye glance movements (Drishti Bhedas) are codified in Natyashastra?",
                "question_text_te": "నాట్యశాస్త్రంలో పేర్కొనబడిన శిరో భేదాలు మరియు దృష్టి భేదాల సంఖ్య ఎంత?",
                "option_a_en": "9 Shiro Bhedas and 8 Drishti Bhedas",
                "option_a_te": "9 శిరో భేదాలు మరియు 8 దృష్టి భేదాలు",
                "option_b_en": "8 Shiro Bhedas and 9 Drishti Bhedas",
                "option_b_te": "8 శిరో భేదాలు మరియు 9 దృష్టి భేదాలు",
                "option_c_en": "7 Shiro Bhedas and 7 Drishti Bhedas",
                "option_c_te": "7 శిరో భేదాలు మరియు 7 దృష్టి భేదాలు",
                "option_d_en": "12 Shiro Bhedas and 10 Drishti Bhedas",
                "option_d_te": "12 శిరో భేదాలు మరియు 10 దృష్టి భేదాలు",
                "correct_option": "A",
                "explanation_en": "Natyashastra codifies 9 Shiro Bhedas (Sama through Parivahitam) and 8 Drishti Bhedas (Sama through Avalokita).",
                "explanation_te": "సమం నుండి పరివాహితం వరకు 9 శిరో భేదాలు, సమం నుండి అవలోకితం వరకు 8 దృష్టి భేదాలు ఉన్నాయి."
            },
            {
                "order_num": 8,
                "question_text_en": "How many Dvadasha Hasta Pranamas (hand salutation actions) are prescribed in the classical tradition?",
                "question_text_te": "శాస్త్రీయ సంప్రదాయంలో ద్వాదశ హస్త ప్రణామాలు ఎన్ని?",
                "option_a_en": "7",
                "option_a_te": "7",
                "option_b_en": "10",
                "option_b_te": "10",
                "option_c_en": "12",
                "option_c_te": "12",
                "option_d_en": "16",
                "option_d_te": "16",
                "correct_option": "C",
                "explanation_en": "Dvadasha means twelve: there are 12 canonical hand salutation gestures from Prasaranam to Padartha Tika.",
                "explanation_te": "ద్వాదశ అనగా 12: ప్రసారణం నుండి పదార్థటీక వరకు 12 శాస్త్రీయ హస్త ప్రణామాలు కలవు."
            },
            {
                "order_num": 9,
                "question_text_en": "In the Sapta Tala system, which tala consists of Dhrutam followed by Laghu (0 1)?",
                "question_text_te": "సప్త తాళ వ్యవస్థలో దృతము మరియు లఘువు (0 1) కలయికతో ఏర్పడే తాళము ఏది?",
                "option_a_en": "Eka Tala",
                "option_a_te": "ఏక తాళము",
                "option_b_en": "Rupaka Tala",
                "option_b_te": "రూపక తాళము",
                "option_c_en": "Triputa Tala",
                "option_c_te": "త్రిపుట తాళము",
                "option_d_en": "Dhruva Tala",
                "option_d_te": "ధ్రువ తాళము",
                "correct_option": "B",
                "explanation_en": "Rupaka Tala consists of 2 Angas: one Dhrutam (0) followed by one Laghu (1).",
                "explanation_te": "రూపక తాళం దృతము మరియు లఘువు (0 1) తో 6 అక్షరాల కాలంతో ఏర్పడుతుంది."
            },
            {
                "order_num": 10,
                "question_text_en": "Which creator/deity is traditionally revered as the Tala Karta of Dhruva Tala?",
                "question_text_te": "ధ్రువ తాళానికి సాంప్రదాయ తాళ కర్తగా ఎవరిని గౌరవిస్తారు?",
                "option_a_en": "Lord Brahma",
                "option_a_te": "బ్రహ్మదేవుడు",
                "option_b_en": "Lord Shiva",
                "option_b_te": "పరమశివుడు",
                "option_c_en": "Sage Narada",
                "option_c_te": "నారద మహర్షి",
                "option_d_en": "Sage Agastya",
                "option_d_te": "అగస్త్య మహర్షి",
                "correct_option": "A",
                "explanation_en": "According to Sapta Talamulu treatise, Dhruva Tala was authored by Bharatacharya, Dattila, and Lord Brahma.",
                "explanation_te": "సప్త తాళ గ్రంథం ప్రకారం ధ్రువ తాళ కర్తలు భరతాచార్య, దత్తిల మరియు బ్రహ్మదేవుడు."
            }
        ]

        for q in type_a_questions:
            db.add(Question(quiz_id=q1.id, **q))
        db.commit()

        # =====================================================================
        # QUIZ 2: TYPE B — GESTURE & SHIRO IMAGE IDENTIFICATION
        # =====================================================================
        q2 = Quiz(
            module_id=2,
            quiz_type="image_identification",
            title_en="Classical Gesture & Head Movement Image Identification",
            title_te="నాట్య ముద్రలు మరియు శిరోభేదాల చిత్ర గుర్తింపు పరీక్ష (ఇమేజ్ ఐడెంటిఫికేషన్)",
            description_en="Inspect authentic demonstration photographs from the Anandamayi Nrityalaya documents and identify the canonical mudra or head movement.",
            description_te="ఆనందమయి నృత్యాలయ చిత్రాలను పరిశీలించి, సరైన హస్త ముద్ర లేదా శిరో భేదాన్ని గుర్తించండి.",
            time_limit_minutes=15,
            passing_score_pct=70.0
        )
        db.add(q2)
        db.commit()
        db.refresh(q2)

        type_b_questions = [
            {
                "order_num": 1,
                "image_url": "/assets/curriculum/Asamyutha_Hasthas_p1_0.jpg",
                "question_text_en": "Identify this foundational single-hand gesture (Asamyukta Hasta):",
                "question_text_te": "చిత్రంలో చూపబడిన ప్రారంభ ఏకహస్త ముద్రను గుర్తించండి:",
                "option_a_en": "Pathaka (Flag)",
                "option_a_te": "పతాక (ధ్వజము)",
                "option_b_en": "Tripathaka (Three parts of flag)",
                "option_b_te": "త్రిపతాక",
                "option_c_en": "Ardhachandra (Half moon)",
                "option_c_te": "అర్ధచంద్ర",
                "option_d_en": "Arala (Bent)",
                "option_d_te": "అరాల",
                "correct_option": "A",
                "explanation_en": "Pathaka is formed when all fingers are extended straight together while the thumb is bent inward against the side of the palm.",
                "explanation_te": "వేళ్లన్నీ నిటారుగా కలిసి ఉండి, బొటనవేలు అరచేయి పక్కకు వంచబడి ఉంటే అది 'పతాక' హస్తం."
            },
            {
                "order_num": 2,
                "image_url": "/assets/curriculum/Asamyutha_Hasthas_p1_1.jpg",
                "question_text_en": "Identify this single-hand gesture where the ring finger is bent downward:",
                "question_text_te": "పతాక హస్తంలో ఉంగరపు వేలు క్రిందికి వంచబడిన ముద్రను గుర్తించండి:",
                "option_a_en": "Mayura (Peacock)",
                "option_a_te": "మయూర",
                "option_b_en": "Kartarimukha (Scissors)",
                "option_b_te": "కర్తరీముఖ",
                "option_c_en": "Tripathaka (Three-fold flag)",
                "option_c_te": "త్రిపతాక",
                "option_d_en": "Sukatunda (Parrot beak)",
                "option_d_te": "శుకతుండ",
                "correct_option": "C",
                "explanation_en": "Tripathaka is formed by bending only the ring finger of the Pathaka hand. Used to denote a crown, tree, thunderbolt, or Indra.",
                "explanation_te": "పతాక హస్తంలో అనామిక (ఉంగరపు వేలు) వంచబడితే అది త్రిపతాక హస్తం అవుతుంది. కిరీటం, చెట్టు, ఇంద్రుని వజ్రాయుధాన్ని సూచిస్తుంది."
            },
            {
                "order_num": 3,
                "image_url": "/assets/curriculum/Asamyutha_Hasthas_p1_4.jpg",
                "question_text_en": "Identify this graceful mudra where the ring fingertip touches the thumb tip:",
                "question_text_te": "ఉంగరపు వేలి కొన బొటనవేలి కొనను తాకుతూ, మిగిలిన వేళ్లు విస్తరించిన ముద్ర ఏది?",
                "option_a_en": "Mayura (Peacock)",
                "option_a_te": "మయూర (నెమలి)",
                "option_b_en": "Alapadma (Bloomed lotus)",
                "option_b_te": "అలపద్మ",
                "option_c_en": "Katakamukha (Opening of link)",
                "option_c_te": "కటకాముఖ",
                "option_d_en": "Kapittha (Wood apple)",
                "option_d_te": "కపిత్థ",
                "correct_option": "A",
                "explanation_en": "Mayura hasta is formed by joining the ring fingertip to the tip of the thumb while the other fingers are extended like peacock feathers.",
                "explanation_te": "బొటనవేలు మరియు అనామిక కొనలు కలిసి ఉండి, ఇతర వేళ్లు నిటారుగా ఉంటే నెమలిని సూచించే 'మయూర' హస్తం."
            },
            {
                "order_num": 4,
                "image_url": "/assets/curriculum/Asamyutha_Hasthas_p2_0.jpg",
                "question_text_en": "Identify this hand gesture where the thumb is raised upright over the closed fist:",
                "question_text_te": "పిడికిలిపై బొటనవేలు పైకి నిటారుగా ఎత్తబడిన ముద్ర ఏది?",
                "option_a_en": "Mushti (Fist)",
                "option_a_te": "ముష్టి",
                "option_b_en": "Shikhara (Peak / Spire)",
                "option_b_te": "శిఖర",
                "option_c_en": "Chandrakala (Crescent moon)",
                "option_c_te": "చంద్రకళ",
                "option_d_en": "Soochi (Needle)",
                "option_d_te": "సూచి",
                "correct_option": "B",
                "explanation_en": "Shikhara is formed when the thumb of a closed Mushti hand is raised upright. Used to depict Lord Shiva, bow, bell, pillar, and firmness.",
                "explanation_te": "ముష్టి హస్తంలో బొటనవేలు నిటారుగా పైకి నిలిపితే అది 'శిఖర' హస్తం. శివుడు, ధనుస్సు, గంట మరియు దృఢత్వాన్ని సూచిస్తుంది."
            },
            {
                "order_num": 5,
                "image_url": "/assets/curriculum/Samyutha_Hasthas_p1_0.jpg",
                "question_text_en": "Identify this primary two-hand combined gesture (Samyukta Hasta):",
                "question_text_te": "రెండు పతాక హస్తాలు జోడించబడిన ఈ ప్రముఖ సంయుక్త హస్తాన్ని గుర్తించండి:",
                "option_a_en": "Kapotha (Dove)",
                "option_a_te": "కపోత",
                "option_b_en": "Anjali (Salutation / Namaskaram)",
                "option_b_te": "అంజలి (నమస్కార ముద్ర)",
                "option_c_en": "Karkata (Crab)",
                "option_c_te": "కర్కట",
                "option_d_en": "Dola (Swing)",
                "option_d_te": "డోలా",
                "correct_option": "B",
                "explanation_en": "Anjali hasta is formed by joining two Pataka hands palm-to-palm in reverence. Used in greeting deities, gurus, and audience.",
                "explanation_te": "రెండు పతాక హస్తాలను అరచేతులు కలిసేలా జోడించితే 'అంజలి' హస్తం ఏర్పడుతుంది. దైవ ప్రార్థన, గురు వందనం కోసం వినియోగిస్తారు."
            },
            {
                "order_num": 6,
                "image_url": "/assets/curriculum/Samyutha_Hasthas_p2_1.jpg",
                "question_text_en": "Identify this two-hand gesture where Pataka hands are crossed at the wrists:",
                "question_text_te": "మణికట్టు వద్ద రెండు పతాక హస్తాలు పరస్పరం దాటిన (క్రాస్ అయిన) సంయుక్త హస్తం ఏది?",
                "option_a_en": "Swastika (Crossed)",
                "option_a_te": "స్వస్తిక",
                "option_b_en": "Katakavardhana (Link)",
                "option_b_te": "కటకావర్ధన",
                "option_c_en": "Kartari-Swastika",
                "option_c_te": "కర్తరీస్వస్తిక",
                "option_d_en": "Nagabandha (Serpent coil)",
                "option_d_te": "నాగబంధ",
                "correct_option": "A",
                "explanation_en": "Swastika hasta is formed by crossing two Pataka hands at the wrists. Used to denote fear, praise, and formal discourse.",
                "explanation_te": "రెండు పతాక హస్తాలు మణికట్టు వద్ద కలిస్తే అది 'స్వస్తిక' హస్తం."
            },
            {
                "order_num": 7,
                "image_url": "/assets/curriculum/Samyutha_Hasthas_p3_2.jpg",
                "question_text_en": "Identify this sacred Samyukta Hasta depicting Lord Shiva's symbol:",
                "question_text_te": "అర్ధచంద్ర హస్తంపై శిఖర హస్తం ఉంచడం ద్వారా ఏర్పడే పవిత్ర సంయుక్త హస్తం ఏది?",
                "option_a_en": "Shivalinga (Lord Shiva Lingam)",
                "option_a_te": "శివలింగ",
                "option_b_en": "Matsya (Fish)",
                "option_b_te": "మత్స్య",
                "option_c_en": "Kurma (Tortoise)",
                "option_c_te": "కూర్మ",
                "option_d_en": "Garuda (Eagle)",
                "option_d_te": "గరుడ",
                "correct_option": "A",
                "explanation_en": "Shivalinga hasta is formed by placing an upright Shikhara hand upon the open palm of an Ardhachandra hand.",
                "explanation_te": "ఎడమచేతి అర్ధచంద్ర హస్తంపై కుడిచేతి శిఖర హస్తం నిలబెడితే పరమశివుని 'శివలింగ' హస్తం అవుతుంది."
            },
            {
                "order_num": 8,
                "image_url": "/assets/curriculum/Shiro_Bhedhas_p2_0.jpg",
                "question_text_en": "Identify this neutral, erect, still head position (Shiro Bheda):",
                "question_text_te": "శిరస్సు ఎటూ వంచకుండా నిటారుగా, నిశ్చలంగా ఉంచే ప్రథమ శిరో భేదాన్ని గుర్తించండి:",
                "option_a_en": "Sama Shiro Bheda",
                "option_a_te": "సమ శిరో భేదము",
                "option_b_en": "Udvahitam",
                "option_b_te": "ఉద్వాహితమ్",
                "option_c_en": "Adhomukham",
                "option_c_te": "అధోముఖమ్",
                "option_d_en": "Alolitam",
                "option_d_te": "ఆలోలితమ్",
                "correct_option": "A",
                "explanation_en": "Sama Shiro Bheda is the motionless, erect, natural posture of the head at the start of dance, prayer, and neutral reflection.",
                "explanation_te": "శిరస్సును స్థిరంగా, నిటారుగా ఉంచడం 'సమ' శిరోభేదం. నాట్య ప్రారంభం, శాంత స్థితిలో వాడతారు."
            },
            {
                "order_num": 9,
                "image_url": "/assets/curriculum/Shiro_Bhedhas_p2_1.jpg",
                "question_text_en": "Identify this head movement where the face and head are lifted upward gazing skyward:",
                "question_text_te": "ముఖాన్ని మరియు శిరస్సును పైకెత్తి ఆకాశం వైపు చూసే శిరో భేదం ఏది?",
                "option_a_en": "Udvahitam Shiro Bheda",
                "option_a_te": "ఉద్వాహితమ్ శిరో భేదము",
                "option_b_en": "Adhomukham",
                "option_b_te": "అధోముఖమ్",
                "option_c_en": "Dhutam",
                "option_c_te": "ధూతమ్",
                "option_d_en": "Kampitam",
                "option_d_te": "కంపితమ్",
                "correct_option": "A",
                "explanation_en": "Udvahitam is lifting the head upward looking at the celestial sky, moon, tall flagstaff, or soaring flight.",
                "explanation_te": "తల పైకెత్తి ఆకాశం, చంద్రుడు, దేవాలయ శిఖరం చూడటాన్ని 'ఉద్వాహితమ్' అంటారు."
            },
            {
                "order_num": 10,
                "image_url": "/assets/curriculum/Shiro_Bhedhas_p2_2.jpg",
                "question_text_en": "Identify this head movement where the head is cast downwards in modesty, sorrow or bowing:",
                "question_text_te": "తల క్రిందికి దించి నేలవైపు చూసే లజ్జ లేదా వినమ్ర శిరో భేదం ఏది?",
                "option_a_en": "Adhomukham Shiro Bheda",
                "option_a_te": "అధోముఖమ్ శిరో భేదము",
                "option_b_en": "Paravrittam",
                "option_b_te": "పరావృత్తమ్",
                "option_c_en": "Utkshiptam",
                "option_c_te": "ఉత్క్షిప్తమ్",
                "option_d_en": "Parivahitam",
                "option_d_te": "పరివాహితమ్",
                "correct_option": "A",
                "explanation_en": "Adhomukham is casting the head and eyes downward. Used to express modesty, shame, bowing in surrender, or observing the earth.",
                "explanation_te": "శిరస్సును క్రిందికి దించితే అది 'అధోముఖమ్'. సిగ్గు, విచారం, నమస్కారం మరియు నేలను పరిశీలించే సందర్భంలో వాడతారు."
            }
        ]

        for q in type_b_questions:
            db.add(Question(quiz_id=q2.id, **q))
        db.commit()

        # =====================================================================
        # QUIZ 3: TYPE C — PRACTICAL CAMERA EXAMINATION PREPARATION
        # =====================================================================
        q3 = Quiz(
            module_id=4,
            quiz_type="practical_camera",
            title_en="Practical Camera Examination: Classical Hasta & Movement Benchmark",
            title_te="ప్రత్యక్ష కెమెరా మూల్యాంకన పరీక్ష (హస్త ముద్ర మరియు కదలికల ప్రాక్టికల్ పరీక్ష)",
            description_en="Interactive webcam assessment module evaluating physical mudra formation, landmark stability, and Shiro/Drishti temporal coordination with AI consensus.",
            description_te="వెబ్‌క్యామ్ ద్వారా ముద్రల ఖచ్చితత్వాన్ని మరియు శిరో-దృష్టి కదలికలను ఏఐ సమన్వయంతో నేరుగా మూల్యాంకనం చేసే ప్రాక్టికల్ పరీక్ష.",
            time_limit_minutes=15,
            passing_score_pct=75.0
        )
        db.add(q3)
        db.commit()
        db.refresh(q3)

        type_c_questions = [
            {
                "order_num": 1,
                "question_text_en": "PRACTICAL TASK 1: Present 'Pathaka Hasta' to the camera. Maintain steady palm facing outward for 3 seconds.",
                "question_text_te": "ప్రాక్టికల్ టాస్క్ 1: కెమెరా ముందు 'పతాక హస్తం' ప్రదర్శించండి. అరచేయి ముందుకి ఉంచి 3 సెకన్లు స్థిరంగా ఉంచండి.",
                "option_a_en": "Ready to perform with Webcam",
                "option_a_te": "వెబ్‌క్యామ్‌తో ప్రదర్శించడానికి సిద్ధంగా ఉన్నాను",
                "option_b_en": "Review Pathaka tutorial first",
                "option_b_te": "ముందుగా పతాక పాఠాన్ని సమీక్షించండి",
                "option_c_en": "Adjust camera lighting",
                "option_c_te": "కెమెరా వెలుతురును సరిచేయండి",
                "option_d_en": "Skip this task",
                "option_d_te": "ఈ విభాగాన్ని దాటవేయండి",
                "correct_option": "A",
                "explanation_en": "AI landmarks measure: all 4 fingers aligned at 180 degrees, thumb adducted at palm margin, palm normal pointing directly at lens.",
                "explanation_te": "ఏఐ హస్త ముద్ర కొలమానం: నాలుగు వేళ్లు నిటారుగా ఉండాలి, బొటనవేలు అరచేయి వద్ద సమలేఖనమై ఉండాలి."
            },
            {
                "order_num": 2,
                "question_text_en": "PRACTICAL TASK 2: Present 'Shikhara Hasta'. Hold closed fist with vertical thumb facing the camera.",
                "question_text_te": "ప్రాక్టికల్ టాస్క్ 2: 'శిఖర హస్తం' ప్రదర్శించండి. నిటారుగా ఉన్న బొటనవేలితో పిడికిలిని కెమెరా ముందు ఉంచండి.",
                "option_a_en": "Ready to perform with Webcam",
                "option_a_te": "వెబ్‌క్యామ్‌తో ప్రదర్శించడానికి సిద్ధంగా ఉన్నాను",
                "option_b_en": "Review Shikhara tutorial first",
                "option_b_te": "ముందుగా శిఖర పాఠాన్ని సమీక్షించండి",
                "option_c_en": "Adjust hand distance",
                "option_c_te": "చేతి దూరాన్ని సరిచేయండి",
                "option_d_en": "Skip this task",
                "option_d_te": "ఈ విభాగాన్ని దాటవేయండి",
                "correct_option": "A",
                "explanation_en": "AI landmarks measure: 4 curled PIP/DIP finger joints and thumb CMC/MCP upright at 90 degrees.",
                "explanation_te": "ఏఐ శిఖర హస్త కొలమానం: నాలుగు వేళ్ల ముడుతలు మరియు నిటారుగా ఉన్న బొటనవేలు పరిశీలించబడతాయి."
            },
            {
                "order_num": 3,
                "question_text_en": "PRACTICAL TASK 3: Perform 'Udvahitam Shiro Bheda' (lifting head upward at +25 degrees) smoothly.",
                "question_text_te": "ప్రాక్టికల్ టాస్క్ 3: 'ఉద్వాహితమ్ శిరో భేదం' (శిరస్సును పైకి ఎత్తడం) కెమెరా ఎదుట లయబద్ధంగా ప్రదర్శించండి.",
                "option_a_en": "Ready to perform with Webcam",
                "option_a_te": "వెబ్‌క్యామ్‌తో ప్రదర్శించడానికి సిద్ధంగా ఉన్నాను",
                "option_b_en": "Review Udvahitam tutorial first",
                "option_b_te": "ముందుగా ఉద్వాహితం పాఠాన్ని సమీక్షించండి",
                "option_c_en": "Adjust face position",
                "option_c_te": "ముఖ స్థానాన్ని సరిచేయండి",
                "option_d_en": "Skip this task",
                "option_d_te": "ఈ విభాగాన్ని దాటవేయండి",
                "correct_option": "A",
                "explanation_en": "AI BiGRU model evaluates pitch kinematic vector across temporal frames from neutral (0 deg) to upward (+25 deg).",
                "explanation_te": "శిరోభేద పిచ్ కోణం మరియు ఫ్రేమ్‌ల స్థిరత్వాన్ని ఏఐ మోడల్ పరిశీలిస్తుంది."
            }
        ]

        for q in type_c_questions:
            db.add(Question(quiz_id=q3.id, **q))
        db.commit()

        print(f"[SUCCESS] Seeded 3 comprehensive quizzes with {len(type_a_questions) + len(type_b_questions) + len(type_c_questions)} authentic questions into SQLite!")

    finally:
        if close_at_end:
            db.close()

if __name__ == "__main__":
    seed_quizzes_data()
