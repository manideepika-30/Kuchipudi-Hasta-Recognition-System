"""
Authentic Kuchipudi Curriculum Seeder
Provides complete authentic modules, shlokas, Telugu translations, and lessons
extracted from the Anandamayi Nrityalaya canonical curriculum PDFs.
"""
from app.models.curriculum import Module, Lesson

def get_authentic_modules_data():
    # Module 1: Introduction to Kuchipudi
    m1 = {
        "module_number": 1,
        "title_en": "Introduction to Kuchipudi",
        "title_te": "కూచిపూడి పరిచయం",
        "sanskrit_title": "Kuchipudi Natya Parichayam",
        "source_pdf": "Traditional Natyashastra Canon",
        "total_classes": 4,
        "description_en": "The historical evolution, Natyashastra roots, and spiritual significance of classical Kuchipudi dance.",
        "description_te": "కూచిపూడి నృత్యం యొక్క చారిత్రక పరిణామం, నాట్యశాస్త్ర మూలాలు మరియు ఆధ్యాత్మిక ప్రాముఖ్యత.",
        "lessons": [
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
                "meaning_te": "చేయి ఎటు వెళితే దృష్టి అటు వెళుతుంది; దృష్టి ఎటు వెళితే మనసు అటు వెళుతుంది; మనసు ఉన్నచోట భావం, భావం ఉన్నచోట రససిద్ధి కలుగుతుంది.",
                "image_url": "/assets/hastha_anjali.jpeg"
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
                "meaning_te": "నృత్తం లయ-తాళాలను ఆశ్రయిస్తుంది; నృత్యం భావాన్ని ఆశ్రయిస్తుంది; నాట్యం రసాన్ని ఆశ్రయిస్తుంది.",
                "image_url": "/assets/asamyutha_sample.jpeg"
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
                "meaning_te": "సమస్త విశ్వం ఆంగికము, సమస్త వాఙ్మయం వాచికము, చంద్రతారలు ఆహార్యముగా కల సాత్విక శివునికి నమస్కరిస్తున్నాము.",
                "image_url": "/assets/samyutha_sample.jpeg"
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
                "meaning_te": "నాట్యంలో శృంగారము, హాస్యము, కరుణ, రౌద్రము, వీరము, భయానకము, బీభత్సము, అద్భుతము, శాంతము అను తొమ్మిది రసాలు ముఖ్యమైనవి.",
                "image_url": "/assets/drishti_sample.jpeg"
            }
        ]
    }

    # Module 2: Asamyukta Hasthas (28)
    shloka_m2 = "పతాకః త్రిపతాకోఽర్థపతాకః కర్తరీముఖః | మయూరాఖ్యోఽర్థచంద్రశ్చ హ్యరాళః శుకతుండకః || ముష్టిశ్చ శిఖరాఖ్యశ్చ కపిత్థః కటకాముఖః | సూచీ చంద్రకళా పద్మకోశః సర్పశిరస్తథా || మృగశీర్షః సింహముఖః కాంగూలశ్చాలపద్మకః | చతురో భ్రమరశ్చైవ హంసాస్యో హంసపక్షకః || సందంశో ముకులశ్చైవ తామ్రచూడస్త్రిశూలకః | ఇత్యసంయుతహస్తానామష్టావింశతిరీరితా ||"
    asamyukta_tuples = [
        ("Pathaka", "పతాక", "Flag", "When the thumb is bent to touch the base of the forefinger, and other fingers are held closely together and stretched straight.", "asamyukta_01_pathaka.jpg"),
        ("Tripathaka", "త్రిపతాక", "Three parts of flag", "When the ring finger of Pathaka hastha is bent inward while remaining fingers remain straight.", "asamyukta_02_tripathaka.jpg"),
        ("Ardhapathaka", "అర్ధపతాక", "Half-flag", "When both the ring finger and little finger of Pathaka are bent inward.", "asamyukta_03_ardhapathaka.jpg"),
        ("Kartarimukha", "కర్తరీముఖ", "Scissors face", "When the forefinger and little finger of Tripathaka are extended back or spread like scissors.", "asamyukta_04_kartarimukha.jpg"),
        ("Mayura", "మయూర", "Peacock", "In Tripathaka hastha, the ring finger and thumb join tips together while the other fingers are outstretched.", "asamyukta_05_mayura.jpg"),
        ("Ardhachandra", "అర్ధచంద్ర", "Half-moon", "When the thumb of Pathaka hastha is stretched outward away from the other fingers to form a crescent.", "asamyukta_06_ardhachandra.jpg"),
        ("Arala", "అరాళ", "Bent / Curved", "When the index finger is bent curved while other fingers are held upright like Pathaka.", "asamyukta_07_arala.jpg"),
        ("Sukatunda", "శుకతుండ", "Parrot beak", "When the ring finger of Arala hastha is also bent down, resembling a parrot's beak.", "asamyukta_08_sukatunda.jpg"),
        ("Mushti", "ముష్టి", "Fist", "When four fingers are firmly bent into the palm and the thumb is placed securely over them.", "asamyukta_09_mushti.jpg"),
        ("Shikhara", "శిఖర", "Spire / Peak", "From Mushti hastha, when the thumb is raised vertically upright.", "asamyukta_10_shikhara.jpg"),
        ("Kapittha", "కపిత్థ", "Wood-apple", "When the forefinger of Shikhara hastha is bent over the tip of the thumb.", "asamyukta_11_kapittha.jpg"),
        ("Katakamukha", "కటకాముఖ", "Opening of bracelet", "When the index finger, middle finger and thumb join tips together while ring and little fingers are raised.", "asamyukta_12_katakamukha.jpg"),
        ("Soochi", "సూచి", "Needle", "When the middle, ring, and little fingers touch the thumb and the index finger is pointed straight upward.", "asamyukta_13_soochi.jpg"),
        ("Chandrakala", "చంద్రకళ", "Crescent digit of moon", "From Suchi hastha, releasing the thumb and extending it outward at an angle.", "asamyukta_14_chandrakala.jpg"),
        ("Padmakosha", "పద్మకోశ", "Lotus bud", "When all fingers including thumb are curved inward like the petals of a blossoming flower bud.", "asamyukta_15_padmakosha.jpg"),
        ("Sarpasheersha", "సర్పశీర్ష", "Serpent hood", "When the fingers of Pathaka are slightly cupped inward like a cobra's hood.", "asamyukta_16_sarpasheersha.jpg"),
        ("Mrugasheersha", "మృగశీర్ష", "Deer head", "When the thumb and little finger are extended upward while index, middle and ring fingers are bent forward.", "asamyukta_17_mrugasheersha.jpg"),
        ("Simhamukha", "సింహముఖ", "Lion face", "When the middle and ring fingers touch the tip of the thumb while index and little fingers are raised upright.", "asamyukta_18_simhamukha.jpg"),
        ("Kaangula", "కాంగూల", "Tail / Bell", "When the ring finger is bent into the palm while the remaining fingers are curved outwards.", "asamyukta_19_kaangula.jpg"),
        ("Alapadma", "అలపద్మ", "Full-bloomed lotus", "When all fingers are separated and curved gracefully starting from the little finger like a full-blown lotus.", "asamyukta_20_alapadma.jpg"),
        ("Chatura", "చతుర", "Four-sided / Clever", "When the thumb is placed at the base of the middle finger while other three fingers are held together straight.", "asamyukta_21_chatura.jpg"),
        ("Bhramara", "భ్రమర", "Bee", "When the index finger is curled inside while thumb and middle finger touch tips.", "asamyukta_22_bhramara.jpg"),
        ("Hamsasya", "హంసాస్య", "Swan beak", "When the forefinger, middle finger and thumb touch tips together gracefully.", "asamyukta_23_hamsasya.jpg"),
        ("Hamsapaksha", "హంసపక్ష", "Swan wing", "When the little finger is held straight up while the other three fingers are extended sideways.", "asamyukta_24_hamsapaksha.jpg"),
        ("Samdamsha", "సందంశ", "Pincers / Grasp", "When the fingers of Padmakosha are opened and closed repeatedly against the thumb.", "asamyukta_25_samdamsha.jpg"),
        ("Mukula", "ముకుల", "Flower bud", "When all five fingertips are brought together touching at a single point.", "asamyukta_26_mukula.jpg"),
        ("Tamrachuda", "తామ్రచూడ", "Rooster crest", "When the index finger of Mukula is curved hooked outward like a rooster's comb.", "asamyukta_27_tamrachuda.jpg"),
        ("Trishoola", "త్రిశూల", "Trident", "When the thumb and little finger meet at their tips while index, middle, and ring fingers are held straight upright.", "asamyukta_28_trishoola.jpg")
    ]
    m2_lessons = []
    for idx, (nen, nte, ten, den, img) in enumerate(asamyukta_tuples):
        m2_lessons.append({
            "lesson_order": idx + 1,
            "name_en": nen,
            "name_te": nte,
            "sanskrit_name": f"{nen} Hasta",
            "category": "Asamyukta",
            "description_en": f"{nen} ({ten}): {den}",
            "description_te": f"{nte} హస్తము ({ten}): {den}",
            "shloka": shloka_m2,
            "image_url": f"/assets/hastas/{img}"
        })
    m2 = {
        "module_number": 2,
        "title_en": "Asamyukta Hasthas",
        "title_te": "అసంయుక్త హస్తాలు (28 ఏకహస్త ముద్రలు)",
        "sanskrit_title": "Asamyukta Hasta Prakaranam",
        "source_pdf": "Asamyutha_Hasthas.pdf",
        "total_classes": 28,
        "description_en": "The complete canon of 28 single-hand mudras from Abhinaya Darpana with shlokas, hand structures, and usages.",
        "description_te": "అభినయ దర్పణంలో పేర్కొన్న 28 ఒంటిచేతి ముద్రలు, వాటి శ్లోకాలు, చేతి అమరిక మరియు వినియోగాలు.",
        "lessons": m2_lessons
    }

    # Module 3: Samyukta Hasthas (24)
    shloka_m3 = "అంజలిశ్చ కపోతశ్చ కర్కటః స్వస్తికస్తథా | డోలాహస్తః పుష్పపుట ఉత్సంగః శివలింగకః || కటకావర్ధనశ్చైవ కర్తరీస్వస్తికస్తథా | శకటః శంఖచక్రౌ చ సంపుటః పాశకీలకౌ || మత్స్యః కూర్మో వరాహశ్చ గరుడో నాగబంధకః | ఖట్వా భేరుండకశ్చైవ తథావహిత్థకస్తథా || చతుర్వింశతిసంఖ్యాకాః సంయుతాః కథితాః కరాః ||"
    samyukta_tuples = [
        ("Anjali", "అంజలి", "Salutation", "When two Pataka hands are joined palm to palm in front of chest, forehead or head.", "samyukta_01_anjali.jpg"),
        ("Kapotha", "కపోత", "Dove", "When the palms of Anjali touch only at the tips, bases, and edges leaving hollow center.", "samyukta_02_kapothamu.jpg"),
        ("Karkata", "కర్కట", "Crab", "When the fingers of both hands are interlocked and turned inwards or outwards.", "samyukta_03_karkatamu.jpg"),
        ("Swastika", "స్వస్తిక", "Crossed", "When two Pataka hands are held crossed at the wrists.", "samyukta_04_swasthikamu.jpg"),
        ("Dola", "డోలా", "Swing", "When two Pataka hands hang loosely at the respective sides of the thighs.", "samyukta_05_dola_hastamu.jpg"),
        ("Pushpaputa", "పుష్పపుట", "Flower casket", "When two Sarpashirsha hands are joined side-by-side touching to hold flowers.", "samyukta_06_pushpaputamu.jpg"),
        ("Uthsanga", "ఉత్సంగ", "Embrace", "When Arala hands are placed crossed on opposite shoulders.", "samyukta_07_uthsungamu.jpg"),
        ("Shivalinga", "శివలింగ", "Lord Shiva phallus", "When Shikhara hand is placed upright upon the palm of left Ardhachandra hand.", "samyukta_08_sivalingamu.jpg"),
        ("Katakavardhana", "కటకావర్ధన", "Link of bracelets", "When two Katakamukha hands are held crossed at the wrists.", "samyukta_09_katakavardhanamu.jpg"),
        ("Kartari-Swastika", "కర్తరీస్వస్తిక", "Crossed scissors", "When two Kartarimukha hands are held crossed at the wrists.", "samyukta_10_karthariswasthikamu.jpg"),
        ("Sakata", "శకట", "Cart", "When Bhramara hands are held with thumb and middle finger extended.", "samyukta_11_sakatamu.jpg"),
        ("Shankha", "శంఖ", "Conch", "When the thumb of left Shikhara is enclosed by four fingers of right hand while right thumb touches left index.", "samyukta_12_sankhamu.jpg"),
        ("Chakra", "చక్ర", "Discus", "When Ardhachandra hands are placed across each other palm-to-palm at right angles.", "samyukta_13_chakramu.jpg"),
        ("Samputa", "సంపుట", "Box / Casket", "When the fingers of Chakra hands are curved inward forming an enclosed container.", "samyukta_14_samputamu.jpg"),
        ("Pasha", "పాశ", "Noose", "When the index fingers of Suchi hands are interlocked like a chain link.", "samyukta_15_paasamu.jpg"),
        ("Kilaka", "కీలక", "Bond / Intimacy", "When the little fingers of Mrigashirsha hands are interlinked.", "samyukta_16_keelakamu.jpg"),
        ("Matsya", "మత్స్య", "Fish", "When two Pataka hands are placed one over the other with thumbs extended sideways like fins.", "samyukta_17_matsyamu.jpg"),
        ("Kurma", "కూర్మ", "Tortoise", "When Chakra hands have fingers curled inward with thumb and little finger emerging.", "samyukta_18_kurmamu.jpg"),
        ("Varaha", "వరాహ", "Boar", "When Mrigashirsha hands are held one on top of the other, thumbs and little fingers touching.", "samyukta_19_varahamu.jpg"),
        ("Garuda", "గరుడ", "Eagle / Kite", "When Ardhachandra hands are hooked across each other by the thumbs, fluttering like wings.", "samyukta_20_garudamu.jpg"),
        ("Nagabandha", "నాగబంధ", "Serpent coil", "When Sarpashirsha hands are crossed at the wrists and interlocked.", "samyukta_21_nagabandhakamu.jpg"),
        ("Khatva", "ఖట్వా", "Cot / Bed", "When the thumbs and index fingers of Chatushra hands are placed facing downwards.", "samyukta_22_khatva.jpg"),
        ("Bherunda", "భేరుండ", "Double-headed bird", "When Kapittha hands are joined crossed at the wrists.", "samyukta_23_bherundamu.jpg"),
        ("Avahittha", "అవహిత్థ", "Dissimulation / Secret", "When two Alapadma hands are held crossed over the chest.", "samyukta_24_avahithamu.jpg")
    ]
    m3_lessons = []
    for idx, (nen, nte, ten, den, img) in enumerate(samyukta_tuples):
        m3_lessons.append({
            "lesson_order": idx + 1,
            "name_en": nen,
            "name_te": nte,
            "sanskrit_name": f"{nen} Samyukta Hasta",
            "category": "Samyukta",
            "description_en": f"{nen} ({ten}): {den}",
            "description_te": f"{nte} సంయుక్త హస్తము ({ten}): {den}",
            "shloka": shloka_m3,
            "image_url": f"/assets/hastas/{img}"
        })
    m3 = {
        "module_number": 3,
        "title_en": "Samyukta Hasthas",
        "title_te": "సంయుక్త హస్తాలు (24 ద్విహస్త ముద్రలు)",
        "sanskrit_title": "Samyukta Hasta Prakaranam",
        "source_pdf": "Samyutha Hasthas.pdf",
        "total_classes": 24,
        "description_en": "The 24 combined two-hand gestures from Abhinaya Darpana requiring coordinate alignment of both hands.",
        "description_te": "రెండు చేతుల కలయికతో రూపొందే 24 సంయుక్త హస్తాలు మరియు వాటి నాట్య వినియోగాలు.",
        "lessons": m3_lessons
    }

    # Module 4: Shiro Bhedas (9)
    shloka_m4 = "సమం ఉద్వాహితం చాధోముఖం ఆలోలితం ధుతమ్ | కంపితం చ పరావృత్తం ఉత్క్షిప్తం పరివాహితమ్ || నవధా కథితం శీర్షం నాట్యశాస్త్రవిశారదైః ||"
    shiro_tuples = [
        ("Sama", "సమ", "Level / Still", "Head is held motionless in neutral erect center posture. Used in beginning of dance, prayer, and calm reflection.", "shiro_01_sama.jpg"),
        ("Udvahitam", "ఉద్వాహితమ్", "Raised up", "Head is lifted upward looking toward the sky. Used to depict the moon, flag, sky, divine temples.", "shiro_02_udvahitam.jpg"),
        ("Adhomukham", "అధోముఖమ్", "Cast down", "Face and head are bowed downward. Used to portray modesty, sorrow, shame, bowing, and looking at the earth.", "shiro_03_adhomukham.jpg"),
        ("Alolitam", "ఆలోలితమ్", "Rolling circular", "Head moves gently in a circular rolling motion. Used in dizziness, sleepiness, possession, uncontrollable laughter.", "shiro_04_alolitam.jpg"),
        ("Dhutam", "ధూతమ్", "Side-to-side shake", "Head turns horizontally from left to right and right to left. Used to show denial, rejection, looking sideways.", "shiro_05_dhutam.jpg"),
        ("Kampitam", "కంపితమ్", "Nodding up-down", "Head moves up and down vertically in quick nods. Used in inquiry, beckoning, affirmation, and threatening.", "shiro_06_kampitam.jpg"),
        ("Paravrittam", "పరావృత్తమ్", "Turned aside", "Head is briskly turned sharply to one side. Used in turning away in anger, modesty, hair dressing, aversion.", "shiro_07_paravrittam.jpg"),
        ("Ukshiptam", "ఉత్క్షిప్తమ్", "Tilted upward", "Head is turned sideways and tilted upwards. Used in indicating 'take this', cherishing, pride, and divine welcome.", "shiro_08_ukshiptam.jpg"),
        ("Parivahitam", "పరివాహితమ్", "Gently swinging", "Head rolls smoothly side to side like a gentle fan. Used in joyous appreciation, love longing, and ecstasy.", "shiro_09_parivahittam.jpg")
    ]
    m4_lessons = []
    for idx, (nen, nte, ten, den, img) in enumerate(shiro_tuples):
        m4_lessons.append({
            "lesson_order": idx + 1,
            "name_en": nen,
            "name_te": nte,
            "sanskrit_name": f"{nen} Shiro Bheda",
            "category": "Shiro",
            "description_en": f"{nen} ({ten}): {den}",
            "description_te": f"{nte} శిరోభేదము ({ten}): {den}",
            "shloka": shloka_m4,
            "image_url": f"/assets/bhedas/{img}"
        })
    m4 = {
        "module_number": 4,
        "title_en": "Shiro Bhedhas",
        "title_te": "శిరో భేదాలు (9 శిరస్సు కదలికలు)",
        "sanskrit_title": "Shiro Bheda Prakaranam",
        "source_pdf": "Shiro Bhedhas.pdf",
        "total_classes": 9,
        "description_en": "The 9 canonical head movements from Natyashastra and Abhinaya Darpana expressing distinct emotional states.",
        "description_te": "నాట్యశాస్త్రం మరియు అభినయ దర్పణంలో చెప్పబడిన 9 శిరో భేదాలు మరియు వాటి భావ ప్రకటనలు.",
        "lessons": m4_lessons
    }

    # Module 5: Drishti Bhedas (8)
    shloka_m5 = "సమం ఆలోకితం సాచీ ప్రలోకిత నిమీలితే | ఉల్లోకితానువృత్తే చ తథా చైవావలోకితమ్ || ఇత్యష్టౌ దృష్టిభేదాః స్యుః కీర్తితాః పూర్వసూరిభిః ||"
    drishti_tuples = [
        ("Sama", "సమ", "Level glance", "Eyes kept straight, open, still and tranquil without blinking. Used at the opening of dance, deities, and neutral thought.", "drishti_01_sama.jpg"),
        ("Alokita", "ఆలోకిత", "Rolling circular", "Rolling the eyeballs briskly in a wide circular motion. Used in looking at a potter's wheel, all directions, or begging.", "drishti_02_alokita.jpg"),
        ("Sachi", "సాచి", "Corner glance", "Looking obliquely through the corner of the eyes without moving the face. Used in shooting arrows, remembrance, secret love.", "drishti_03_saachi.jpg"),
        ("Pralokita", "ప్రలోకిత", "Side-to-side scan", "Eyes moving swiftly from one side to the other. Used to look at objects on both sides, balance, and scanning audience.", "drishti_04_pralokita.jpg"),
        ("Nimilita", "నిమీలిత", "Half-closed gaze", "Eyes half shut, gazing downward toward the heart. Used in meditation, prayer, submission, and inner trance.", "drishti_05_nimilita.jpg"),
        ("Ullokita", "ఉల్లోకిత", "Upward glance", "Gazing upward directly toward the heights. Used to behold tall trees, the sky, flying birds, and celestial beings.", "drishti_06_ullokita.jpg"),
        ("Anuvritta", "అనువృత్త", "Rapid up-down scan", "Rapid movement of the eyes up and down in quick succession. Used in anger, surveying someone head to toe, and agitation.", "drishti_07_anuvritta.jpg"),
        ("Avalokita", "అవలోకిత", "Downward glance", "Looking steadily downward towards the ground. Used in looking at one's shadow, modesty, fatigue, and observing the earth.", "drishti_08_avalokita.jpg")
    ]
    m5_lessons = []
    for idx, (nen, nte, ten, den, img) in enumerate(drishti_tuples):
        m5_lessons.append({
            "lesson_order": idx + 1,
            "name_en": nen,
            "name_te": nte,
            "sanskrit_name": f"{nen} Drishti Bheda",
            "category": "Drishti",
            "description_en": f"{nen} ({ten}): {den}",
            "description_te": f"{nte} దృష్టి భేదము ({ten}): {den}",
            "shloka": shloka_m5,
            "image_url": f"/assets/bhedas/{img}"
        })
    m5 = {
        "module_number": 5,
        "title_en": "Drishti Bhedhas",
        "title_te": "దృష్టి భేదాలు (8 నేత్ర కదలికలు)",
        "sanskrit_title": "Drishti Bheda Prakaranam",
        "source_pdf": "Drishti Bhedhas.pdf",
        "total_classes": 8,
        "description_en": "The 8 canonical glances of the eyes fundamental to classical Sattvika Abhinaya and emotional expression.",
        "description_te": "సాత్త్వికాభినయానికి మూలమైన 8 నేత్ర దృష్టి భేదాలు, వాటి వీక్షణ పద్ధతులు మరియు భావాభినయాలు.",
        "lessons": m5_lessons
    }

    # Module 6: Dvadasha Hasta Pranamas (12)
    shloka_m6 = "ప్రసారణం కుంచితం చ రేచితం పుంఖితం తథా | అపవేష్టితకం చాపి ప్రేరితోద్వేష్టితౌ తథా || వ్యవృత్తః పరివృత్తశ్చ సంకేతస్తదనంతరమ్ | చిహ్నం పదార్థటీకేతి ప్రాణా ద్వాదశ హస్తజాః ||"
    pranama_tuples = [
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
    m6_lessons = []
    for idx, (nen, nte, den) in enumerate(pranama_tuples):
        m6_lessons.append({
            "lesson_order": idx + 1,
            "name_en": nen,
            "name_te": nte,
            "sanskrit_name": f"{nen} Pranama",
            "category": "Pranama",
            "description_en": f"{nen}: {den}",
            "description_te": f"{nte}: {den}",
            "shloka": shloka_m6,
            "image_url": None
        })
    m6 = {
        "module_number": 6,
        "title_en": "Dvadasha Hasta Pranamas",
        "title_te": "ద్వాదశ హస్త ప్రణామాలు (12 నమస్కార ముద్రలు)",
        "sanskrit_title": "Dvadasha Hasta Pranama Prakaranam",
        "source_pdf": "Dvadasha Hasta Pranamas.pdf",
        "total_classes": 12,
        "description_en": "12 traditional classical hand salutation actions used in invocation, guru-vandana, and sabha pranamam.",
        "description_te": "దైవము, గురువు మరియు సభకు భక్తిపూర్వకంగా సమర్పించే 12 శాస్త్రీయ హస్త ప్రణామాలు.",
        "lessons": m6_lessons
    }

    # Module 7: Sapta Talamulu (7) — With Individual Canonical Shlokas from Sapta Talamulu.pdf
    tala_records = [
        {
            "order": 1,
            "name_en": "Dhruva Tala",
            "name_te": "ధ్రువ తాళము",
            "sanskrit_name": "Dhruva Tāla (ధ్రువ తాళః)",
            "shloka": "భరతో దత్తిశ్చైవ ఆంజనేయో మతో ధ్రువః ||\nBharato Dattishchaiva Anjaneyo Matodhruva ||",
            "meaning_en": "Bharatacharya, Dattishi, and Anjaneya together created Dhruva Tala. Components (Angamulu): Laghu, Dhrutam, Laghu, Laghu (Pattern: 1 0 1 1).",
            "meaning_te": "భరతాచార్యుడు, దత్తి మరియు ఆంజనేయుడు కలిసి ధ్రువ తాళాన్ని సృష్టించారు. అంగములు: లఘు, ధ్రుతం, లఘు, లఘు (లయ సూత్రం: 1 0 1 1).",
            "description_en": "Dhruva Tala is the first of the Sapta Talas. Structure: Laghu (1), Dhrutam (0), Laghu (1), Laghu (1) [Pattern: 1 0 1 1]. Created by Bharatacharya, Dattishi, and Anjaneya. In Chaturasra Jaati (Laghu = 4), it comprises 14 Aksharas (4+2+4+4) known as Srikara Tala.",
            "description_te": "సప్త తాళములలో మొదటిది ధ్రువ తాళము. అంగములు: లఘువు, ధ్రుతము, లఘువు, లఘువు (అమరిక: 1 0 1 1). తాళ కర్తలు: భరతాచార్యుడు, దత్తి మరియు ఆంజనేయుడు. చతురస్ర జాతిలో (లఘువు = 4) దీనిని శ్రీకర తాళము (14 అక్షరాలు: 4+2+4+4) అంటారు."
        },
        {
            "order": 2,
            "name_en": "Matya Tala",
            "name_te": "మత్య తాళము",
            "sanskrit_name": "Matya Tāla (మత్య తాళః)",
            "shloka": "మత్య తాళస్య కర్తారో ముతంగర్నుస కోహల ||\nMatya Talasya Kartaromutangarnusa Kohala ||",
            "meaning_en": "Matanga Maharshi, Arjuna, and Kohalacharyudu together created Matya Tala. Components (Angamulu): Laghu, Dhrutam, Laghu (Pattern: 1 0 1).",
            "meaning_te": "మాతంగ మహర్షి, అర్జునుడు మరియు కోహలాచార్యుడు కలిసి మత్య తాళాన్ని సృష్టించారు. అంగములు: లఘు, ధ్రుతం, లఘు (లయ సూత్రం: 1 0 1).",
            "description_en": "Matya Tala is the second of the Sapta Talas. Structure: Laghu (1), Dhrutam (0), Laghu (1) [Pattern: 1 0 1]. Created by Matanga Maharshi, Arjuna, and Kohala. In Chaturasra Jaati, it consists of 10 Aksharas (4+2+4) and is named Sama Tala.",
            "description_te": "సప్త తాళములలో రెండవది మత్య తాళము. అంగములు: లఘువు, ధ్రుతము, లఘువు (అమరిక: 1 0 1). తాళ కర్తలు: మాతంగ మహర్షి, అర్జునుడు మరియు కోహలాచార్యుడు. చతురస్ర జాతిలో దీనిని సమ తాళము (10 అక్షరాలు: 4+2+4) అంటారు."
        },
        {
            "order": 3,
            "name_en": "Rupaka Tala",
            "name_te": "రూపక తాళము",
            "sanskrit_name": "Rūpaka Tāla (రూపక తాళః)",
            "shloka": "ఆంజనేయ కుంభోజి పి యాజ్ఞవల్క్యస్తు రూపకే ||\nAnjaneya KumbhojiPi Yajnavalkasthu Rupake ||",
            "meaning_en": "Anjaneya, Kumbhoja (Sage Agastya), and Yajnavalkya together created Rupaka Tala. Components (Angamulu): Dhrutam, Laghu (Pattern: 0 1).",
            "meaning_te": "ఆంజనేయుడు, కుంభోజుడు (అగస్త్యుడు), మరియు యాజ్ఞవల్క్యుడు కలిసి రూపక తాళాన్ని సృష్టించారు. అంగములు: ధ్రుతం, లఘు (లయ సూత్రం: 0 1).",
            "description_en": "Rupaka Tala is the third of the Sapta Talas. Structure: Dhrutam (0), Laghu (1) [Pattern: 0 1]. Created by Anjaneya, Kumbhoja (Agastya), and Yajnavalkya. In Chaturasra Jaati, it has 6 Aksharas (2+4) and is designated as Prati Tala.",
            "description_te": "సప్త తాళములలో మూడవది రూపక తాళము. అంగములు: ధ్రుతము, లఘువు (అమరిక: 0 1). తాళ కర్తలు: ఆంజనేయుడు, కుంభోజుడు (అగస్త్య మహర్షి) మరియు యాజ్ఞవల్క్యుడు. చతురస్ర జాతిలో దీనిని ప్రతి తాళము (6 అక్షరాలు: 2+4) అంటారు."
        },
        {
            "order": 4,
            "name_en": "Jhampe Tala",
            "name_te": "ఝంపె తాళము",
            "sanskrit_name": "Jhampe Tāla (ఝంపె తాళః)",
            "shloka": "నందిశ్చ భరతశ్చ శనక ఝంపాయాం చైవ తథా ||\nNandishcha Bharatashcha Shanaka Jhampayamchevasthadha ||",
            "meaning_en": "Nandikeswara, Bharata, and Shanaka together created Jhampe Tala. Components (Angamulu): Laghu, Anudhrutam, Dhrutam (Pattern: 1 c 0).",
            "meaning_te": "నందికేశ్వరుడు, భరతుడు మరియు శనకుడు కలిసి ఝంపె తాళాన్ని సృష్టించారు. అంగములు: లఘు, అనుధ్రుతం, ధ్రుతం (లయ సూత్రం: 1 c 0).",
            "description_en": "Jhampe Tala is the fourth of the Sapta Talas. Structure: Laghu (1), Anudhrutam (c), Dhrutam (0) [Pattern: 1 c 0]. Created by Nandikeswara, Bharata, and Shanaka. In Chaturasra Jaati, it has 7 Aksharas (4+1+2) called Madhura Tala.",
            "description_te": "సప్త తాళములలో నాల్గవది ఝంపె తాళము. అంగములు: లఘువు, అనుధ్రుతము, ధ్రుతము (అమరిక: 1 c 0). తాళ కర్తలు: నందికేశ్వరుడు, భరతాచార్యుడు మరియు శనకుడు. చతురస్ర జాతిలో దీనిని మధుర తాళము (7 అక్షరాలు: 4+1+2) అంటారు."
        },
        {
            "order": 5,
            "name_en": "Triputa Tala",
            "name_te": "త్రిపుట తాళము",
            "sanskrit_name": "Tripuṭa Tāla (త్రిపుట తాళః)",
            "shloka": "నిశాచరపతిః బ్రహ్మ త్రిపుటే శంఖ పాలకః ||\nNishacharapatihi Brahmo Tripute Shankha Palaka ||",
            "meaning_en": "Ravanudu (Nishacharapati), Brahma, and Shankha Paludu together created Triputa Tala. Components (Angamulu): Laghu, Dhrutam, Dhrutam (Pattern: 1 0 0).",
            "meaning_te": "నిశాచరపతి (రావణాసురుడు), బ్రహ్మదేవుడు మరియు శంఖపాలుడు కలిసి త్రిపుట తాళాన్ని సృష్టించారు. అంగములు: లఘు, ధ్రుతం, ధ్రుతం (లయ సూత్రం: 1 0 0).",
            "description_en": "Triputa Tala is the fifth of the Sapta Talas. Structure: Laghu (1), Dhrutam (0), Dhrutam (0) [Pattern: 1 0 0]. Created by Nishacharapati (Ravana), Brahma, and Shankha Pala. In Chaturasra Jaati (4+2+2 = 8 Aksharas), this is the famous Adi Tala, the cornerstone rhythm of South Indian classical music and dance.",
            "description_te": "సప్త తాళములలో ఐదవది త్రిపుట తాళము. అంగములు: లఘువు, ధ్రుతము, ధ్రుతము (అమరిక: 1 0 0). తాళ కర్తలు: నిశాచరపతి (రావణుడు), బ్రహ్మ మరియు శంఖపాలుడు. చతురస్ర జాతిలో (4+2+2 = 8 అక్షరాలు) దీనిని ప్రసిద్ధ ఆది తాళము అంటారు."
        },
        {
            "order": 6,
            "name_en": "Ata Tala",
            "name_te": "అట తాళము",
            "sanskrit_name": "Aṭa Tāla (అట తాళః)",
            "shloka": "బలిః కుమార దేవేంద్ర అట తాళ సముద్భవః ||\nBalihi Kumara Devendra Ata Tala Samudbhava ||",
            "meaning_en": "Balichakravarti, Devendra, and Kumaraswami together created Ata Tala. Components (Angamulu): Laghu, Laghu, Dhrutam, Dhrutam (Pattern: 1 1 0 0).",
            "meaning_te": "బలిచక్రవర్తి, దేవేంద్రుడు మరియు కుమారస్వామి కలిసి అట తాళాన్ని సృష్టించారు. అంగములు: లఘు, లఘు, ధ్రుతం, ధ్రుతం (లయ సూత్రం: 1 1 0 0).",
            "description_en": "Ata Tala is the sixth of the Sapta Talas. Structure: Laghu (1), Laghu (1), Dhrutam (0), Dhrutam (0) [Pattern: 1 1 0 0]. Created by Bali Chakravarti, Devendra, and Kumaraswami (Kartikeya). In Chaturasra Jaati, it consists of 12 Aksharas (4+4+2+2) known as Lekha Tala, and in Khanda Jaati it forms the grand 14-beat Khanda Ata Tala.",
            "description_te": "సప్త తాళములలో ఆరవది అట తాళము. అంగములు: లఘువు, లఘువు, ధ్రుతము, ధ్రుతము (అమరిక: 1 1 0 0). తాళ కర్తలు: బలిచక్రవర్తి, దేవేంద్రుడు మరియు కుమారస్వామి. చతురస్ర జాతిలో దీనిని లేఖ తాళము (12 అక్షరాలు: 4+4+2+2) అంటారు."
        },
        {
            "order": 7,
            "name_en": "Eka Tala",
            "name_te": "ఏక తాళము",
            "sanskrit_name": "Eka Tāla (ఏక తాళః)",
            "shloka": "మహతాంచాపి సర్వేషాం ఏకతాళ సమం విదుః ||\nMahatanchaapi Sarvesham Ekatala Samam Viduh ||",
            "meaning_en": "By the consensus and agreement of all the great ancient masters, Eka Tala was formulated. Components (Angamulu): Laghu (Pattern: 1).",
            "meaning_te": "మహాత్ములందరి ఏకాభిప్రాయంతో ఏక తాళం రూపొందించబడింది. అంగములు: లఘువు (లయ సూత్రం: 1).",
            "description_en": "Eka Tala is the seventh of the Sapta Talas. Structure: single Laghu (1) [Pattern: 1]. Formulated through the mutual consensus of all ancient scholars. In Chaturasra Jaati, it is a 4-beat cycle called Mana Tala, representing the fundamental unit of rhythmic pulse.",
            "description_te": "సప్త తాళములలో ఏడవది ఏక తాళము. అంగము: కేవలం ఒక లఘువు (అమరిక: 1). తాళ కర్తలు: సర్వ మహనీయుల సమ్మతితో రూపొందించబడింది. చతురస్ర జాతిలో 4 అక్షరాల కాలంతో దీనిని మాన తాళము అంటారు."
        }
    ]
    m7_lessons = []
    for item in tala_records:
        m7_lessons.append({
            "lesson_order": item["order"],
            "name_en": item["name_en"],
            "name_te": item["name_te"],
            "sanskrit_name": item["sanskrit_name"],
            "category": "Tala",
            "description_en": item["description_en"],
            "description_te": item["description_te"],
            "shloka": item["shloka"],
            "meaning_en": item["meaning_en"],
            "meaning_te": item["meaning_te"],
            "image_url": None
        })
    m7 = {
        "module_number": 7,
        "title_en": "Sapta Talamulu",
        "title_te": "సప్త తాళములు (7 ప్రధాన తాళాలు మరియు 35 తాళ వ్యవస్థ)",
        "sanskrit_title": "Sapta Tala Prakaranam",
        "source_pdf": "Sapta Talamulu.pdf",
        "total_classes": 7,
        "description_en": "The 7 foundational rhythmic cycles, their mythological creators (Kartalu), components (Angas), and the 35 Tala Matrix.",
        "description_te": "కూచిపూడి నృత్యానికి ప్రాణమైన 7 ప్రధాన తాళాలు, వాటి కర్తలు, అంగములు (లఘు, ధ్రుత, అనుధ్రుత) మరియు 35 తాళాల సమగ్ర పట్టిక.",
        "lessons": m7_lessons
    }

    return [m1, m2, m3, m4, m5, m6, m7]
