"""Generate a synthetic hospital corpus (no real people).

Creates 40 documents under data/raw/:
  12 case notes, 10 discharge summaries, 10 billing records, 8 policies
plus data/manifest.json (doc -> category, allowed roles) and
data/pii_ground_truth.json (every planted personal value, used to prove
that redaction worked).

Run:  python scripts/generate_data.py
"""

from __future__ import annotations

import json
import random
import shutil
import sys
from datetime import date, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from medrag.config import CATEGORY_ROLES, DATA_DIR, MANIFEST_PATH, PII_TRUTH_PATH, RAW_DIR  # noqa: E402

rng = random.Random(2026)
HOSPITAL = "Deccan Sunrise Multispeciality Hospital"

# --- Name and address pools (fictional combinations) ----------------------

MALE = ["Ravi", "Srinivas", "Venkatesh", "Arjun", "Mohammed", "Kiran", "Suresh", "Rahul", "Praveen",
        "Imran", "Naveen", "Harsha", "Rajesh", "Anil", "Vikram", "Sai", "Karthik", "Farhan"]
FEMALE = ["Lakshmi", "Priya", "Sravani", "Ananya", "Fatima", "Kavitha", "Divya", "Swathi", "Padma",
          "Ayesha", "Bhavana", "Meena", "Sneha", "Revathi", "Sowmya", "Nikhila", "Zoya", "Haritha"]
SURNAMES = ["Reddy", "Rao", "Naidu", "Chowdary", "Varma", "Goud", "Sharma", "Iyer", "Khan", "Siddiqui",
            "Patel", "Kulkarni", "Yadav", "Murthy", "Pillai", "Deshmukh", "Nair", "Ansari", "Bhat", "Menon"]
LOCALITIES = [("Banjara Hills", "500034"), ("Jubilee Hills", "500033"), ("Madhapur", "500081"),
              ("Gachibowli", "500032"), ("Kukatpally", "500072"), ("Ameerpet", "500016"),
              ("Secunderabad", "500003"), ("Dilsukhnagar", "500060"), ("Mehdipatnam", "500028"),
              ("Kondapur", "500084"), ("LB Nagar", "500074"), ("Miyapur", "500049"),
              ("Tarnaka", "500017"), ("Himayatnagar", "500029"), ("Begumpet", "500016")]
BUILDINGS = ["Sai Residency", "Lakshmi Enclave", "Green Meadows Apartments", "Sri Balaji Towers",
             "Navya Heights", "Aditya Nest", "Pearl Residency", "Vasavi Homes", "Ramky Towers"]
BANKS = [("State Bank of India", "SBIN"), ("HDFC Bank", "HDFC"), ("ICICI Bank", "ICIC"),
         ("Axis Bank", "UTIB"), ("Canara Bank", "CNRB"), ("Union Bank of India", "UBIN")]
EMAIL_DOMAINS = ["gmail.com", "yahoo.co.in", "rediffmail.com", "outlook.com"]

# --- Verhoeff checksum (Aadhaar numbers use it) ---------------------------

_D = [[0, 1, 2, 3, 4, 5, 6, 7, 8, 9], [1, 2, 3, 4, 0, 6, 7, 8, 9, 5], [2, 3, 4, 0, 1, 7, 8, 9, 5, 6],
      [3, 4, 0, 1, 2, 8, 9, 5, 6, 7], [4, 0, 1, 2, 3, 9, 5, 6, 7, 8], [5, 9, 8, 7, 6, 0, 4, 3, 2, 1],
      [6, 5, 9, 8, 7, 1, 0, 4, 3, 2], [7, 6, 5, 9, 8, 2, 1, 0, 4, 3], [8, 7, 6, 5, 9, 3, 2, 1, 0, 4],
      [9, 8, 7, 6, 5, 4, 3, 2, 1, 0]]
_P = [[0, 1, 2, 3, 4, 5, 6, 7, 8, 9], [1, 5, 7, 6, 2, 8, 3, 0, 9, 4], [5, 8, 0, 3, 7, 9, 6, 1, 4, 2],
      [8, 9, 1, 6, 0, 4, 3, 5, 2, 7], [9, 4, 5, 3, 1, 2, 6, 8, 7, 0], [4, 2, 8, 6, 5, 7, 3, 9, 0, 1],
      [2, 7, 9, 3, 8, 0, 6, 4, 1, 5], [7, 0, 4, 6, 9, 1, 3, 2, 5, 8]]
_INV = [0, 4, 3, 2, 1, 5, 6, 7, 8, 9]


def verhoeff_digit(number: str) -> str:
    c = 0
    for i, ch in enumerate(reversed(number)):
        c = _D[c][_P[(i + 1) % 8][int(ch)]]
    return str(_INV[c])


# --- Personal data generators ---------------------------------------------


class Planted:
    """Collects every personal value written into a document."""

    def __init__(self):
        self.items: list[dict] = []

    def __call__(self, kind: str, value: str) -> str:
        self.items.append({"type": kind, "value": value})
        return value


def aadhaar() -> str:
    base = str(rng.randint(2, 9)) + "".join(str(rng.randint(0, 9)) for _ in range(10))
    full = base + verhoeff_digit(base)
    return f"{full[:4]} {full[4:8]} {full[8:]}"


def pan(surname: str) -> str:
    letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    return "".join(rng.choice(letters) for _ in range(3)) + "P" + surname[0].upper() + \
        f"{rng.randint(1000, 9999)}" + rng.choice(letters)


def phone() -> str:
    digits = str(rng.choice([6, 7, 8, 9])) + "".join(str(rng.randint(0, 9)) for _ in range(9))
    style = rng.randint(0, 2)
    if style == 0:
        return f"+91 {digits[:5]} {digits[5:]}"
    if style == 1:
        return f"+91-{digits}"
    return digits


def email(first: str, last: str) -> str:
    return f"{first.lower()}.{last.lower()}{rng.randint(1, 99)}@{rng.choice(EMAIL_DOMAINS)}"


def address() -> str:
    locality, pin = rng.choice(LOCALITIES)
    return (f"Flat {rng.randint(1, 9)}0{rng.randint(1, 9)}, {rng.choice(BUILDINGS)}, "
            f"Road No. {rng.randint(1, 45)}, {locality}, Hyderabad, Telangana {pin}")


def uhid() -> str:
    return f"DSH-26-{rng.randint(100000, 999999)}"


def bank_account() -> tuple[str, str, str]:
    bank, code = rng.choice(BANKS)
    acct = "".join(str(rng.randint(0, 9)) for _ in range(rng.choice([11, 12, 14])))
    ifsc = f"{code}0{rng.randint(100000, 999999)}"
    return bank, acct, ifsc


def person(sex: str) -> tuple[str, str]:
    first = rng.choice(MALE if sex == "M" else FEMALE)
    return first, rng.choice(SURNAMES)


def inr(amount: float) -> str:
    """Format rupees with Indian digit grouping: 1,45,320.00"""
    whole, frac = f"{amount:.2f}".split(".")
    if len(whole) > 3:
        head, tail = whole[:-3], whole[-3:]
        groups = []
        while len(head) > 2:
            groups.insert(0, head[-2:])
            head = head[:-2]
        if head:
            groups.insert(0, head)
        whole = ",".join(groups + [tail])
    return f"Rs. {whole}.{frac}"


# --- Clinical scenarios ---------------------------------------------------
# Each scenario drives the case note, discharge summary and bill of one patient.

SCENARIOS = [
    dict(key="stemi", dept="Cardiology", sex="M", age=58, los=5, room="ICU / CCU",
         diagnosis="Acute inferior wall ST-elevation myocardial infarction (STEMI)",
         complaint="Crushing central chest pain radiating to the left arm for 2 hours with sweating.",
         history="Known hypertensive for 8 years on amlodipine 5 mg. Smoker, 20 pack-years. No prior cardiac events.",
         exam="BP 100/70 mmHg, HR 52/min, SpO2 95% on room air. S1 S2 heard, no murmur. Chest clear.",
         investigations="ECG: ST elevation in II, III, aVF. Troponin I 18.4 ng/mL. Echo: inferior wall hypokinesia, EF 45%.",
         plan="Loaded with aspirin 300 mg, ticagrelor 180 mg. Primary PCI to the right coronary artery (RCA) with one drug-eluting stent. Door-to-balloon time 62 minutes.",
         procedure="Primary PCI with drug-eluting stent to RCA", procedure_cost=185000,
         meds=["Aspirin 75 mg once daily", "Ticagrelor 90 mg twice daily for 12 months",
               "Atorvastatin 80 mg at night", "Metoprolol succinate 25 mg once daily", "Ramipril 2.5 mg once daily"],
         care=["No heavy lifting or driving for 2 weeks", "Low-salt, low-fat cardiac diet",
               "Do not stop ticagrelor without consulting the cardiologist", "Enrol in cardiac rehabilitation in week 2",
               "Return immediately for chest pain, breathlessness or black stools"],
         follow_up="Cardiology OPD after 7 days with lipid profile", payer="Star Health Insurance (cashless)"),
    dict(key="dka", dept="Endocrinology", sex="F", age=34, los=4, room="HDU",
         diagnosis="Diabetic ketoacidosis (DKA) in type 1 diabetes mellitus",
         complaint="Vomiting, abdominal pain and drowsiness for 1 day after missing insulin doses.",
         history="Type 1 diabetes for 12 years on basal-bolus insulin. Missed insulin for 3 days during a viral illness.",
         exam="Drowsy but arousable, GCS 14/15. Dehydrated, Kussmaul breathing. BP 96/60 mmHg, HR 118/min.",
         investigations="Blood glucose 486 mg/dL, pH 7.12, bicarbonate 9 mmol/L, serum ketones 5.8 mmol/L, potassium 5.6 mmol/L.",
         plan="DKA protocol: 0.9% saline resuscitation, fixed-rate IV insulin infusion at 0.1 units/kg/hour, hourly glucose and 2-hourly potassium monitoring.",
         procedure="DKA management with IV insulin infusion", procedure_cost=18000,
         meds=["Insulin glargine 18 units at bedtime", "Insulin aspart 6 units before each meal",
               "Tab. Ondansetron 4 mg as needed for nausea"],
         care=["Check capillary blood glucose four times a day and keep a log",
               "Never skip basal insulin, even when unwell (sick-day rules explained)",
               "Check urine ketones if glucose is above 250 mg/dL",
               "Carry glucose tablets for hypoglycaemia"],
         follow_up="Endocrinology OPD after 10 days with HbA1c", payer="ICICI Lombard (reimbursement)"),
    dict(key="dengue", dept="Internal Medicine", sex="M", age=27, los=6, room="General Ward",
         diagnosis="Dengue fever with warning signs and severe thrombocytopenia",
         complaint="High-grade fever for 4 days with body ache, retro-orbital pain and gum bleeding.",
         history="No comorbidities. Lives in an area with a recent dengue cluster.",
         exam="Temperature 102.4 F, BP 104/70 mmHg. Petechiae over both forearms. Mild hepatomegaly.",
         investigations="NS1 antigen positive. Platelet count 28,000/cumm on day 2, nadir 18,000/cumm. Haematocrit 46%.",
         plan="IV crystalloids as per WHO dengue guidelines, strict input-output charting, 6-hourly haematocrit and platelet count. Avoid NSAIDs and IM injections.",
         procedure="Single donor platelet transfusion (1 unit)", procedure_cost=14500,
         meds=["Paracetamol 650 mg every 6 hours if fever (maximum 4 tablets a day)",
               "ORS 2 litres a day", "Avoid ibuprofen, diclofenac and aspirin"],
         care=["Drink at least 3 litres of fluids daily", "Watch for bleeding, black stools or abdominal pain",
               "Repeat platelet count after 3 days", "Use mosquito nets and repellents at home"],
         follow_up="Medicine OPD after 3 days with CBC", payer="Aarogyasri (Telangana State scheme)"),
    dict(key="pneumonia", dept="Pulmonology", sex="F", age=67, los=7, room="Semi-Private Ward",
         diagnosis="Community-acquired pneumonia, right lower lobe (CURB-65 score 2)",
         complaint="Fever with productive cough and breathlessness for 5 days.",
         history="Type 2 diabetes on metformin. No history of tuberculosis.",
         exam="RR 26/min, SpO2 89% on room air, bronchial breath sounds right lower zone.",
         investigations="Chest X-ray: right lower lobe consolidation. WBC 16,800/cumm. CRP 142 mg/L. Sputum culture: Streptococcus pneumoniae.",
         plan="Oxygen by nasal cannula to keep SpO2 above 94%. IV ceftriaxone 2 g daily plus oral azithromycin 500 mg daily. Chest physiotherapy.",
         procedure="IV antibiotics and oxygen therapy", procedure_cost=22000,
         meds=["Tab. Cefuroxime 500 mg twice daily for 5 more days", "Tab. Metformin 500 mg twice daily",
               "Paracetamol 650 mg as needed"],
         care=["Breathing exercises with incentive spirometer 10 times every hour while awake",
               "Monitor sugars twice daily", "Pneumococcal and influenza vaccination advised after recovery"],
         follow_up="Pulmonology OPD after 2 weeks with repeat chest X-ray", payer="Self-pay"),
    dict(key="appendicitis", dept="General Surgery", sex="M", age=22, los=3, room="Semi-Private Ward",
         diagnosis="Acute appendicitis (Alvarado score 8)",
         complaint="Periumbilical pain shifting to the right iliac fossa for 18 hours, with vomiting.",
         history="No prior surgery. No comorbidities.",
         exam="Tenderness and guarding at McBurney's point, rebound tenderness positive. Temperature 100.8 F.",
         investigations="WBC 14,200/cumm. Ultrasound: non-compressible appendix 9 mm with periappendiceal fluid.",
         plan="Laparoscopic appendectomy under general anaesthesia. Pre-operative IV cefuroxime and metronidazole.",
         procedure="Laparoscopic appendectomy", procedure_cost=65000,
         meds=["Tab. Amoxicillin-clavulanate 625 mg twice daily for 5 days", "Tab. Paracetamol 650 mg every 8 hours for 3 days",
               "Tab. Pantoprazole 40 mg before breakfast for 5 days"],
         care=["Keep port-site dressings dry; dressing change on day 3",
               "Soft diet for 2 days, then normal diet", "No lifting above 5 kg for 4 weeks",
               "Report fever, redness or discharge from the wounds"],
         follow_up="Surgery OPD on day 7 for suture removal", payer="Star Health Insurance (cashless)"),
    dict(key="hip", dept="Orthopaedics", sex="F", age=74, los=8, room="Private Room",
         diagnosis="Displaced fracture neck of left femur (Garden type IV)",
         complaint="Fall at home in the bathroom, unable to bear weight on the left leg.",
         history="Osteoporosis, hypothyroidism on levothyroxine 50 mcg.",
         exam="Left lower limb shortened and externally rotated. Distal pulses present.",
         investigations="X-ray pelvis: displaced intracapsular fracture of the left femoral neck. Hb 10.2 g/dL. Vitamin D 12 ng/mL.",
         plan="Left cemented bipolar hemiarthroplasty. DVT prophylaxis with enoxaparin 40 mg subcutaneously daily. Fall-risk precautions.",
         procedure="Left cemented bipolar hemiarthroplasty", procedure_cost=210000,
         meds=["Inj. Enoxaparin 40 mg subcutaneously once daily for 4 weeks", "Tab. Calcium 500 mg with Vitamin D3 twice daily",
               "Tab. Levothyroxine 50 mcg before breakfast", "Tab. Paracetamol 650 mg three times daily"],
         care=["Walk with a walker under supervision; physiotherapy daily",
               "Hip precautions: do not cross legs or bend the hip beyond 90 degrees",
               "Use a raised toilet seat", "Fall prevention at home: night lights, non-slip mats"],
         follow_up="Orthopaedics OPD after 2 weeks for wound check", payer="CGHS"),
    dict(key="tb", dept="Pulmonology", sex="M", age=41, los=5, room="Isolation Room",
         diagnosis="Sputum-positive pulmonary tuberculosis (drug-sensitive)",
         complaint="Cough for 6 weeks, evening fever, night sweats and weight loss of 7 kg.",
         history="Daily wage worker. Contact with a family member treated for TB last year.",
         exam="Thin built, BMI 17.2. Crepitations in the right upper zone.",
         investigations="Sputum CBNAAT: MTB detected, rifampicin resistance not detected. Chest X-ray: right upper lobe cavity. HIV non-reactive.",
         plan="Started on 4-drug fixed-dose combination (HRZE) by weight band. Notified on the Ni-kshay portal. Airborne isolation until sputum conversion.",
         procedure="Anti-tubercular therapy initiation and isolation", procedure_cost=9000,
         meds=["FDC HRZE 4 tablets daily on an empty stomach for 2 months, then HRE for 4 months",
               "Tab. Pyridoxine 10 mg daily"],
         care=["Wear a mask at home and cover the mouth when coughing",
               "Take medicines daily under DOTS supervision; do not miss doses",
               "Report yellow eyes, vomiting or joint pain immediately",
               "Household contacts to be screened at the nearest TB unit"],
         follow_up="DOTS centre weekly; Pulmonology OPD at 2 months with sputum smear", payer="Free (National TB Elimination Programme)"),
    dict(key="preeclampsia", dept="Obstetrics & Gynaecology", sex="F", age=29, los=5, room="Private Room",
         diagnosis="Severe pre-eclampsia at 36 weeks gestation; delivered by emergency LSCS",
         complaint="Headache, blurring of vision and swelling of feet at 36 weeks of pregnancy.",
         history="Primigravida. Antenatal visits regular until 32 weeks.",
         exam="BP 168/112 mmHg, pedal oedema grade 2, brisk reflexes. Fetal heart rate 142/min.",
         investigations="Urine protein 3+. Platelets 1.1 lakh/cumm. LFT: AST 86 U/L. NST reactive.",
         plan="IV labetalol for BP control, magnesium sulphate (Pritchard regimen) for seizure prophylaxis. Emergency lower segment caesarean section.",
         procedure="Emergency lower segment caesarean section (LSCS)", procedure_cost=95000,
         meds=["Tab. Labetalol 100 mg twice daily", "Tab. Iron and folic acid once daily",
               "Tab. Paracetamol 650 mg as needed for pain"],
         care=["Check BP twice daily and record it", "Exclusive breastfeeding advised",
               "Keep the wound clean and dry", "Report headache, visual disturbance or BP above 150/100 immediately"],
         follow_up="OBG OPD after 1 week for BP check and wound review", payer="ICICI Lombard (cashless)"),
    dict(key="gastro_child", dept="Paediatrics", sex="M", age=3, los=3, room="Paediatric Ward",
         diagnosis="Acute gastroenteritis with moderate dehydration",
         complaint="Loose stools 10 times a day and vomiting for 2 days, reduced urine output.",
         history="Fully immunised as per schedule. Attends day care.",
         exam="Irritable, sunken eyes, skin pinch goes back slowly. Weight 13 kg.",
         investigations="Serum sodium 132 mmol/L, potassium 3.2 mmol/L. Stool rotavirus antigen positive.",
         plan="IV Ringer lactate 20 mL/kg bolus, then ORS. Zinc 20 mg daily for 14 days. No antibiotics needed.",
         procedure="IV rehydration therapy", procedure_cost=7000,
         meds=["ORS after every loose stool", "Zinc syrup 20 mg once daily for 14 days"],
         care=["Continue breastfeeding and normal diet", "Wash hands after diaper changes",
               "Return if the child is lethargic, not drinking, or has blood in stool"],
         follow_up="Paediatrics OPD after 3 days", payer="Self-pay"),
    dict(key="ckd", dept="Nephrology", sex="M", age=62, los=6, room="HDU",
         diagnosis="Chronic kidney disease stage 5 with hyperkalaemia; haemodialysis initiated",
         complaint="Breathlessness, reduced urine output and generalised swelling for 1 week.",
         history="Diabetic nephropathy, CKD stage 4 for 2 years. Not on dialysis.",
         exam="BP 170/100 mmHg, bilateral crepitations, pedal oedema grade 3.",
         investigations="Creatinine 9.8 mg/dL, urea 210 mg/dL, potassium 6.9 mmol/L. ECG: tall tented T waves.",
         plan="IV calcium gluconate, insulin-dextrose and salbutamol nebulisation for hyperkalaemia. Right internal jugular dialysis catheter inserted; 3 sessions of haemodialysis.",
         procedure="Haemodialysis (3 sessions) with temporary catheter", procedure_cost=36000,
         meds=["Tab. Sevelamer 800 mg with meals", "Inj. Erythropoietin 4000 units twice weekly",
               "Tab. Amlodipine 10 mg once daily", "Tab. Sodium bicarbonate 500 mg twice daily"],
         care=["Fluid restriction to 1 litre a day", "Low-potassium diet: avoid bananas, coconut water and tomatoes",
               "Keep the dialysis catheter site dry and covered", "Maintenance dialysis three times a week"],
         follow_up="Nephrology OPD in 1 week; AV fistula surgery to be planned", payer="Aarogyasri (Telangana State scheme)"),
    dict(key="stroke", dept="Neurology", sex="F", age=66, los=7, room="Neuro ICU",
         diagnosis="Acute ischaemic stroke, left middle cerebral artery territory",
         complaint="Sudden weakness of the right arm and leg with slurred speech, onset 90 minutes before arrival.",
         history="Atrial fibrillation not on anticoagulation, hypertension.",
         exam="NIHSS 11. Right hemiparesis power 2/5, expressive aphasia.",
         investigations="Non-contrast CT brain: no haemorrhage. CT angiography: left M1 occlusion. ECG: atrial fibrillation.",
         plan="IV alteplase 0.9 mg/kg given at 110 minutes from onset, followed by mechanical thrombectomy. Started apixaban on day 5.",
         procedure="IV thrombolysis and mechanical thrombectomy", procedure_cost=320000,
         meds=["Tab. Apixaban 5 mg twice daily", "Tab. Atorvastatin 40 mg at night", "Tab. Telmisartan 40 mg once daily"],
         care=["Daily physiotherapy and speech therapy", "Swallow assessment before oral feeds; soft diet",
               "Turn every 2 hours to prevent pressure sores", "Watch for bleeding while on apixaban"],
         follow_up="Neurology OPD after 2 weeks", payer="Star Health Insurance (cashless)"),
    dict(key="ugib", dept="Gastroenterology", sex="M", age=49, los=4, room="Semi-Private Ward",
         diagnosis="Upper gastrointestinal bleed due to duodenal ulcer (Forrest IIa)",
         complaint="Two episodes of vomiting blood and black tarry stools since morning.",
         history="Chronic use of diclofenac for back pain. Alcohol use, 3 units daily.",
         exam="Pallor, HR 112/min, BP 98/64 mmHg. Per rectal: melena.",
         investigations="Hb 7.4 g/dL. Upper GI endoscopy: 1.5 cm duodenal ulcer with visible vessel. H. pylori rapid urease test positive.",
         plan="IV pantoprazole 80 mg bolus then infusion. Endoscopic clipping and adrenaline injection. 2 units packed red cells transfused.",
         procedure="Therapeutic upper GI endoscopy with haemoclips", procedure_cost=42000,
         meds=["Tab. Pantoprazole 40 mg twice daily for 8 weeks",
               "H. pylori eradication: amoxicillin, clarithromycin and pantoprazole for 14 days"],
         care=["Stop diclofenac and all painkillers except paracetamol", "Avoid alcohol completely",
               "Report black stools or vomiting blood immediately"],
         follow_up="Gastroenterology OPD after 4 weeks; repeat endoscopy at 8 weeks", payer="Self-pay"),
    dict(key="asthma", dept="Pulmonology", sex="F", age=19, los=2, room="General Ward",
         diagnosis="Acute severe exacerbation of bronchial asthma",
         complaint="Wheezing and breathlessness since night, unable to complete sentences.",
         history="Asthma since childhood, poor inhaler adherence. Triggered by dust exposure.",
         exam="RR 30/min, SpO2 91%, diffuse wheeze. Peak flow 38% of predicted.",
         investigations="ABG: pH 7.44, pCO2 34 mmHg. Chest X-ray: hyperinflation, no pneumothorax.",
         plan="Nebulised salbutamol and ipratropium every 20 minutes for 1 hour, IV hydrocortisone 100 mg, then oral prednisolone 40 mg for 5 days.",
         procedure="Nebulisation and systemic steroids", procedure_cost=6000,
         meds=["Budesonide-formoterol 200/6 inhaler 2 puffs twice daily", "Salbutamol inhaler as needed",
               "Tab. Prednisolone 40 mg once daily for 3 more days"],
         care=["Inhaler technique demonstrated with spacer", "Avoid dust and smoke exposure",
               "Follow the written asthma action plan"],
         follow_up="Pulmonology OPD after 2 weeks with spirometry", payer="Self-pay"),
    dict(key="typhoid", dept="Internal Medicine", sex="F", age=31, los=5, room="General Ward",
         diagnosis="Enteric fever (typhoid), blood culture positive",
         complaint="Step-ladder fever for 8 days with abdominal discomfort and loss of appetite.",
         history="Recently ate street food. No travel outside Hyderabad.",
         exam="Temperature 103 F, relative bradycardia, coated tongue, mild splenomegaly.",
         investigations="Blood culture: Salmonella Typhi sensitive to ceftriaxone and azithromycin. Widal not relied upon.",
         plan="IV ceftriaxone 2 g twice daily, antipyretics, hydration. Switched to oral azithromycin after defervescence.",
         procedure="IV antibiotic therapy", procedure_cost=12000,
         meds=["Tab. Azithromycin 500 mg once daily for 7 days", "Tab. Paracetamol 650 mg as needed"],
         care=["Drink boiled or filtered water only", "Hand washing before meals and after using the toilet",
               "Complete the full antibiotic course"],
         follow_up="Medicine OPD after 1 week", payer="Star Health Insurance (cashless)"),
]

ROOM_RATES = {"ICU / CCU": 12000, "HDU": 8000, "Neuro ICU": 12000, "General Ward": 2500,
              "Semi-Private Ward": 4500, "Private Room": 7000, "Isolation Room": 5000, "Paediatric Ward": 3000}

CASE_NOTE_IDX = list(range(12))
DISCHARGE_IDX = [0, 1, 2, 3, 4, 5, 6, 7, 12, 13]
BILLING_IDX = [0, 1, 2, 4, 5, 7, 8, 9, 12, 13]


# --- Patient records ------------------------------------------------------


def make_patient(sc: dict, doctors: dict) -> dict:
    first, last = person(sc["sex"])
    att_first, att_last = person(rng.choice("MF"))
    if sc["dept"] == "Paediatrics":
        att_last = last  # parent shares the surname
    admit = date(2026, 1, 5) + timedelta(days=rng.randint(0, 240))
    bank, acct, ifsc = bank_account()
    return dict(
        sc=sc, first=first, last=last, name=f"{first} {last}",
        honorific=("Mr." if sc["sex"] == "M" else "Ms.") if sc["age"] >= 18 else "Master",
        uhid=uhid(), aadhaar=aadhaar(), pan=pan(last), phone=phone(), email=email(first, last),
        address=address(), attendant=f"{att_first} {att_last}", attendant_phone=phone(),
        admit=admit, discharge=admit + timedelta(days=sc["los"]), doctor=doctors[sc["dept"]],
        bank=bank, account=acct, ifsc=ifsc,
    )


def case_note(p: dict, pl: Planted) -> str:
    sc = p["sc"]
    sex = "Male" if sc["sex"] == "M" else "Female"
    ref = f"{p['honorific']} {p['last']}" if sc["age"] >= 18 else p["first"]
    return f"""{HOSPITAL}
CLINICAL CASE NOTE - Department of {sc['dept']}

Patient Name: {pl('PERSON', p['name'])}
Age / Sex: {sc['age']} years / {sex}
UHID: {pl('PATIENT_ID', p['uhid'])}
Aadhaar No.: {pl('AADHAAR', p['aadhaar'])}
Contact: {pl('PHONE', p['phone'])}
Address: {pl('ADDRESS', p['address'])}
Attendant: {pl('PERSON', p['attendant'])} ({pl('PHONE', p['attendant_phone'])})
Date of Admission: {p['admit']:%d %b %Y}
Consultant: Dr. {pl('PERSON', p['doctor'])}

Presenting Complaint:
{sc['complaint']}

History:
{sc['history']}

Examination:
{sc['exam']}

Investigations:
{sc['investigations']}

Assessment:
{sc['diagnosis']}.

Plan:
{sc['plan']}

Progress:
{pl('PERSON', ref)} was monitored closely in the {sc['room']}. The treating team discussed the diagnosis,
treatment plan and expected length of stay of about {sc['los']} days with the attendant, who gave informed consent.

Signed: Dr. {p['doctor']}, Consultant, {sc['dept']}
"""


def discharge_summary(p: dict, pl: Planted, nurse: str) -> str:
    sc = p["sc"]
    sex = "Male" if sc["sex"] == "M" else "Female"
    meds = "\n".join(f"  {i}. {m}" for i, m in enumerate(sc["meds"], 1))
    care = "\n".join(f"  - {c}" for c in sc["care"])
    return f"""{HOSPITAL}
DISCHARGE SUMMARY - Department of {sc['dept']}

Patient Name: {pl('PERSON', p['name'])}    Age / Sex: {sc['age']} / {sex}
UHID: {pl('PATIENT_ID', p['uhid'])}
Phone: {pl('PHONE', p['phone'])}    Email: {pl('EMAIL', p['email'])}
Address: {pl('ADDRESS', p['address'])}
Admitted: {p['admit']:%d %b %Y}    Discharged: {p['discharge']:%d %b %Y}    Ward: {sc['room']}
Treating Consultant: Dr. {pl('PERSON', p['doctor'])}

Final Diagnosis:
{sc['diagnosis']}

Procedure / Treatment:
{sc['procedure']}

Hospital Course:
{sc['plan']} The patient improved steadily and was discharged in stable condition on day {sc['los']}.

Discharge Medications:
{meds}

Nursing Care and Home Instructions:
{care}

Follow-up: {sc['follow_up']}.
Discharge counselling given by Staff Nurse {pl('PERSON', nurse)} to the patient and attendant
{pl('PERSON', p['attendant'])}, contact {pl('PHONE', p['attendant_phone'])}.
"""


def billing_record(p: dict, pl: Planted, clerk: str, n: int) -> str:
    sc = p["sc"]
    room_rate = ROOM_RATES[sc["room"]]
    items = [
        (f"Room charges - {sc['room']} ({sc['los']} days x {inr(room_rate)})", room_rate * sc["los"]),
        (f"Procedure - {sc['procedure']}", sc["procedure_cost"]),
        ("Consultation and doctor visit fees", 1500 * sc["los"]),
        ("Pharmacy and consumables", rng.randint(6, 40) * 1000 + rng.randint(0, 9) * 100),
        ("Laboratory and radiology", rng.randint(4, 25) * 1000 + rng.randint(0, 9) * 50),
        ("Nursing charges", 800 * sc["los"]),
    ]
    gross = sum(a for _, a in items)
    discount = round(gross * 0.05) if "Self-pay" in sc["payer"] else 0
    # Matches the billing policy: deposits only for self-pay and reimbursement admissions.
    pays_upfront = "Self-pay" in sc["payer"] or "reimbursement" in sc["payer"]
    deposit = (50000 if "ICU" in sc["room"] else 20000) if pays_upfront else 0
    covered = 0 if pays_upfront else gross - discount
    balance = gross - discount - covered - deposit
    lines = "\n".join(f"  {desc:<70} {inr(amt):>16}" for desc, amt in items)
    refund = ""
    if balance < 0:
        refund = f"\nRefund due to patient: {inr(-balance)} via NEFT within 7 working days."
    return f"""{HOSPITAL}
FINAL BILL / INVOICE No. DSH/INV/2026/{n:05d}

Patient Name: {pl('PERSON', p['name'])}
UHID: {pl('PATIENT_ID', p['uhid'])}
Aadhaar (KYC): {pl('AADHAAR', p['aadhaar'])}    PAN: {pl('PAN', p['pan'])}
Billing Address: {pl('ADDRESS', p['address'])}
Mobile: {pl('PHONE', p['phone'])}    Email: {pl('EMAIL', p['email'])}
Department: {sc['dept']}    Admission: {p['admit']:%d %b %Y} to {p['discharge']:%d %b %Y}
Payer: {sc['payer']}

Itemised Charges:
{lines}

Gross Amount:                {inr(gross)}
Discount (self-pay 5%):      {inr(discount)}
Covered by payer:            {inr(covered)}
Advance deposit received:    {inr(deposit)}
Balance payable by patient:  {inr(max(balance, 0))}{refund}

Payment / Refund Bank Details (patient):
  Account Holder: {pl('PERSON', p['name'])}
  Bank: {p['bank']}    Account No.: {pl('BANK_ACCOUNT', p['account'])}    IFSC: {pl('IFSC', p['ifsc'])}

Prepared by: {pl('PERSON', clerk)}, Billing Desk
"""


# --- Policies (hospital-wide, readable by every role) ---------------------

def policies(pl: Planted) -> list[tuple[str, str]]:
    dpo_first, dpo_last = person("F")
    dpo = f"{dpo_first} {dpo_last}"
    dpo_mail = f"{dpo_first.lower()}.{dpo_last.lower()}@deccansunrise-hospital.example"
    dpo_phone = phone()
    return [
        ("Visitor and Attendant Policy", """Purpose: to protect patients from infection and ensure a calm healing environment.
Visiting hours for general and semi-private wards are 4:00 PM to 7:00 PM daily. Only one attendant
pass is issued per patient; the attendant must carry the pass at all times.
ICU, CCU, HDU and Neuro ICU allow visitors only from 5:00 PM to 5:30 PM, maximum two visitors, one at a time.
Children below 12 years are not permitted in ICUs or isolation rooms.
Visitors to isolation rooms (airborne precautions, e.g. tuberculosis) must wear an N95 mask provided at the nursing station.
Flowers and outside food are not allowed in ICUs. Security may refuse entry to visitors who are unwell."""),
        ("Hand Hygiene and Infection Control Policy", """All staff must follow the WHO five moments of hand hygiene: before touching a patient, before
a clean or aseptic procedure, after body fluid exposure risk, after touching a patient, and after touching patient surroundings.
Use alcohol-based hand rub for 20 to 30 seconds; wash with soap and water for 40 to 60 seconds when hands are visibly soiled
or after caring for patients with diarrhoea (e.g. rotavirus or C. difficile).
Hand hygiene compliance is audited monthly by the Infection Control Nurse; the target compliance is 90% or above.
Central line and urinary catheter bundles must be documented daily. Needle-stick injuries must be reported within 2 hours
to the Infection Control Team for post-exposure prophylaxis."""),
        ("Medication Administration and High-Alert Drugs Policy", """Nurses must follow the 7 rights: right patient, right drug, right dose, right route, right time,
right documentation and right reason. Patient identity is confirmed with two identifiers: full name and UHID.
High-alert medicines require an independent double check by two nurses before administration: insulin infusions,
heparin and enoxaparin, concentrated potassium chloride, magnesium sulphate, thrombolytics such as alteplase, and chemotherapy.
Concentrated potassium chloride is never stored in ward stock; it is issued by pharmacy as premixed bags only.
Verbal orders are allowed only in emergencies and must be countersigned by the doctor within 24 hours.
Any medication error or near miss must be reported in the incident reporting system within 24 hours; reporting is non-punitive."""),
        ("Patient Data Privacy and Confidentiality Policy", f"""This policy implements the Digital Personal Data Protection Act, 2023 for all patient records.
Access to patient records is role-based: doctors may access clinical case notes and discharge summaries of patients;
nurses may access discharge summaries and care instructions; billing staff may access billing records only;
administrative staff may access hospital policies only.
Personal data such as Aadhaar numbers, phone numbers, addresses and bank details must never be shared over messaging apps,
personal email or printed and left unattended. Aadhaar numbers must be masked to show only the last four digits on printed documents.
Every access to the AI knowledge assistant is logged and audited monthly.
Data breaches must be reported to the Data Protection Officer within 6 hours.
Data Protection Officer: {pl('PERSON', dpo)}, email {pl('EMAIL', dpo_mail)}, phone {pl('PHONE', dpo_phone)}."""),
        ("Discharge Process Policy", """Discharge planning starts at admission. The treating consultant marks the planned discharge
date in the hospital information system at least 24 hours in advance.
On the day of discharge, the discharge order should be written by 10:00 AM. The discharge summary must be completed
and signed by the treating doctor within 4 hours of the discharge order.
The nurse provides discharge counselling covering medications, wound care, diet, warning signs and follow-up,
and documents that the patient or attendant understood the instructions.
Billing clearance must be completed within 2 hours for self-pay patients and within 4 hours for cashless insurance patients.
Patients leaving against medical advice (LAMA) must sign the LAMA form in the presence of a witness."""),
        ("Billing, Deposits, Refunds and Insurance Pre-authorisation Policy", """An advance deposit of Rs. 20,000 is collected for self-pay and reimbursement admissions
to wards, and Rs. 50,000 for ICU admissions. No deposit is collected for cashless insurance, CGHS, Aarogyasri or
National TB Elimination Programme patients.
For cashless insurance, the billing desk must send the pre-authorisation request to the TPA within 2 hours of admission
and an enhancement request when the estimated cost exceeds the approved amount.
A 5% discount is offered to self-pay patients on the final bill. PAN is mandatory for cash payments above Rs. 2,00,000.
Refunds of excess deposit are processed by NEFT to the patient's bank account within 7 working days of discharge,
after verifying the account holder name with the bank details form.
Billing disputes are escalated to the Billing Manager and must be resolved within 3 working days."""),
        ("Code Blue and Medical Emergency Response Policy", """A Code Blue is called for cardiac or respiratory arrest anywhere in the hospital by dialling extension 2222
and stating 'Code Blue' with the location. The Code Blue team (ICU doctor, anaesthetist, two ICU nurses and a
respiratory therapist) must reach any location within 3 minutes.
High-quality CPR follows the AHA ACLS guidelines: compression rate 100 to 120 per minute, depth 5 to 6 cm,
minimal interruptions, and defibrillation as early as possible for shockable rhythms.
The Rapid Response Team is called earlier, when the Early Warning Score is 5 or above, for example sudden drop in BP,
SpO2 below 90% or new confusion. Crash carts are checked every shift and after every use, and the check is documented."""),
        ("Fall Prevention and Patient Identification Policy", """Every inpatient is assessed with the Morse Fall Scale at admission, every shift and after any change in condition.
Patients scoring 45 or above are high fall risk: they get a yellow wristband, bed in the lowest position, bed rails up,
call bell within reach, non-slip footwear and supervised mobilisation.
Elderly patients after hip surgery and patients on sedatives are treated as high fall risk by default.
Every fall must be reported within 24 hours and reviewed by the Nursing Superintendent.
Patient identification uses two identifiers, full name and UHID, checked against the wristband before giving medicines,
blood transfusion, sample collection or any procedure. Bed number must never be used as an identifier."""),
    ]


# --- Main -----------------------------------------------------------------


def main() -> None:
    if RAW_DIR.exists():
        shutil.rmtree(RAW_DIR)
    RAW_DIR.mkdir(parents=True)

    depts = sorted({s["dept"] for s in SCENARIOS})
    doctors = {d: " ".join(person(rng.choice("MF"))) for d in depts}
    nurses = [" ".join(person("F")) for _ in range(5)]
    clerks = [" ".join(person(rng.choice("MF"))) for _ in range(3)]
    patients = [make_patient(sc, doctors) for sc in SCENARIOS]

    manifest, truth = [], {}

    def save(doc_id: str, category: str, title: str, text: str, pl: Planted, dept: str = "Hospital-wide"):
        filename = f"{doc_id}.txt"
        (RAW_DIR / filename).write_text(text, encoding="utf-8")
        manifest.append(dict(doc_id=doc_id, file=filename, category=category, title=title,
                             department=dept, allowed_roles=CATEGORY_ROLES[category]))
        truth[doc_id] = pl.items

    for n, i in enumerate(CASE_NOTE_IDX, 1):
        p, pl = patients[i], Planted()
        save(f"CN-{n:03d}", "case_note", f"Case note - {p['sc']['dept']} - {p['sc']['diagnosis'].split(' (')[0]}",
             case_note(p, pl), pl, p["sc"]["dept"])
    for n, i in enumerate(DISCHARGE_IDX, 1):
        p, pl = patients[i], Planted()
        save(f"DS-{n:03d}", "discharge_summary", f"Discharge summary - {p['sc']['dept']} - {p['sc']['diagnosis'].split(' (')[0]}",
             discharge_summary(p, pl, rng.choice(nurses)), pl, p["sc"]["dept"])
    for n, i in enumerate(BILLING_IDX, 1):
        p, pl = patients[i], Planted()
        save(f"BL-{n:03d}", "billing", f"Final bill - {p['sc']['dept']} - {p['sc']['procedure']}",
             billing_record(p, pl, rng.choice(clerks), n), pl, p["sc"]["dept"])
    pol_pl = Planted()
    for n, (title, body) in enumerate(policies(pol_pl), 1):
        pl = Planted()
        if "Privacy" in title:
            pl.items = pol_pl.items
        text = f"{HOSPITAL}\nHOSPITAL POLICY POL-{n:03d}: {title}\nEffective: 01 Apr 2026    Review: 31 Mar 2027\n\n{body}\n"
        save(f"POL-{n:03d}", "policy", f"Policy - {title}", text, pl)

    MANIFEST_PATH.write_text(json.dumps({"hospital": HOSPITAL, "documents": manifest}, indent=2), encoding="utf-8")
    PII_TRUTH_PATH.write_text(json.dumps(truth, indent=2, ensure_ascii=False), encoding="utf-8")

    counts = {}
    for m in manifest:
        counts[m["category"]] = counts.get(m["category"], 0) + 1
    n_pii = sum(len(v) for v in truth.values())
    print(f"Wrote {len(manifest)} documents to {RAW_DIR.relative_to(DATA_DIR.parent)}")
    for cat, c in counts.items():
        print(f"  {cat:<18} {c:>3}  roles: {', '.join(CATEGORY_ROLES[cat])}")
    print(f"Planted {n_pii} personal-data values -> {PII_TRUTH_PATH.name}")


if __name__ == "__main__":
    main()
