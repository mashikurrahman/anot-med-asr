#!/usr/bin/env python3
"""
Medical-term sentence generator for synthetic-speech training data.

Real recordings contain too few distinct drugs and conditions for the engine to
learn them (e.g. "albuterol" was transcribed as "albedo"). This builds a large
bank of terms - the repo's medical lexicon plus drugs and conditions typical of
allergy / asthma / urgent-care / primary-care visits - and wraps every term in
natural clinic-style sentences (clinician and patient phrasing). Doses use digits
with spelled-out units ("20 milligrams"), matching how the human references
are written. Deterministic for a given seed. No patient data is involved.
"""

import random
from typing import Dict, List

from language_model.medical_lexicon import MedicalLexicon

# Lexicon entries that are fragments of longer terms; replaced by full phrases.
_FRAGMENTS = {
    "artery", "atrial", "chronic", "congestive", "disease", "failure", "kidney", "mellitus",
    "flutter", "fibrillation", "coronary", "renal", "breath", "shortness", "abdominal", "chest",
    "computed", "magnetic", "resonance", "tomography", "aspiration", "x-ray", "xray",
    "anterior", "posterior", "superior", "inferior", "medial", "lateral", "proximal", "distal",
    "bilateral", "bilaterally", "cranial", "sensitivity", "culture",
}

EXTRA: Dict[str, List[str]] = {
    "medications": [
        "solu-medrol", "medrol dosepak", "z-pak", "tessalon", "xopenex", "levalbuterol", "atrovent",
        "airsupra", "breztri", "wixela", "armonair", "alvesco", "rhinocort", "astepro", "zyrtec-d",
        "allegra-d", "tylenol", "motrin", "advil", "aleve", "epipen", "auvi-q", "pepcid", "famotidine",
        "ondansetron", "zofran", "promethazine", "phenergan", "methocarbamol", "flexeril", "toradol",
        "ketorolac", "ceftriaxone", "rocephin", "penicillin", "amoxicillin-clavulanate", "bactrim ds",
        "cefixime", "tamiflu", "paxlovid", "xofluza", "valtrex", "sumatriptan", "imitrex", "lidocaine",
        "tezspire", "tezepelumab", "cetirizine hydrochloride", "prednisone taper", "dexamethasone",
        "azithromycin", "robitussin", "delsym", "claritin-d", "saline nasal spray", "neti pot",
        "afrin", "oxymetazoline", "ipratropium bromide", "cromolyn sodium", "benzonatate",
        "hydrocodone", "oxycodone", "acetaminophen with codeine", "naloxone", "lasix", "coumadin",
        "eliquis", "xarelto", "plavix", "lipitor", "crestor", "norvasc", "zestril", "cozaar", "glucophage",
        "januvia", "prilosec", "nexium", "zantac", "synthroid", "zoloft", "lexapro", "prozac",
    ],
    "diagnoses": [
        "allergic rhinitis", "sinusitis", "rhinosinusitis", "bronchospasm", "otitis media", "pharyngitis",
        "strep throat", "conjunctivitis", "urticaria", "angioedema", "anaphylaxis", "eczema",
        "atopic dermatitis", "urinary tract infection", "cystitis", "gastroenteritis", "sprain",
        "laceration", "contusion", "tendinitis", "bursitis", "sciatica", "nasal polyps",
        "deviated septum", "turbinate hypertrophy", "post nasal drip", "eosinophilic asthma",
        "coronary artery disease", "congestive heart failure", "atrial fibrillation",
        "type 2 diabetes mellitus", "chronic kidney disease", "chronic obstructive pulmonary disease",
        "gastroesophageal reflux disease", "upper respiratory infection", "viral syndrome",
        "bronchiolitis", "croup", "pertussis", "influenza", "covid", "mononucleosis", "shingles",
        "food allergy", "drug allergy", "seasonal allergies", "reactive airway disease",
    ],
    "symptoms": [
        "shortness of breath", "chest pain", "chest tightness", "abdominal pain", "sneezing",
        "nasal congestion", "runny nose", "itchy eyes", "sore throat", "earache", "post nasal drainage",
        "loss of smell", "sinus pressure", "facial pain", "night sweats", "hives", "itching",
        "difficulty breathing", "lightheadedness", "body aches", "loss of appetite",
    ],
    "procedures": [
        "ct scan", "chest x-ray", "skin prick test", "peak flow", "pulse oximetry", "rapid strep test",
        "flu swab", "covid test", "pcr test", "nasal endoscopy", "allergy testing", "patch test",
        "pulmonary function test", "nebulizer treatment", "spirometry",
    ],
    "lab_tests": [
        "ige level", "eosinophil count", "cbc", "complete blood count", "cmp",
        "comprehensive metabolic panel", "fev1", "oxygen saturation", "blood pressure",
    ],
}

_DOSES = ["5 milligrams", "10 milligrams", "20 milligrams", "40 milligrams", "50 milligrams",
          "100 milligrams", "250 milligrams", "500 milligrams", "1 tablet", "2 tablets",
          "2 puffs", "1 puff", "1 teaspoon", "10 milliliters", "1 spray in each nostril"]
_FREQS = ["once a day", "twice a day", "three times a day", "every 4 hours", "every 6 hours as needed",
          "at bedtime", "every morning", "with food", "for 5 days", "for 10 days", "as needed"]

_MED_T = [
    "I'll start you on {t}.", "Are you taking any {t}?", "She has been taking {t} for about two weeks.",
    "Take {d} of {t} {f}.", "We're going to prescribe {t}, {d}, {f}.",
    "Do you have any allergies to {t}?", "I was on {t} before, but it didn't help.",
    "Let's try {t} and see how you do.", "He ran out of {t} last week.",
    "Continue the {t} and come back in two weeks.", "The {t} gave me a rash.",
    "Have you ever used {t}?", "I'm going to send {t} to your pharmacy.",
    "My doctor put me on {t}.", "Did the {t} help with your symptoms?", "Stop the {t} if it upsets your stomach.",
]
_DX_T = [
    "You have {t}.", "Do you have a history of {t}?", "Her {t} has been well controlled.",
    "This looks like {t}.", "I'm concerned about {t}.", "He was diagnosed with {t} last year.",
    "Any family history of {t}?", "We'll treat this as {t}.", "I think it's {t}.",
]
_SX_T = [
    "Any {t}?", "She's been having {t} for three days.", "Are you experiencing {t}?",
    "The {t} gets worse at night.", "I've had {t} since Monday.", "Is the {t} constant or does it come and go?",
    "He denies any {t}.", "Have you noticed any {t}?",
]
_PROC_T = [
    "We'll order a {t} today.", "You had a {t} last year, right?", "I'd like to get a {t} to check.",
    "The {t} was normal.", "Let's schedule a {t} for next week.", "Have you ever had a {t}?",
]
_LAB_T = [
    "Your {t} came back normal.", "Let's check your {t} again.", "We'll recheck the {t} in a month.",
    "Your {t} is a little high.", "I'd like to see your {t}.",
]
_ANAT_T = [
    "I'm listening to your {t}.", "There's some tenderness over the {t}.", "Let me check your {t}.",
    "Does it hurt when I press on the {t}?", "The {t} looks fine.",
]
_AGES = [str(a) for a in (7, 12, 24, 31, 38, 45, 52, 58, 63, 71, 79)]
_COMPOSITE_T = [
    "The patient is a {a}-year-old {sex} with {dx}, currently on {m1}, {d}, {f}, and {m2}.",
    "She is here for {sx} and {sx2}, and she takes {m1} {f}.",
    "Assessment is {dx}. Plan is {m1}, {d}, {f}, and follow up in 2 weeks.",
    "He has {dx} and {dx2}. I'm adding {m1} and continuing {m2}.",
    "On exam, no {sx}. We'll start {m1}, {d}, {f}, and get a {pr}.",
    "I take {m1} {f} and {m2} for my {dx}.",
]


def build_term_bank() -> Dict[str, List[str]]:
    """Lexicon terms (minus fragments) plus the extra clinic vocabulary, de-duplicated."""
    lex = MedicalLexicon().lexicon
    bank: Dict[str, List[str]] = {}
    for cat in ("medications", "diagnoses", "symptoms", "procedures", "anatomy", "lab_tests"):
        base = [w for w in lex.get(cat, []) if w.lower() not in _FRAGMENTS]
        seen, merged = set(), []
        for w in base + EXTRA.get(cat, []):
            key = w.strip().lower()
            if key and key not in seen:
                seen.add(key)
                merged.append(w.strip())
        bank[cat] = merged
    return bank


def _sentences_for(terms: List[str], templates: List[str], per_term: int, rng: random.Random,
                   with_dose: bool = False) -> List[str]:
    out = []
    for t in terms:
        for tpl in rng.sample(templates, min(per_term, len(templates))):
            out.append(tpl.format(t=t, d=rng.choice(_DOSES), f=rng.choice(_FREQS)) if with_dose
                       else tpl.format(t=t))
    return out


def generate_sentences(seed: int = 0, composites: int = 500) -> List[str]:
    """Return a shuffled, de-duplicated list of clinic-style sentences covering every term."""
    rng = random.Random(seed)
    bank = build_term_bank()
    sents: List[str] = []
    sents += _sentences_for(bank["medications"], _MED_T, 3, rng, with_dose=True)
    sents += _sentences_for(bank["diagnoses"], _DX_T, 2, rng)
    sents += _sentences_for(bank["symptoms"], _SX_T, 2, rng)
    sents += _sentences_for(bank["procedures"], _PROC_T, 2, rng)
    sents += _sentences_for(bank["lab_tests"], _LAB_T, 2, rng)
    sents += _sentences_for(bank["anatomy"], _ANAT_T, 2, rng)
    for _ in range(composites):
        tpl = rng.choice(_COMPOSITE_T)
        sents.append(tpl.format(
            a=rng.choice(_AGES), sex=rng.choice(["male", "female"]),
            dx=rng.choice(bank["diagnoses"]), dx2=rng.choice(bank["diagnoses"]),
            m1=rng.choice(bank["medications"]), m2=rng.choice(bank["medications"]),
            sx=rng.choice(bank["symptoms"]), sx2=rng.choice(bank["symptoms"]),
            pr=rng.choice(bank["procedures"]), d=rng.choice(_DOSES), f=rng.choice(_FREQS)))
    sents = list(dict.fromkeys(sents))
    rng.shuffle(sents)
    return sents
