"""
Target Configurations for BioActive-MultiTarget.

Each target has:
    - chembl_id: ChEMBL target identifier
    - name: Human-readable target name
    - short_name: Short identifier for file naming
    - activity_types: Activity measurement types to extract
    - activity_threshold_nM: Cutoff for active/inactive classification
    - description: Brief description of target role
    - benchmark_compounds: Curated panel for biological validation
"""

# ── Shared unrelated compounds (no known activity against any target) ───────────
UNRELATED_COMPOUNDS = [
    {
        "name": "Aspirin",
        "smiles": "CC(=O)OC1=CC=CC=C1C(=O)O",
        "category": "unrelated",
        "notes": "NSAID, no known activity against target",
    },
    {
        "name": "Amoxicillin",
        "smiles": "CC1(C)SC2C(NC(=O)C(N)C3=CC=C(O)C=C3)C(=O)N2C1C(=O)O",
        "category": "unrelated",
        "notes": "Antibiotic, no known activity against target",
    },
    {
        "name": "Metformin",
        "smiles": "CN(C)C(=N)NC(=N)N",
        "category": "unrelated",
        "notes": "Antidiabetic (AMPK), no direct enzyme inhibition",
    },
    {
        "name": "Ibuprofen",
        "smiles": "CC(C)CC1=CC=C(C=C1)C(C)C(=O)O",
        "category": "unrelated",
        "notes": "NSAID, no known activity against target",
    },
    {
        "name": "Caffeine",
        "smiles": "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",
        "category": "unrelated",
        "notes": "Stimulant, no known activity against target",
    },
]


# ══════════════════════════════════════════════════════════════════════════════════
# PHASE 1 — Diabetes Targets
# ══════════════════════════════════════════════════════════════════════════════════

AMYLASE_CONFIG = {
    "chembl_id": ["CHEMBL2045", "CHEMBL2478", "CHEMBL6066863", "CHEMBL5730"],
    "name": "amylase",
    "display_name": "Alpha-Amylase",
    "activity_types": ["IC50", "Ki"],
    "activity_threshold_nM": 10_000,
    "description": "Pancreatic Alpha-Amylase — diabetes target (starch digestion, pooled human+porcine)",
    "benchmark_compounds": [
        # ── Potent inhibitors (from ChEMBL active compounds, IC50 < 500 nM) ──
        {
            "name": "CHEMBL5624535 (rhodanine-gallate)",
            "smiles": "O=C(Nc1ccc(/C=C2\\SC(=O)N(C(=O)CN3C(=O)c4ccccc4C3=O)C2=O)cc1)c1cc(O)c(O)c(O)c1",
            "category": "potent",
            "notes": "ChEMBL verified, IC50 190 nM",
        },
        {
            "name": "CHEMBL5624476 (rhodanine-sulfonyl)",
            "smiles": "O=C(Nc1ccc(/C=C2\\SC(=O)N(C(=O)CN3CCN(S(=O)(=O)c4ccccc4)CC3)C2=O)cc1)c1cc(O)c(O)c(O)c1",
            "category": "potent",
            "notes": "ChEMBL verified, IC50 201 nM",
        },
        {
            "name": "CHEMBL5624478 (rhodanine-morpholine)",
            "smiles": "O=C(Nc1ccc(/C=C2\\SC(=O)N(C(=O)CN3CCOCC3)C2=O)cc1)c1cc(O)c(O)c(O)c1",
            "category": "potent",
            "notes": "ChEMBL verified, IC50 280 nM",
        },
        {
            "name": "CHEMBL5596871 (azo-sulfonamide)",
            "smiles": "CC(=O)Nc1ccc(O)c(/N=N/c2ccc(S(=O)(=O)Nc3nccs3)cc2)c1",
            "category": "potent",
            "notes": "ChEMBL verified, IC50 8350 nM (active below threshold)",
        },
        # ── Weak inhibitors (borderline, IC50 just above 10000 nM threshold) ──
        {
            "name": "CHEMBL5590296 (triazole-rhodanine)",
            "smiles": "Cc1cccc(-n2cc(CN3C(=O)S/C(=C\\N(C)C)C3=O)nn2)c1",
            "category": "weak",
            "notes": "IC50 30530 nM, model P=0.21",
        },
        {
            "name": "CHEMBL5594825 (triazole-rhodanine)",
            "smiles": "Cc1cc(C)cc(-n2cc(CN3C(=O)S/C(=C\\N(C)C)C3=O)nn2)c1",
            "category": "weak",
            "notes": "IC50 33033 nM, model P=0.21",
        },
        # ── Unrelated compounds (no amylase activity) ──
        {
            "name": "Aspirin",
            "smiles": "CC(=O)OC1=CC=CC=C1C(=O)O",
            "category": "unrelated",
            "notes": "NSAID, no known activity against target",
        },
        {
            "name": "Metformin",
            "smiles": "CN(C)C(=N)NC(=N)N",
            "category": "unrelated",
            "notes": "Antidiabetic (AMPK), no direct enzyme inhibition",
        },
        {
            "name": "Ibuprofen",
            "smiles": "CC(C)CC1=CC=C(C=C1)C(C)C(=O)O",
            "category": "unrelated",
            "notes": "NSAID, no known activity against target",
        },
        {
            "name": "Amlodipine",
            "smiles": "CCOC(=O)C1=C(COCCN)NC(C)=C(C1C1=CC=CC=C1Cl)C(=O)OC",
            "category": "unrelated",
            "notes": "Calcium channel blocker, no amylase activity",
        },
        {
            "name": "Omeprazole",
            "smiles": "COc1ccc2[nH]c(S(=O)Cc3ncc(C)c(OC)c3C)nc2c1",
            "category": "unrelated",
            "notes": "Proton pump inhibitor, no amylase activity",
        },
    ],
}

GLUCOSIDASE_CONFIG = {
    "chembl_id": "CHEMBL3833502",
    "name": "glucosidase",
    "display_name": "Alpha-Glucosidase",
    "activity_types": ["IC50", "Ki"],
    "activity_threshold_nM": 10_000,
    "description": "Human Intestinal Alpha-Glucosidase — diabetes target (carbohydrate digestion)",
    "benchmark_compounds": [
        # ── Potent inhibitors (drug-like compounds from ChEMBL data) ──
        {
            "name": "CHEMBL4448264 (coumarin-pyrazole)",
            "smiles": "O=c1oc2ccccc2cc1-c1cc(O)n(-c2ccc(NC(=O)Nc3ccc(Oc4ccc(NC(=O)c5cccc(Br)c5)cc4)cc3)cc2)n1",
            "category": "potent",
            "notes": "ChEMBL verified, IC50 70 nM",
        },
        {
            "name": "CHEMBL498833 (triterpenoid)",
            "smiles": "C=C(C)[C@@H]1CC[C@]2(C(=O)O)CC[C@]3(C)[C@H](CC[C@@H]4[C@@]5(C)CC[C@H](O)C(C)(C)[C@@H]5CC[C@]43C)[C@@H]2[C@H]1C",
            "category": "potent",
            "notes": "ChEMBL verified, IC50 40 nM (betulinic acid derivative)",
        },
        {
            "name": "Quercetin (flavonoid)",
            "smiles": "O=c1c(O)c(-c2ccc(O)c(O)c2)oc2cc(O)cc(O)c12",
            "category": "potent",
            "notes": "Well-established alpha-glucosidase inhibitor, IC50 ~5 uM",
        },
        {
            "name": "CHEMBL4430218 (pyrazole)",
            "smiles": "O=C(Nc1ccc(-c2ccc(NC(=O)c3cc(-c4cc5ccccc5oc4=O)nn3-c3ccccc3)cc2)cc1)c1cc(-c2cc3ccccc3oc2=O)nn1-c1ccccc1",
            "category": "potent",
            "notes": "ChEMBL verified, IC50 80 nM",
        },
        {
            "name": "Myricetin",
            "smiles": "OC1=CC(O)=C2C(=O)C(O)=C(OC2=C1)C1=CC(O)=C(O)C(O)=C1",
            "category": "potent",
            "notes": "Flavonoid, IC50 ~5 µM for alpha-glucosidase",
        },
        # ── Weak inhibitors (moderate IC50 10-100 µM, flavonoid-like) ──
        {
            "name": "Kaempferol",
            "smiles": "OC1=CC(O)=C2C(=O)C(O)=C(OC2=C1)C1=CC=C(O)C=C1",
            "category": "weak",
            "notes": "Flavonoid, IC50 ~50 µM for alpha-glucosidase",
        },
        {
            "name": "Luteolin",
            "smiles": "OC1=CC(O)=C2C(=O)C=C(OC2=C1)C1=CC(O)=C(O)C=C1",
            "category": "weak",
            "notes": "Flavonoid, IC50 ~30 µM for alpha-glucosidase",
        },
        # ── Unrelated compounds (no glucosidase activity) ──
        {
            "name": "Aspirin",
            "smiles": "CC(=O)OC1=CC=CC=C1C(=O)O",
            "category": "unrelated",
            "notes": "NSAID, no known activity against target",
        },
        {
            "name": "Metformin",
            "smiles": "CN(C)C(=N)NC(=N)N",
            "category": "unrelated",
            "notes": "Antidiabetic (AMPK), no direct enzyme inhibition",
        },
        {
            "name": "Ibuprofen",
            "smiles": "CC(C)CC1=CC=C(C=C1)C(C)C(=O)O",
            "category": "unrelated",
            "notes": "NSAID, no known activity against target",
        },
        {
            "name": "Atenolol",
            "smiles": "CC(C)NCC(O)COC1=CC=C(CC(N)=O)C=C1",
            "category": "unrelated",
            "notes": "Beta-blocker, no glucosidase activity",
        },
        {
            "name": "Simvastatin",
            "smiles": "CCC(C)(C)C(=O)OC1CC(O)C=C2C=CC(C)C(CCC3CC(O)CC(=O)O3)C21",
            "category": "unrelated",
            "notes": "Statin, no glucosidase activity",
        },
    ],
}

LIPASE_CONFIG = {
    "chembl_id": "CHEMBL1812",
    "name": "lipase",
    "display_name": "Pancreatic Lipase",
    "activity_types": ["IC50", "Ki"],
    "activity_threshold_nM": 10_000,
    "description": "Human Pancreatic Lipase — diabetes/obesity target (fat digestion)",
    "benchmark_compounds": [
        # ── Potent inhibitors ──
        {
            "name": "Orlistat",
            "smiles": "CCCCCCCCCCC[C@@H](OC(=O)[C@H](CC(C)C)NC=O)C[C@H]1OC(=O)[C@H]1CCCCCC",
            "category": "potent",
            "notes": "FDA-approved lipase inhibitor, IC50 ~12 nM",
        },
        {
            "name": "Cetilistat",
            "smiles": "CCCCCCCCCCCCCCCCOC(=O)C1=CC(=CC=C1)NC(=O)C2=CC=CC=C2",
            "category": "potent",
            "notes": "Approved lipase inhibitor, IC50 ~30 nM",
        },
        {
            "name": "Ebelactone A",
            "smiles": "CCC(/C=C/C1OC(=O)C1C)C(O)CC(/C=C/C)C(=O)CC",
            "category": "potent",
            "notes": "Natural product lipase inhibitor, IC50 ~0.8 µM",
        },
        {
            "name": "Oleic acid",
            "smiles": "CCCCCCCC/C=C\\CCCCCCCC(=O)O",
            "category": "potent",
            "notes": "Known lipase substrate/inhibitor at high conc, used as positive control",
        },
        {
            "name": "Hexadecylsulfonyl fluoride",
            "smiles": "CCCCCCCCCCCCCCCCS(=O)(=O)F",
            "category": "potent",
            "notes": "Serine hydrolase inhibitor, IC50 ~0.2 µM for pancreatic lipase",
        },
        # ── Weak inhibitors ──
        {
            "name": "EGCG",
            "smiles": "OC1=CC(O)=C2CC(OC(=O)C3=CC(O)=C(O)C(O)=C3)C(OC2=C1)C1=CC(O)=C(O)C(O)=C1",
            "category": "weak",
            "notes": "Green tea catechin, IC50 ~50 µM for lipase",
        },
        {
            "name": "Resveratrol",
            "smiles": "OC1=CC=C(/C=C/C2=CC(O)=CC(O)=C2)C=C1",
            "category": "weak",
            "notes": "Polyphenol, IC50 ~65 µM for lipase",
        },
    ] + UNRELATED_COMPOUNDS,
}


# ══════════════════════════════════════════════════════════════════════════════════
# PHASE 2 — Hypertension
# ══════════════════════════════════════════════════════════════════════════════════

ACE_CONFIG = {
    "chembl_id": "CHEMBL1808",
    "name": "ace",
    "display_name": "ACE",
    "activity_types": ["IC50", "Ki"],
    "activity_threshold_nM": 10_000,
    "description": "Angiotensin-Converting Enzyme — hypertension target",
    "benchmark_compounds": [
        # ── Potent inhibitors ──
        {
            "name": "Captopril",
            "smiles": "CC(CS)C(=O)N1CCCC1C(=O)O",
            "category": "potent",
            "notes": "FDA-approved ACE inhibitor, IC50 ~6 nM",
        },
        {
            "name": "Enalaprilat",
            "smiles": "OC(=O)C(CC1=CC=CC=C1)NC(C)C(=O)N1CCCC1C(=O)O",
            "category": "potent",
            "notes": "Active metabolite of enalapril, IC50 ~1.2 nM",
        },
        {
            "name": "Lisinopril",
            "smiles": "NCCCC[C@@H](NC(CCc1ccccc1)C(=O)O)C(=O)N1CCC[C@H]1C(=O)O",
            "category": "potent",
            "notes": "FDA-approved ACE inhibitor, IC50 ~1.2 nM",
        },
        {
            "name": "Ramiprilat",
            "smiles": "OC(=O)[C@@H]1CC2CCCC[C@H]2CN1C(=O)[C@H](NC(CCc1ccccc1)C(=O)O)C",
            "category": "potent",
            "notes": "Active metabolite of ramipril, IC50 ~0.3 nM",
        },
        {
            "name": "Perindoprilat",
            "smiles": "CCC(NC(C)C(=O)N1C2CCCC[C@@H]2C[C@@H]1C(=O)O)C(=O)O",
            "category": "potent",
            "notes": "Active metabolite of perindopril, IC50 ~1.5 nM",
        },
        # ── Weak inhibitors (confirmed moderate/weak from ChEMBL) ──
        {
            "name": "CHEMBL54677 (renin-like)",
            "smiles": "CC(C)[C@@H](O)C[C@H](O)[C@H](CC1CCCCC1)NC(=O)[C@H](Cc1c[nH]c2ccccc12)NC(=O)OCc1ccccc1",
            "category": "weak",
            "notes": "Confirmed inactive against ACE, IC50 >100 µM in ChEMBL",
        },
        {
            "name": "CHEMBL77258 (hydroxamate)",
            "smiles": "CCCCC(CC(=O)NO)S(=O)(=O)C1CCCCC1",
            "category": "weak",
            "notes": "Confirmed inactive against ACE, IC50 >100 µM in ChEMBL",
        },
    ] + UNRELATED_COMPOUNDS,
}


# ══════════════════════════════════════════════════════════════════════════════════
# PHASE 3 — Cancer Targets (1 µM threshold for kinases)
# ══════════════════════════════════════════════════════════════════════════════════

EGFR_CONFIG = {
    "chembl_id": "CHEMBL203",
    "name": "egfr",
    "display_name": "EGFR",
    "activity_types": ["IC50", "Ki"],
    "activity_threshold_nM": 1_000,
    "description": "Epidermal Growth Factor Receptor (ErbB1) — non-small cell lung cancer target",
    "benchmark_compounds": [
        # ── Potent inhibitors ──
        {
            "name": "Erlotinib",
            "smiles": "COCCOC1=CC2=C(C=C1OCCOC)C(=NC=N2)NC1=CC=CC(=C1)C#C",
            "category": "potent",
            "notes": "FDA-approved EGFR inhibitor, IC50 ~2 nM",
        },
        {
            "name": "Gefitinib",
            "smiles": "COC1=C(OCCCN2CCOCC2)C=C2C(NC3=CC(Cl)=C(F)C=C3)=NC=NC2=C1",
            "category": "potent",
            "notes": "FDA-approved EGFR inhibitor, IC50 ~33 nM",
        },
        {
            "name": "Afatinib",
            "smiles": "CN(C)C/C=C/C(=O)NC1=CC2=C(C=C1)NC(=NC2)NC1=CC(=C(F)C=C1)Cl",
            "category": "potent",
            "notes": "FDA-approved irreversible EGFR inhibitor, IC50 ~0.5 nM",
        },
        {
            "name": "Osimertinib",
            "smiles": "COC1=C(NC2=NC=CC(=N2)C2=CN(C)C3=CC=CC=C23)C=C(NC(=O)/C=C/CN(C)C)C(=C1)NC",
            "category": "potent",
            "notes": "Third-gen EGFR inhibitor (T790M mutant), IC50 ~1 nM",
        },
        {
            "name": "Lapatinib",
            "smiles": "CS(=O)(=O)CCNCC1=CC=C(O1)C1=CC=C2NC=NC(NC3=CC=C(OCC4=CC(F)=CC=C4)C(Cl)=C3)=C2C=C1",
            "category": "potent",
            "notes": "FDA-approved dual EGFR/HER2 inhibitor, IC50 ~10.8 nM for EGFR",
        },
        # ── Weak inhibitors ──
        {
            "name": "Vandetanib",
            "smiles": "COC1=CC2=C(C=C1OCC1CCN(C)CC1)N=CN=C2NC1=CC(=C(F)C=C1)Br",
            "category": "weak",
            "notes": "Primarily VEGFR2 inhibitor, moderate EGFR IC50 ~500 nM",
        },
        {
            "name": "4-Anilinoquinazoline",
            "smiles": "NC1=CC=CC=C1NC1=NC=NC2=CC=CC=C12",
            "category": "weak",
            "notes": "Basic scaffold, weak EGFR inhibitor, IC50 ~5 µM",
        },
    ] + UNRELATED_COMPOUNDS,
}

HER2_CONFIG = {
    "chembl_id": "CHEMBL1824",
    "name": "her2",
    "display_name": "HER2",
    "activity_types": ["IC50", "Ki"],
    "activity_threshold_nM": 1_000,
    "description": "Human Epidermal Growth Factor Receptor 2 (ErbB2) — breast cancer target",
    "benchmark_compounds": [
        # ── Potent inhibitors (verified in ChEMBL data for HER2) ──
        {
            "name": "Lapatinib",
            "smiles": "CS(=O)(=O)CCNCC1=CC=C(O1)C1=CC=C2NC=NC(NC3=CC=C(OCC4=CC(F)=CC=C4)C(Cl)=C3)=C2C=C1",
            "category": "potent",
            "notes": "FDA-approved dual EGFR/HER2 inhibitor, IC50 ~9.2 nM for HER2",
        },
        {
            "name": "Afatinib",
            "smiles": "CN(C)C/C=C/C(=O)NC1=CC2=C(C=C1)NC(=NC2)NC1=CC(=C(F)C=C1)Cl",
            "category": "potent",
            "notes": "Pan-ErbB inhibitor, IC50 ~14 nM for HER2",
        },
        {
            "name": "CHEMBL941 (imatinib-like)",
            "smiles": "Cc1ccc(NC(=O)c2ccc(CN3CCN(C)CC3)cc2)cc1Nc1nccc(-c2cccnc2)n1",
            "category": "potent",
            "notes": "ChEMBL verified HER2 inhibitor, IC50 0.06 nM",
        },
        {
            "name": "CHEMBL5221067 (thiophene)",
            "smiles": "N#Cc1c(NC(=O)CNc2n[nH]c3ncccc23)sc2c1CCCC2",
            "category": "potent",
            "notes": "ChEMBL verified HER2 inhibitor, IC50 0.14 nM",
        },
        {
            "name": "Neratinib",
            "smiles": "CCOC1=CC2=C(C=C1NC(=O)/C=C/CN(C)C)NC(=NC2)NC1=CC=C(OCC2=CC(CC)=CC=C2)C(Cl)=C1",
            "category": "potent",
            "notes": "FDA-approved irreversible HER2 inhibitor, IC50 ~59 nM",
        },
        # ── Weak inhibitors (confirmed high IC50 in ChEMBL) ──
        {
            "name": "CHEMBL281872 (quinazoline)",
            "smiles": "COc1cc2c(Nc3ccc(Br)cc3F)ncnc2cc1OCCn1ccnn1",
            "category": "weak",
            "notes": "ChEMBL confirmed inactive for HER2, IC50 >100 µM",
        },
        {
            "name": "CHEMBL87402 (oxindole)",
            "smiles": "O=C1Nc2ccccc2/C1=C/c1n[nH]cc1Cl",
            "category": "weak",
            "notes": "ChEMBL confirmed inactive for HER2, IC50 >100 µM",
        },
    ] + UNRELATED_COMPOUNDS,
}

VEGFR2_CONFIG = {
    "chembl_id": "CHEMBL279",
    "name": "vegfr2",
    "display_name": "VEGFR2",
    "activity_types": ["IC50", "Ki"],
    "activity_threshold_nM": 1_000,
    "description": "Vascular Endothelial Growth Factor Receptor 2 (KDR) — anti-angiogenesis cancer target",
    "benchmark_compounds": [
        # ── Potent inhibitors ──
        {
            "name": "Sorafenib",
            "smiles": "CNC(=O)C1=CC(OC2=CC=C(NC(=O)NC3=CC(=C(Cl)C=C3)C(F)(F)F)C=C2)=CC=N1",
            "category": "potent",
            "notes": "FDA-approved multi-kinase inhibitor, IC50 ~90 nM for VEGFR2",
        },
        {
            "name": "Sunitinib",
            "smiles": "CCN(CC)CCNC(=O)C1=C(C)[NH]C(/C1=C\\1/C(=O)NC2=CC(F)=CC=C21)=C",
            "category": "potent",
            "notes": "FDA-approved VEGFR2 inhibitor, IC50 ~9 nM",
        },
        {
            "name": "Axitinib",
            "smiles": "CNC(=O)C1=CC=CC=C1SC1=CC=C2C(\\C=C\\C3=CC=CC=N3)=N[NH]C2=C1",
            "category": "potent",
            "notes": "FDA-approved selective VEGFR inhibitor, IC50 ~0.2 nM",
        },
        {
            "name": "Lenvatinib",
            "smiles": "COC1=C2C=CC(OC3=CC=C(NC(=O)NC4CC4)C(F)=C3)=NC2=CC=C1C(N)=O",
            "category": "potent",
            "notes": "FDA-approved VEGFR inhibitor, IC50 ~4 nM",
        },
        {
            "name": "Pazopanib",
            "smiles": "CC1=C(C=C(C=C1)NC1=NC(=CC=N1)N(C)C1=CC2=C(C=C1)C(=N[NH]2)C)S(=O)(=O)N",
            "category": "potent",
            "notes": "FDA-approved VEGFR inhibitor, IC50 ~30 nM",
        },
        # ── Weak inhibitors ──
        {
            "name": "Vandetanib",
            "smiles": "COC1=CC2=C(C=C1OCC1CCN(C)CC1)N=CN=C2NC1=CC(=C(F)C=C1)Br",
            "category": "weak",
            "notes": "Multi-kinase inhibitor, IC50 ~40 nM for VEGFR2 but less potent than leaders",
        },
        {
            "name": "Semaxanib (SU5416)",
            "smiles": "CC1=CC(/C=C\\2/C(=O)NC3=CC=CC=C23)=C[NH]1",
            "category": "weak",
            "notes": "First-gen VEGFR2 inhibitor, IC50 ~1.2 µM",
        },
    ] + UNRELATED_COMPOUNDS,
}

BRAF_CONFIG = {
    "chembl_id": "CHEMBL5145",
    "name": "braf",
    "display_name": "BRAF",
    "activity_types": ["IC50", "Ki"],
    "activity_threshold_nM": 1_000,
    "description": "BRAF Kinase (V600E mutant) — melanoma and colorectal cancer target",
    "benchmark_compounds": [
        # ── Potent inhibitors ──
        {
            "name": "Vemurafenib",
            "smiles": "CCCS(=O)(=O)NC1=CC=C(F)C(C(=O)C2=CNC3=NC=C(C=C23)C2=CC=C(Cl)C=C2)=C1",
            "category": "potent",
            "notes": "FDA-approved BRAF V600E inhibitor, IC50 ~31 nM",
        },
        {
            "name": "Dabrafenib",
            "smiles": "CC(C)(C)C1=NC(=C(S1)C1=CC(=C(F)C=C1)NS(=O)(=O)C1=C(F)C=CC=C1F)C1=CC=NC(N)=N1",
            "category": "potent",
            "notes": "FDA-approved BRAF inhibitor, IC50 ~0.6 nM for V600E",
        },
        {
            "name": "Encorafenib",
            "smiles": "COC(=O)NC1=CC=C(C=C1F)N1C(=O)C(NC2=CC(=CC=C2)C2=C(Cl)C(NC(=O)C3CC3)=NN2C)=CC1=O",
            "category": "potent",
            "notes": "FDA-approved BRAF inhibitor, IC50 ~0.35 nM",
        },
        {
            "name": "PLX-4720",
            "smiles": "CCCS(=O)(=O)NC1=CC=C(F)C(C(=O)C2=CNC3=NC=C(C=C23)C2=CC=C(Cl)C=C2F)=C1",
            "category": "potent",
            "notes": "Preclinical BRAF V600E inhibitor, IC50 ~13 nM",
        },
        {
            "name": "GDC-0879",
            "smiles": "OC(C1=CC=C(C=C1)C1=CC=NC2=NC(=CC=C12)N1CCCC1)C1=CC=CC=C1",
            "category": "potent",
            "notes": "Selective BRAF inhibitor, IC50 ~0.13 nM",
        },
        # ── Weak inhibitors ──
        {
            "name": "Sorafenib",
            "smiles": "CNC(=O)C1=CC(OC2=CC=C(NC(=O)NC3=CC(=C(Cl)C=C3)C(F)(F)F)C=C2)=CC=N1",
            "category": "weak",
            "notes": "Multi-kinase inhibitor, IC50 ~22 nM for wt-BRAF, weaker for V600E",
        },
        {
            "name": "Regorafenib",
            "smiles": "CNC(=O)C1=CC(OC2=CC(=C(NC(=O)NC3=CC(=C(Cl)C=C3)C(F)(F)F)C=C2)F)=CC=N1",
            "category": "weak",
            "notes": "Multi-kinase inhibitor with moderate BRAF activity, IC50 ~28 nM",
        },
    ] + UNRELATED_COMPOUNDS,
}

CDK2_CONFIG = {
    "chembl_id": "CHEMBL301",
    "name": "cdk2",
    "display_name": "CDK2",
    "activity_types": ["IC50", "Ki"],
    "activity_threshold_nM": 1_000,
    "description": "Cyclin-Dependent Kinase 2 — cell cycle regulation, multiple cancer types",
    "benchmark_compounds": [
        # ── Potent inhibitors (verified in ChEMBL CDK2 data) ──
        {
            "name": "CHEMBL476578 (aminopyrimidine)",
            "smiles": "CC(=O)N[C@@H]1CCN(c2ccc(Nc3ncc(F)c(-c4cnc(C)n4C(C)C)n3)cc2)C1",
            "category": "potent",
            "notes": "ChEMBL verified CDK2 inhibitor, IC50 0.3 nM",
        },
        {
            "name": "CHEMBL526110 (pyrimidine)",
            "smiles": "O=[N+]([O-])c1cccc(Nc2nccc(-c3cnn4ncccc34)n2)c1",
            "category": "potent",
            "notes": "ChEMBL verified CDK2 inhibitor, IC50 0.3 nM",
        },
        {
            "name": "CHEMBL317703 (pyrazole-sulfonamide)",
            "smiles": "NS(=O)(=O)c1ccc(Nc2cc(-c3ccc([N+](=O)[O-])cc3)[nH]n2)cc1",
            "category": "potent",
            "notes": "ChEMBL verified CDK2 inhibitor, IC50 0.33 nM",
        },
        {
            "name": "Dinaciclib (SCH-727965)",
            "smiles": "CC(O)C1=CN=C(NCC2=C[NH]C3=CC=CC=C23)N=C1NC1CCC(=O)N(C1)C1CCCC1",
            "category": "potent",
            "notes": "Potent CDK1/2/5/9 inhibitor, IC50 ~1 nM for CDK2",
        },
        {
            "name": "SNS-032 (BMS-387032)",
            "smiles": "CC(=O)C1=CC=C(NC2=NC=C(S2)C(=O)NCCN2CCOCC2)C=C1",
            "category": "potent",
            "notes": "CDK2/7/9 inhibitor, IC50 ~48 nM for CDK2",
        },
        # ── Weak inhibitors (confirmed high IC50 in ChEMBL) ──
        {
            "name": "Olomoucine",
            "smiles": "CC(O)CNC1=NC(NC2=CC=CC=C2)=C2N=CN(C2=N1)C",
            "category": "weak",
            "notes": "First-gen CDK inhibitor, IC50 ~7 µM for CDK2",
        },
        {
            "name": "CHEMBL457047 (naphthol-sulfonamide)",
            "smiles": "NS(=O)(=O)c1cccc2c(O)cccc12",
            "category": "weak",
            "notes": "ChEMBL confirmed very weak, IC50 >120 µM",
        },
    ] + UNRELATED_COMPOUNDS,
}


# ══════════════════════════════════════════════════════════════════════════════════
# Registry of all target configs
# ══════════════════════════════════════════════════════════════════════════════════

ALL_TARGETS = {
    # Phase 1 — Diabetes
    "amylase": AMYLASE_CONFIG,
    "glucosidase": GLUCOSIDASE_CONFIG,
    "lipase": LIPASE_CONFIG,
    # Phase 2 — Hypertension
    "ace": ACE_CONFIG,
    # Phase 3 — Cancer
    "egfr": EGFR_CONFIG,
    "her2": HER2_CONFIG,
    "vegfr2": VEGFR2_CONFIG,
    "braf": BRAF_CONFIG,
    "cdk2": CDK2_CONFIG,
}


def get_config(target_name: str) -> dict:
    """Get configuration for a target by name.

    Args:
        target_name: Short target name (e.g., 'amylase', 'ace', 'egfr').

    Returns:
        Configuration dictionary.

    Raises:
        ValueError: If target name is not recognized.
    """
    if target_name not in ALL_TARGETS:
        valid = ", ".join(sorted(ALL_TARGETS.keys()))
        raise ValueError(f"Unknown target '{target_name}'. Valid targets: {valid}")
    return ALL_TARGETS[target_name]
