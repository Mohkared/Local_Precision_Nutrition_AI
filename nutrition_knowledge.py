"""
nutrition_knowledge.py
────────────────────────────────────────────────────────────────────────────
Curated, evidence-based nutrition knowledge base.
17 documents spanning USDA, ADA, AHA, NIH, ISSN, WHO, AAP, NKF guidelines.
Each document is chunked into the RAG engine on first initialisation.
────────────────────────────────────────────────────────────────────────────
"""
NUTRITION_FILES = [
    {
        "directory": r"Sources\Dietary_Guidelines_for_Americans_2020-2025.md",
        "source": "U.S. Department of Agriculture & U.S. Department of Health and Human Services. (2020). Dietary Guidelines for Americans, 2020-2025 (9th ed.). https://www.dietaryguidelines.gov/sites/default/files/2020-12/Dietary_Guidelines_for_Americans_2020-2025.pdf",
    },
    {
        "directory": r"Sources\healthy-diet-fact-sheet-394.md",
        "source": "World Health Organization. (2020, April 29). Healthy diet. https://cdn.who.int/media/docs/default-source/healthy-diet/healthy-diet-fact-sheet-394.pdf",
    },
    {
        "directory": r"Sources\MOPH_DIETARY_BOOKLET_ENG.md",
        "source": "Health Promotion and Non-communicable Diseases Section. Public Health Department. The Ministry of Public Health. Qatar, Doha. 2015 https://hiap.moph.gov.qa/EN/Documents/Library/MOPH_DIETARY_BOOKLET_ENG.pdf"
    },
    {
        "directory": r"Sources\nutrients-15-04314.md",
        "source": "Iizuka, K., & Yabe, D. (2023). Dietary and Nutritional Guidelines for People with Diabetes. Nutrients, 15(20), 4314. https://doi.org/10.3390/nu15204314"
    },
    {
        "directory": r"Sources\PIIS0272638620307265.md",
        "source": "Ikizler, T. A., Burrowes, J. D., Byham-Gray, L. D., Campbell, K. L., Carrero, J., Chan, W., . . . Cuppari, L. (2020c). KDOQI Clinical Practice Guideline for Nutrition in CKD: 2020 Update. American Journal of Kidney Diseases, 76(3), S1–S107. https://doi.org/10.1053/j.ajkd.2020.05.006"
    },
    {
        "directory": r"Sources\nihms-816355.md",
        "source": "Kominiarek, M. A., & Rajan, P. (2016). Nutrition Recommendations in Pregnancy and Lactation. The Medical clinics of North America, 100(6), 1199–1215. https://doi.org/10.1016/j.mcna.2016.06.004"
    },
    {
        "directory": r"Sources\fnut-10-1331854.md",
        "source": "Amawi, A., AlKasasbeh, W., Jaradat, M., Almasri, A., Alobaidi, S., Hammad, A. A., Bishtawi, T., Fataftah, B., Turk, N., Saoud, H. A., Jarrar, A., & Ghazzawi, H. (2024). Athletes' nutritional demands: a narrative review of nutritional requirements. Frontiers in nutrition, 10, 1331854. https://doi.org/10.3389/fnut.2023.1331854"
    },
    {
        "directory": r"Sources\infant-feeding-guide.md",
        "source": "Infant Nutrition and Feeding: a Guide for Use in the Special Supplemental Nutrition Program for Women, Infants, and Children (WIC). (2019). Retrieved from https://wicworks.fns.usda.gov/resources/infant-nutrition-and-feeding-guide"
    },
    {
        "directory": r"Sources\Healthy_Diet_WHO.md",
        "source": "World Health Organization: WHO. (2026, January 26). Healthy diet. Retrieved from https://www.who.int/news-room/fact-sheets/detail/healthy-diet"
    },
    {
        "directory": r"Sources\miller-et-al-2026-a-clinician-s-guide-for-trending-cardiovascular-nutritional-controversies-in-2026.md",
        "source": "Miller, M, Aggarwal, M, Allen, K. et al. A Clinician’s Guide for Trending Cardiovascular Nutritional Controversies in 2026. JACC Adv. 2026 Mar, 5 (3) . https://doi.org/10.1016/j.jacadv.2026.102591"
    },
    
]