# prompt.py
GRAPH_FIELD_SEP = "<SEP>"

PROMPTS = {}

PROMPTS["DEFAULT_LANGUAGE"] = "English"
PROMPTS["DEFAULT_TUPLE_DELIMITER"] = "<|>"
PROMPTS["DEFAULT_RECORD_DELIMITER"] = "##"
PROMPTS["DEFAULT_COMPLETION_DELIMITER"] = "<|COMPLETE|>"
PROMPTS["process_tickers"] = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]

PROMPTS["DEFAULT_ENTITY_TYPES"] = ["Diseases", "Anatomy", "Interventions","Biomarkers", "Attributes"]

PROMPTS["entity_extraction"] = """-Goal-
You are an expert Clinical Oncologist and Researcher specializing in Endometrial Cancer (EC).
Your task is to extract structured knowledge from medical guidelines (e.g., NCCN, FIGO) and high-impact journal papers to build a **Clinical Decision Support Knowledge Graph**.
**Optimization Goal**: Capture actionable clinical rules, treatment outcomes, and molecular associations while rigorously filtering out academic noise.
Use {language} as output language.

-Schema Definition (Strict Constraints)-
* **Entity Types**:
  - **Diseases**: Specific cancer subtypes (e.g., "Serous EC", "Recurrent EC"), comorbidities, or patient cohorts.
  - **Anatomy**: Organs, metastasis sites, or specific tissue levels (e.g., "Para-aortic nodes", "Myometrium").
  - **Interventions**: Drugs, Regimens, Surgeries, Radiotherapy (e.g., "VBT", "EBRT").
  - **Biomarkers**: Genes, proteins, or molecular statuses (e.g., "POLE", "p53", "HER2").
  - **Attributes**: Quantitative (HR, PFS) or Qualitative (Grade 3, High-risk).

* **Semantic Roles (CRITICAL)**:
  - **CONDITION**: Input triggers, prerequisites, or patient characteristics (e.g., "Stage IB", "LVSI+", "Deep invasion").
  - **RECOMMENDATION**: Proposed actions or positive advice (e.g., "VBT", "Systemic therapy").
  - **CONTRAINDICATION**: Factors that exclude a treatment or explicit negative advice (e.g., "Fertility desire", "NOT recommended").
  - **CONTEXT**: Target population context or phase (e.g., "Post-menopausal", "Screening phase").
  - **EVIDENCE**: Clinical trials, statistical metrics, or source references (e.g., "PORTEC-3", "HR=0.6").

-Steps-
1. **Segmentation & Scoring**:
   - Divide text into **atomic knowledge segments**.
   - **Calculate Composite Score (0-10)** based on Clinical Utility and Semantic Completeness.
   - **FILTERING RULE**: If **Total Score < 7**, **DISCARD** this segment entirely.
   - Format: ("hyper-relation"{tuple_delimiter}<knowledge_segment>{tuple_delimiter}<Total_Score>)

2. **Entity Extraction**:
   - **Only for segments with Total Score >= 7**:
   - Extract entities fitting the Schema.
   - **Assign a Semantic ROLE** to each entity based on its function in the segment.
   - **key_score**: 100 (Core), 80 (Context), <50 (Ignore).
   - Format: ("entity"{tuple_delimiter}<entity_name>{tuple_delimiter}<entity_type>{tuple_delimiter}<entity_description>{tuple_delimiter}<ENTITY_ROLE>{tuple_delimiter}<key_score>)

3. Return output as a single list using **{record_delimiter}** as the delimiter.
4. When finished, output {completion_delimiter}
######################
-Examples-
######################
{examples}
#############################
-Real Data-
######################
Text: {input_text}
######################
Output:
"""

PROMPTS["entity_extraction_examples"] = [
    """Example 1: (Clinical Trial Result - Journal Paper style)

Text:
In the KEYNOTE-775 trial, the combination of Lenvatinib and Pembrolizumab significantly improved Progression-Free Survival (PFS) compared to chemotherapy (HR 0.56; 95% CI, 0.43 to 0.73) in patients with advanced endometrial cancer.
################
Output:
("hyper-relation"{tuple_delimiter}"In the KEYNOTE-775 trial, the combination of Lenvatinib and Pembrolizumab significantly improved Progression-Free Survival (PFS) compared to chemotherapy (HR 0.56; 95% CI, 0.43 to 0.73) in patients with advanced endometrial cancer."{tuple_delimiter}10){record_delimiter}
("entity"{tuple_delimiter}"KEYNOTE-775"{tuple_delimiter}"Attributes"{tuple_delimiter}"Clinical trial identifier."{tuple_delimiter}"EVIDENCE"{tuple_delimiter}90){record_delimiter}
("entity"{tuple_delimiter}"Lenvatinib"{tuple_delimiter}"Interventions"{tuple_delimiter}"Tyrosine kinase inhibitor drug."{tuple_delimiter}"RECOMMENDATION"{tuple_delimiter}100){record_delimiter}
("entity"{tuple_delimiter}"Pembrolizumab"{tuple_delimiter}"Interventions"{tuple_delimiter}"PD-1 inhibitor immunotherapy."{tuple_delimiter}"RECOMMENDATION"{tuple_delimiter}100){record_delimiter}
("entity"{tuple_delimiter}"PFS"{tuple_delimiter}"Attributes"{tuple_delimiter}"Progression-Free Survival outcome."{tuple_delimiter}"EVIDENCE"{tuple_delimiter}100){record_delimiter}
("entity"{tuple_delimiter}"HR 0.56"{tuple_delimiter}"Attributes"{tuple_delimiter}"Hazard Ratio indicating benefit."{tuple_delimiter}"EVIDENCE"{tuple_delimiter}100){record_delimiter}
("entity"{tuple_delimiter}"advanced endometrial cancer"{tuple_delimiter}"Diseases"{tuple_delimiter}"Target disease population."{tuple_delimiter}"CONTEXT"{tuple_delimiter}100){record_delimiter}
#############################""",
    """Example 2: (Molecular-Driven Treatment - Guideline style with Contraindication)

Text:
For Stage IA patients, observation is preferred. However, if patients have p53 mutation (high-risk), observation is NOT recommended.
################
Output:
("hyper-relation"{tuple_delimiter}"For Stage IA patients, observation is preferred. However, if patients have p53 mutation (high-risk), observation is NOT recommended."{tuple_delimiter}10){record_delimiter}
("entity"{tuple_delimiter}"Stage IA"{tuple_delimiter}"Attributes"{tuple_delimiter}"Early stage disease."{tuple_delimiter}"CONDITION"{tuple_delimiter}100){record_delimiter}
("entity"{tuple_delimiter}"Observation"{tuple_delimiter}"Interventions"{tuple_delimiter}"No active treatment, surveillance only."{tuple_delimiter}"RECOMMENDATION"{tuple_delimiter}100){record_delimiter}
("entity"{tuple_delimiter}"p53 mutation"{tuple_delimiter}"Biomarkers"{tuple_delimiter}"High-risk molecular feature."{tuple_delimiter}"CONTRAINDICATION"{tuple_delimiter}100){record_delimiter}
("entity"{tuple_delimiter}"Observation"{tuple_delimiter}"Interventions"{tuple_delimiter}"Excluded due to high risk."{tuple_delimiter}"CONTRAINDICATION"{tuple_delimiter}100){record_delimiter}
#############################""",
    """Example 3: (Standard Guideline Rule)

Text:
Vaginal Brachytherapy (VBT) is recommended for patients with deep myometrial invasion and Grade 2 tumors.
################
Output:
("hyper-relation"{tuple_delimiter}"Vaginal Brachytherapy (VBT) is recommended for patients with deep myometrial invasion and Grade 2 tumors."{tuple_delimiter}10){record_delimiter}
("entity"{tuple_delimiter}"VBT"{tuple_delimiter}"Interventions"{tuple_delimiter}"Vaginal Brachytherapy."{tuple_delimiter}"RECOMMENDATION"{tuple_delimiter}100){record_delimiter}
("entity"{tuple_delimiter}"deep myometrial invasion"{tuple_delimiter}"Anatomy"{tuple_delimiter}"Invasion > 50%."{tuple_delimiter}"CONDITION"{tuple_delimiter}100){record_delimiter}
("entity"{tuple_delimiter}"Grade 2"{tuple_delimiter}"Attributes"{tuple_delimiter}"Moderate differentiation."{tuple_delimiter}"CONDITION"{tuple_delimiter}100){record_delimiter}
#############################"""
]

PROMPTS[
    "summarize_entity_descriptions"
] = """You are a helpful assistant responsible for generating a comprehensive summary of the data provided below.
Given one or two entities, and a list of descriptions, all related to the same entity or group of entities.
Please concatenate all of these into a single, comprehensive description. Make sure to include information collected from all the descriptions.
If the provided descriptions are contradictory, please resolve the contradictions and provide a single, coherent summary.
Make sure it is written in third person, and include the entity names so we the have full context.
Use {language} as output language.

#######
-Data-
Entities: {entity_name}
Description List: {description_list}
#######
Output:
"""

PROMPTS[
    "entiti_continue_extraction"
] = """MANY knowdge fragements with entities were missed in the last extraction.  Add them below using the same format:
"""

PROMPTS[
    "entiti_if_loop_extraction"
] = """Please check whether knowdge fragements cover all the given text.  Answer YES | NO if there are knowdge fragements that need to be added.
"""

PROMPTS["fail_response"] = "Sorry, I'm not able to provide an answer to that question."

PROMPTS["rag_response"] = """---Role---

You are a helpful assistant responding to questions about data in the tables provided.

---Goal---

Generate a response of the target length and format that responds to the user's question, summarizing all information in the input data tables appropriate for the response length and format, and incorporating any relevant general knowledge.
If you don't know the answer, just say so. Do not make anything up.
Do not include information where the supporting evidence for it is not provided.

---Target response length and format---

{response_type}

---Data tables---

{context_data}

Add sections and commentary to the response as appropriate for the length and format. Style the response in markdown.
"""

PROMPTS["keywords_extraction"] = """---Role---

You are a helpful assistant tasked with identifying both high-level and low-level keywords in the user's query.

---Goal---

Given the query, list both high-level and low-level keywords. High-level keywords focus on overarching concepts or themes, while low-level keywords focus on specific entities, details, or concrete terms.

---Instructions---

- Output the keywords in JSON format.
- The JSON should have two keys:
  - "high_level_keywords" for overarching concepts or themes.
  - "low_level_keywords" for specific entities or details.

######################
-Examples-
######################
{examples}

#############################
-Real Data-
######################
Query: {query}
######################
The `Output` should be human text, not unicode characters. Keep the same language as `Query`.
Output:

"""

PROMPTS["keywords_extraction_examples"] = [
    """Example 1:

Query: "How does international trade influence global economic stability?"
################
Output:
{{
  "high_level_keywords": ["International trade", "Global economic stability", "Economic impact"],
  "low_level_keywords": ["Trade agreements", "Tariffs", "Currency exchange", "Imports", "Exports"]
}}
#############################""",
    """Example 2:

Query: "What are the environmental consequences of deforestation on biodiversity?"
################
Output:
{{
  "high_level_keywords": ["Environmental consequences", "Deforestation", "Biodiversity loss"],
  "low_level_keywords": ["Species extinction", "Habitat destruction", "Carbon emissions", "Rainforest", "Ecosystem"]
}}
#############################""",
    """Example 3:

Query: "What is the role of education in reducing poverty?"
################
Output:
{{
  "high_level_keywords": ["Education", "Poverty reduction", "Socioeconomic development"],
  "low_level_keywords": ["School access", "Literacy rates", "Job training", "Income inequality"]
}}
#############################""",
]


PROMPTS["naive_rag_response"] = """---Role---

You are a helpful assistant responding to questions about documents provided.

---Goal---

Generate a response of the target length and format that responds to the user's question, summarizing all information in the input data tables appropriate for the response length and format, and incorporating any relevant general knowledge.
If you don't know the answer, just say so. Do not make anything up.
Do not include information where the supporting evidence for it is not provided.

---Target response length and format---

{response_type}

---Documents---

{content_data}

Add sections and commentary to the response as appropriate for the length and format. Style the response in markdown.
"""

PROMPTS[
    "similarity_check"
] = """Please analyze the similarity between these two questions:

Question 1: {original_prompt}
Question 2: {cached_prompt}

Please evaluate the following two points and provide a similarity score between 0 and 1 directly:
1. Whether these two questions are semantically similar
2. Whether the answer to Question 2 can be used to answer Question 1
Similarity score criteria:
0: Completely unrelated or answer cannot be reused, including but not limited to:
   - The questions have different topics
   - The locations mentioned in the questions are different
   - The times mentioned in the questions are different
   - The specific individuals mentioned in the questions are different
   - The specific events mentioned in the questions are different
   - The background information in the questions is different
   - The key conditions in the questions are different
1: Identical and answer can be directly reused
0.5: Partially related and answer needs modification to be used
Return only a number between 0-1, without any additional content.
"""