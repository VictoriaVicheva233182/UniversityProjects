<h1 align="center">Victoria Vicheva</h1>

<p align="center">
  <img src="https://img.shields.io/badge/AI%20assurance%2C%20machine%20learning%20and%20data%20science-0A0A0A?style=for-the-badge" alt="AI assurance, machine learning and data science">
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-0A0A0A?style=flat-square&logo=python&logoColor=E5EE4E" alt="Python">
  <img src="https://img.shields.io/badge/SQL-0A0A0A?style=flat-square" alt="SQL">
  <img src="https://img.shields.io/badge/TensorFlow-0A0A0A?style=flat-square&logo=tensorflow&logoColor=E5EE4E" alt="TensorFlow">
  <img src="https://img.shields.io/badge/scikit--learn-0A0A0A?style=flat-square&logo=scikitlearn&logoColor=E5EE4E" alt="scikit-learn">
  <img src="https://img.shields.io/badge/LLMs%20and%20RAG-0A0A0A?style=flat-square&logo=langchain&logoColor=E5EE4E" alt="LLMs and RAG">
  <img src="https://img.shields.io/badge/FastAPI-0A0A0A?style=flat-square&logo=fastapi&logoColor=E5EE4E" alt="FastAPI">
  <img src="https://img.shields.io/badge/Docker-0A0A0A?style=flat-square&logo=docker&logoColor=E5EE4E" alt="Docker">
  <img src="https://img.shields.io/badge/Power%20BI-0A0A0A?style=flat-square" alt="Power BI">
  <img src="https://img.shields.io/badge/BigQuery-0A0A0A?style=flat-square&logo=googlebigquery&logoColor=E5EE4E" alt="BigQuery">
  <img src="https://img.shields.io/badge/Azure%20and%20GCP-0A0A0A?style=flat-square" alt="Azure and GCP">
</p>

I build machine learning systems and test whether AI can be trusted. This repository collects my university work in computer vision, predictive modelling, analytics and research, plus two recent projects on AI in assurance: one makes an AI system safe, the other uses AI to make an audit better.

<br>

## Featured projects

### LLM Assurance Lab
**Red teaming and hardening an AI assistant for a bank**

<p>
  <img src="https://img.shields.io/badge/attack%20success-32%25%20to%200%25-E5EE4E?style=for-the-badge&labelColor=0A0A0A" alt="Attack success: 32% to 0%">
  <img src="https://img.shields.io/badge/normal%20questions%20blocked-1%20of%2025-ECBDD8?style=for-the-badge&labelColor=0A0A0A" alt="Normal questions blocked: 1 of 25">
  <img src="https://img.shields.io/badge/risks%20tested-7%20categories-ECBDD8?style=for-the-badge&labelColor=0A0A0A" alt="Risks tested: 7 categories">
</p>

I built an AI customer assistant for a fictional bank, attacked it in 7 ways (prompt injection, data leaks, a poisoned document, jailbreaks), then fixed what broke. A machine learning guardrail I trained, access control in code and an output filter took attack success from 32% to 0% against Llama 3.1 8B. The project ends with a client-style assurance report mapped to OWASP, MITRE ATLAS, the EU AI Act, GDPR and DORA.

**Key finding:** the model was not the weak point, the system design was. Llama refused every jailbreak by itself, but still leaked another customer's data in 3 of 4 attempts. Only controls in code fixed that.

`Red teaming` `ML guardrail` `RAG` `Risk rating` `FastAPI` `Ollama`

**[Read the project](./llm-assurance-lab)**

<br>

### LedgerLens
**AI-assisted journal entry testing for auditors**

<p>
  <img src="https://img.shields.io/badge/all%208%20fraud%20schemes-in%20the%20first%20160%20entries-E5EE4E?style=for-the-badge&labelColor=0A0A0A" alt="All 8 fraud schemes in the first 160 entries">
  <img src="https://img.shields.io/badge/classic%20rules%20flagged-2%2C368%20entries-ECBDD8?style=for-the-badge&labelColor=0A0A0A" alt="Classic rules flagged 2,368 entries">
  <img src="https://img.shields.io/badge/copilot%20drafts%20checked-87%25%20grounded-ECBDD8?style=for-the-badge&labelColor=0A0A0A" alt="Copilot drafts: 87% grounded">
</p>

Auditors must test journal entries for fraud (ISA 240), but classic rules flag thousands of normal entries. LedgerLens learns what is normal for each person and account, ranks a ledger of 198,053 entries, and found all 8 hidden fraud schemes in the first 160. An AI copilot then investigates the top entries with read-only ledger tools and drafts a finding. Every number in a draft is checked against the ledger, and the auditor makes the final call in a workbench.

**Key finding:** the copilot put a wrong number in 2 of 15 drafts, and the automatic check caught both before an auditor saw them.

`Anomaly detection` `Autoencoder` `LLM agent with tools` `Audit analytics` `Evaluation`

**[Read the project](./ledgerlens)**

<br>

## University projects

| Project                                          | What I did                                                                                                                                        | Skills                                                            |
| ------------------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------- |
| **Root length prediction** (computer vision)     | A deep learning pipeline that predicts plant root length from images: preprocessing, feature extraction and regression with TensorFlow and Keras. | CNNs, image processing, regression, data augmentation             |
| **Skin cancer detection** (deep learning)        | A CNN that classifies skin lesions as benign or malignant, using transfer learning and careful model evaluation.                                  | TensorFlow, transfer learning, image classification, model tuning |
| **Incident severity forecasting** (client: ANWB) | A predictive model for road incident severity in Breda, built with ANWB. Only the final presentation is shared because of an NDA.                 | Predictive analytics, client work, data modelling                 |
| **Power BI dashboard** (data science)            | An interactive dashboard that explores relationships between variables and presents the insights for business users.                              | Power BI, KPI tracking, data storytelling                         |
| **Data analysis**                                | Exploratory and statistical analysis with Pandas, Matplotlib and Seaborn: trends, distributions and hypothesis tests.                             | EDA, statistics, visualisation                                    |

## Research

- **Individual research paper:** how employees in small and medium-sized companies see generative AI, as a chance to grow their career or as a threat to their job.
- **Group policy paper:** a strategic policy for using AI ethically in organisations.

`Academic writing` `Qualitative and quantitative research` `Policy analysis`

<br>

## About me

I graduated cum laude in Applied Data Science and AI at Breda University of Applied Sciences and I'm now doing a Master's in Data-Driven Business at The Hague University of Applied Sciences. I have AI and data consulting experience at Deloitte.

I'm most interested in AI assurance: proving, with evidence, that an AI system is safe enough to use, and using AI to make assurance work itself better.

## Contact

Open to conversations about roles in AI assurance, AI and data consulting, and machine learning.

[LinkedIn](www.linkedin.com/in/victoria-vicheva-3817b6263) | [Email](mailto:victoria.v.vicheva@gmail.com)