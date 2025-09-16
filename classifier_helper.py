import re
import joblib
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
import os

from langchain.prompts import FewShotPromptTemplate, PromptTemplate
from langchain_openai import ChatOpenAI

load_dotenv()

# ---------------------Regex Classification ---------------------

def classify_with_regex(log_message):
    regex_patterns = {
        r"User User\d+ logged (in|out).": "User Action",
        r"Backup (started|ended) at .*": "System Notification",
        r"Backup completed successfully.": "System Notification",
        r"System updated to version .*": "System Notification",
        r"File .* uploaded successfully by user .*": "System Notification",
        r"Disk cleanup completed successfully.": "System Notification",
        r"System reboot initiated by user .*": "System Notification",
        r"Account with ID .* created by .*": "User Action"
    }
    for pattern, label in regex_patterns.items():
        if re.search(pattern, log_message):
            return label
    return None


# ----------------------ML Classification ----------------------

model_embedding = SentenceTransformer('all-MiniLM-L6-v2')  
model_classification = joblib.load("models/log_classifier.joblib")

def classify_with_ml(log_message):
    embeddings = model_embedding.encode([log_message])
    probabilities = model_classification.predict_proba(embeddings)[0]
    if max(probabilities) < 0.5:
        return "Unclassified"
    predicted_label = model_classification.predict(embeddings)[0]
    
    return predicted_label


# ----------------------LLM Classification ----------------------

def classify_with_llm(log_msg):
    few_shot_examples = [
        {"log": "Lead conversion failed for prospect ID 7842 during workflow execution.", "category": "Workflow Error"},
        {"log": "Customer follow-up process for lead ID 5621 failed due to workflow misconfiguration.", "category": "Workflow Error"},
        {"log": "Escalation rule execution failed for ticket ID 3242.", "category": "Workflow Error"},
        {"log": "API endpoint 'getCustomerDetails' is deprecated. Please use 'fetchCustomerInfo' instead.", "category": "Deprecation Warning"},
        {"log": "The 'ExportToCSV' feature is outdated. Please use the new export tool.", "category": "Deprecation Warning"},
        {"log": "Support for legacy authentication methods will be removed in the next release.", "category": "Deprecation Warning"},
    ]

    example_prompt = PromptTemplate(
        input_variables=["log", "category"],
        template="Log: {log}\nCategory: {category}"
    )

    prompt = FewShotPromptTemplate(
        examples=few_shot_examples,
        example_prompt=example_prompt,
        suffix="""Classify the following LegacyCRM log message into one of these categories:
    (1) Workflow Error
    (2) Deprecation Warning
    If it does not fit either category, return 'Unclassified'.
    Return only the category name.

    Log: {log}
    Category:""",
        input_variables=["log"],
    )

    llm = ChatOpenAI(
        model="gpt-3.5-turbo",
        temperature=0,
        openai_api_key=os.getenv("OPENAI_API_KEY")
    )
    chain = prompt | llm

    result = chain.invoke({"log": log_msg})
    return result.content



if __name__ == "__main__":
    print("Classification with regex")
    print(classify_with_regex("Backup completed successfully."))
    print(classify_with_regex("Account with ID 1234 created by User1."))
    print(classify_with_regex("Hey Bro, chill ya!"))

    print("\nClassification with bert")
    print(classify_with_ml("alpha.osapi_compute.wsgi.server - 12.10.11.1 - API returned 404 not found error"))
    print(classify_with_ml("GET /v2/3454/servers/detail HTTP/1.1 RCODE   404 len: 1583 time: 0.1878400"))
    print(classify_with_ml("System crashed due to drivers errors when restarting the server"))
    print(classify_with_ml("Hey bro, chill ya!"))
    print(classify_with_ml("Multiple login failures occurred on user 6454 account"))
    print(classify_with_ml("Server A790 was restarted unexpectedly during the process of data transfer"))

    print("\nClassification with llm")

    print(classify_with_llm("Case escalation for ticket ID 7324 failed because the assigned support agent is no longer active."))
    print(classify_with_llm("The 'ReportGenerator' module will be retired in version 4.0. Please migrate to the 'AdvancedAnalyticsSuite' by Dec 2025"))
    print(classify_with_llm("System reboot initiated by user 12345."))