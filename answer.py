import re
import os
import json
import pandas as pd
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate

# ==========================================
# 🛑 PASTE YOUR GROQ API KEY HERE 🛑
# Get a free key at: console.groq.com
# ==========================================
os.environ["GROQ_API_KEY"] = "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"

def generate_submission_optimized(kg_file_path, questions_file_path, output_jsonl_path):
    # 1. Load your DataFrames
    print(f"Loading knowledge graph from {kg_file_path}...")
    with open(kg_file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        
    df_projects = pd.DataFrame(data.get('projects', []))
    df_credentials = pd.DataFrame(data.get('credentials', []))
    
    # 2. Load Questions (with the metadata wrapper fix)
    print(f"Loading questions from {questions_file_path}...")
    with open(questions_file_path, 'r', encoding='utf-8') as f:
        raw_data = json.load(f)
        
    # Check if questions are wrapped inside a "questions" metadata block
    if isinstance(raw_data, dict) and "questions" in raw_data:
        questions_data = raw_data["questions"]
    else:
        questions_data = raw_data

    # Safely extract qid and question text
    questions_list = []
    if isinstance(questions_data, list):
        for item in questions_data:
             questions_list.append((item.get('qid'), item.get('question')))
    elif isinstance(questions_data, dict):
        for qid, val in questions_data.items():
             q_text = val.get('question') if isinstance(val, dict) else val
             questions_list.append((qid, q_text))

    # 3. Initialize the highly optimized Groq LLM 
    # llama-3.3-70b-versatile is incredibly fast and operates well within the free tier limits
    llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)

    # 4. The Single-Shot Prompt (Minimizes token usage)
    # 4. The Multi-Line Execution Prompt
    # 4. The Multi-Line Execution Prompt with Strict Guardrails
    prompt = PromptTemplate.from_template("""
    You are an expert Python data analyst. I have two pandas DataFrames:
    1. `df_projects` with columns: {proj_cols}
    2. `df_credentials` with columns: {cred_cols}

    Question: {question}

    Write a multi-line Python script to calculate the exact numerical answer.
    
    CRITICAL HACKATHON RULES TO MAXIMIZE ACCURACY:
    1. THE LINGUISTIC TRAP: If a question asks for the total value of assignments a specific person "has delivered for" a client, DO NOT filter the final aggregation by that person. You must aggregate ALL projects for that client across the entire company.
    2. REFERENCE POINTS: If asked to use a project or date as a "reference point" to find a client, find the client first, and THEN apply any date filters (like "after PMP issuance") to that client's entire portfolio.
    3. SAFE INDEXING: DO NOT use `.iloc[0]` without a fallback. (e.g., `val = df['col'].iloc[0] if not df.empty else None`).
    4. MISSING DATA: The `role` column in df_projects is entirely null. DO NOT filter on it.
    5. Always convert date columns using `pd.to_datetime()` before comparing dates.
    6. You MUST assign your final numerical result to a variable named `final_answer`.

    Output your response EXACTLY in this format:
    
    ### REASONING
    (Write your step-by-step logic here to plan the code)
    
    ### PYTHON CODE
    ```python
    # your code here
    ```
    """)

    proj_cols = list(df_projects.columns)
    cred_cols = list(df_credentials.columns)

    print(f"Processing {len(questions_list)} questions via Groq LPU (Chain-of-Thought mode)...\n")

    # 5. Execute and stream results
    with open(output_jsonl_path, 'w', encoding='utf-8') as out_file:
        for qid, question in questions_list:
            print(f"Processing QID: {qid}...")
            q_prompt = prompt.format(proj_cols=proj_cols, cred_cols=cred_cols, question=question)
            
            raw_code = "No code generated (API Error)"
            
            try:
                # Ask Groq to generate the reasoning and the script
                raw_output = llm.invoke(q_prompt).content
                
                # Handle cases where LangChain returns a list of blocks
                if isinstance(raw_output, list):
                    raw_output = "".join(block.get("text", "") for block in raw_output if isinstance(block, dict))
                
                # Use Regex to cleanly extract only the python code block
                match = re.search(r'```(?:python)?(.*?)```', raw_output, re.DOTALL | re.IGNORECASE)
                if match:
                    raw_code = match.group(1).strip()
                else:
                    # Fallback if it ignored the markdown formatting
                    raw_code = raw_output.replace("### PYTHON CODE", "").replace("### REASONING", "").strip()
                
                # Execute the multi-line code block in a controlled local scope
                local_scope = {"df_projects": df_projects, "df_credentials": df_credentials, "pd": pd}
                exec(raw_code, globals(), local_scope)
                
                # Retrieve the variable assigned by the LLM
                final_answer = local_scope.get("final_answer", 0)
                
                # Convert NumPy / Pandas scalars to native Python types
                if hasattr(final_answer, 'item'):
                    final_answer = final_answer.item()
                if isinstance(final_answer, float) and final_answer.is_integer():
                    final_answer = int(final_answer)
                
            except Exception as e:
                print(f"   [!] Failed to execute code:\n{raw_code}")
                print(f"   [!] Error details: {e}")
                final_answer = 0

            # Wrap in the required JSON format and write to the line
            result_obj = {"qid": qid, "answer": final_answer}
            out_file.write(json.dumps(result_obj) + '\n')
            out_file.flush()
            
    print(f"\n✅ Done! File saved to: {output_jsonl_path}")

if __name__ == "__main__":
    KNOWLEDGE_GRAPH_FILE = "knowledge_graph.json"
    QUESTIONS_FILE = "sample_questions.json"
    SUBMISSION_FILE = "my_answers_groq.jsonl"
    
    generate_submission_optimized(KNOWLEDGE_GRAPH_FILE, QUESTIONS_FILE, SUBMISSION_FILE)