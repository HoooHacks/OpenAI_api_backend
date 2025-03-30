from flask import Flask, request, jsonify
import openai
from flask_cors import CORS
from dotenv import load_dotenv
import os
import json
import time
import re

# Load environment variables and set OpenAI API key
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")
CODE_CHAT_ASSISTANT_ID = os.getenv("CODE_CHAT_ASSISTANT_ID")
# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Allow frontend requests

# ------------------------
# Sonarqube related
# ------------------------

# upload file to openAI to use as background knowledge
def upload_file_to_openai(file_path):
    """Uploads a file to OpenAI and returns the file ID."""
    try:
        client = openai.OpenAI()
        with open(file_path, "rb") as file:
            response = client.files.create(
                file=file,
                purpose="assistants"
            )
        print(f"[INFO] File uploaded to OpenAI. File ID: {response.id}")
        return response.id
    except Exception as e:
        print(f"[ERROR] File Upload Failed: {e}")
        return None

# make file into vector store for openAI to better understand the SonarQube json output
def create_vector_store(file_id):
    """Creates a vector store and adds the file to it."""
    try:
        client = openai.OpenAI()
        vector_store = client.beta.vector_stores.create(name="SonarQube Vector Store")
        vector_store_id = vector_store.id
        print(f"[INFO] Vector Store Created: {vector_store_id}")

        # Add file to vector store
        client.beta.vector_stores.files.create(
            vector_store_id=vector_store_id,
            file_id=file_id
        )
        print(f"[INFO] File {file_id} added to vector store {vector_store_id}")

        return vector_store_id
    except Exception as e:
        print(f"[ERROR] Vector Store Creation Failed: {e}")
        return None

# creates openai assistant for sonarqube, utilizing vector store
def create_sonarqube_assistant(vector_store_id):
    """Creates an OpenAI Assistant with file search capabilities."""
    try:
        client = openai.OpenAI()
        assistant = client.beta.assistants.create(
            model="gpt-4o",
            name="SonarQube Analysis Assistant",
            instructions="You are an expert in analyzing SonarQube reports. Summarize the top issues from the uploaded report.",
            tools=[{"type": "file_search"}],
            tool_resources={"file_search": {"vector_store_ids": [vector_store_id]}}
        )
        print(f"[INFO] Assistant Created: {assistant.id}")
        return assistant.id
    except Exception as e:
        print(f"[ERROR] Assistant Creation Failed: {e}")
        return None

# summarizes sonarqube's json file to extract useful information only
def summarize_sonar_issues(file_path):
    """Extracts only 'type' and 'message' from the SonarQube JSON file before uploading to OpenAI."""
    try:
        with open(file_path, "r") as file:
            data = json.load(file)

        # Extract only the necessary fields
        extracted_issues = [{"type": issue.get("type"), "message": issue.get("message")} for issue in data.get("issues", [])]

        # Save summarized data
        summarized_file_path = file_path.replace(".json", "_summary.json")
        with open(summarized_file_path, "w") as summarized_file:
            json.dump({"issues": extracted_issues}, summarized_file, indent=4)

        print(f"[INFO] Summarized file created: {summarized_file_path}")
        return summarized_file_path
    except Exception as e:
        print(f"[ERROR] Failed to summarize SonarQube file: {e}")
        return None

def analyze_sonarqube_with_assistant(file_id):
    """Uses OpenAI Assistant to analyze the uploaded SonarQube report."""
    try:
        client = openai.OpenAI()

        # Step 1: Create Vector Store & Add File
        vector_store_id = create_vector_store(file_id)
        if not vector_store_id:
            return None

        # Step 2: Create an Assistant
        assistant_id = create_sonarqube_assistant(vector_store_id)
        if not assistant_id:
            return None

        # Step 3: Create a Thread
        thread = client.beta.threads.create()
        thread_id = thread.id
        print(f"[INFO] Thread Created: {thread_id}")

        # Step 4: Attach a message with the file reference
        client.beta.threads.messages.create(
            thread_id=thread_id,
            role="user",
            content="""
        You are an expert in software quality and static analysis.

        Analyze the SonarQube report attached to this thread and return the following:

        1. A summary of the **top 10 most critical and common issues** found.
        2. A medium length explanation for each issue — why it matters and what developers should learn.
        3. For **each issue**, provide a **direct YouTube video link** that best teaches how to fix or learn more about that type of issue. Use an actual public video link you've seen before or one that would help them most. Do not put explanation or title - straight link

        Format the output **EXACTLY** like this (Keep 'IssueNumber'):

        IssueNumber 1. <Issue Name>: <Short Explanation>
        - YouTube: <Direct YouTube Link>
        IssueNumber 2. ...
        IssueNumber 3. ...

        Make sure each YouTube link is a real, useful video and directly relevant to the issue.
        """
        )

        # Step 5: Run the Assistant
        run = client.beta.threads.runs.create(
            thread_id=thread_id,
            assistant_id=assistant_id
        )

        run_id = run.id
        print(f"[INFO] Run Started: {run_id}")

        # Step 6: Wait for Completion
        while True:
            run_status = client.beta.threads.runs.retrieve(
                thread_id=thread_id,
                run_id=run_id
            )
            if run_status.status == "completed":
                break
            print("[INFO] Waiting for completion...")
            time.sleep(2)  # Check every 2 seconds

        # Step 7: Retrieve the Response
        messages = client.beta.threads.messages.list(thread_id=thread_id)
        response_text = messages.data[0].content[0].text.value  # Get the latest response

        return response_text
    except Exception as e:
        print(f"[ERROR] OpenAI API Request Failed: {e}")
        return None

def parse_sonarqube_summary(response_text):
    """
    Parses structured GPT output in this format:

    Issue 1. <Issue Name>: <Explanation>
    - YouTube: <YouTube Link>
    """

    pattern = re.compile(
        r"IssueNumber \d+\.\s+(.+?):\s+(.*?)- YouTube:\s+(https?://[^\s]+)",
        re.DOTALL
    )

    matches = pattern.findall(response_text)
    issues = []

    for name, explanation, link in matches:
        issues.append({
            "name": name.strip(),
            "explanation": explanation.strip(),
            "link": link.strip()
        })

    return {"issues": issues}


@app.route('/analyze_sonarqube', methods=['POST'])
def analyze_sonarqube():
    try:
        # Ensure a file is uploaded
        if 'file' not in request.files:
            return jsonify({"error": "No file uploaded"}), 400

        file = request.files['file']
        file_path = f"/tmp/{file.filename}"
        file.save(file_path)

        print(f"[INFO] File received: {file_path}")

        # Summarize the file before uploading
        summarized_file_path = summarize_sonar_issues(file_path)
        if not summarized_file_path:
            return jsonify({"error": "Failed to summarize the file"}), 500

        # Upload the summarized file
        file_id = upload_file_to_openai(summarized_file_path)
        if file_id is None:
            return jsonify({"error": "Failed to upload summarized file to OpenAI"}), 500

        # Analyze the summarized file
        summary = analyze_sonarqube_with_assistant(file_id)
        if summary is None:
            return jsonify({"error": "Failed to process file with OpenAI"}), 500

        parsed = parse_sonarqube_summary(summary)

        return jsonify({
            "summary": summary,
            "parsed_issues": parsed
        })

    except Exception as e:
        print(f"[ERROR] SonarQube Analysis Failed: {e}")
        return jsonify({"error": "Unable to process SonarQube analysis"}), 500

# ------------------------
# Code_review related
# ------------------------


def parse_code_review_response(response_text):
    """Extracts score, feedback, revised code, and YouTube link from GPT response."""
    score_match = re.search(r"Score:\s*(\d+)/100", response_text)
    feedback_match = re.search(r"Feedback:\n(.+?)\n\nSuggested Revised Code:", response_text, re.DOTALL)
    code_match = re.search(r"Suggested Revised Code:\n```(?:\w+)?\n(.+?)\n```", response_text, re.DOTALL)
    youtube_match = re.search(r"Recommended YouTube Video:\n(https?://[^\s]+)", response_text)

    return {
        "score": int(score_match.group(1)) if score_match else None,
        "feedback": feedback_match.group(1).strip() if feedback_match else "",
        "revised_code": code_match.group(1).strip() if code_match else "",
        "youtube_link": youtube_match.group(1).strip() if youtube_match else None
    }

@app.route('/code_review', methods=['POST'])
def code_review():
    try:
        data = request.get_json()
        code = data.get("code", "")

        if not code.strip():
            return jsonify({"error": "No code provided"}), 400

        prompt = f"""
                You are a senior software engineer and mentor. Review the following code snippet and respond with:

                1. **Score out of 100**, evaluating modularity, naming conventions, readability, and maintainability.
                2. **Feedback**, including positive aspects and improvement suggestions.
                3. **A revised version** of the code, improving quality while maintaining logic.
                4. **A direct YouTube video link** that would help the developer learn about their biggest area for improvement. Use an actual link you've seen before or one that would help them most. Only put the link, no description

                Format your response **exactly** like this:

                Score: <score>/100

                Feedback:
                <your feedback>

                Suggested Revised Code:
                <better version of the code>

                Recommended YouTube Video:
                <youtube_link>
                
                Here is the code snippet I want you to review
                {code}
                
            """
        client = openai.OpenAI()
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a software engineering mentor and coding tutor."},
                {"role": "user", "content": prompt}
            ]
        )

        response_text = response.choices[0].message.content
        parsed = parse_code_review_response(response_text)

        return jsonify({
            "score": parsed["score"],
            "feedback": parsed["feedback"],
            "revised_code": parsed["revised_code"],
            "youtube_link": parsed["youtube_link"],
            # "raw_response": response_text  # Optional
        })

    except Exception as e:
        print(f"[ERROR] Code Review Failed: {e}")
        return jsonify({"error": "Unable to analyze code"}), 500
    
# ------------------------
# Chatbot related
# ------------------------

# call this when starting a chat, receive thread ID for it.
@app.route('/start_chat_thread', methods=['POST'])
def start_chat_thread():
    try:
        data = request.get_json()
        code_snippet = data.get("code", "")
        if not code_snippet:
            return jsonify({"error": "No code provided"}), 400

        client = openai.OpenAI()

        # Create thread
        thread = client.beta.threads.create()
        thread_id = thread.id

        # Add initial message with code context
        client.beta.threads.messages.create(
            thread_id=thread_id,
            role="user",
            content=f"""
You are a software engineering tutor. The user will ask questions about the following code:

{code_snippet}

Please keep this code in memory and help explain, improve, or clarify it as needed.
"""
        )

        return jsonify({"thread_id": thread_id})

    except Exception as e:
        print(f"[ERROR] start_chat_thread failed: {e}")
        return jsonify({"error": "Failed to start chat thread"}), 500

# upon receiving thread_id and message, it returns chatbot response.
@app.route('/chat_in_thread', methods=['POST'])
def chat_in_thread():
    try:
        data = request.get_json()
        thread_id = data.get("thread_id")
        user_message = data.get("message")

        if not thread_id or not user_message:
            return jsonify({"error": "Missing thread_id or message"}), 400

        client = openai.OpenAI()

        # Add user message
        client.beta.threads.messages.create(
            thread_id=thread_id,
            role="user",
            content=user_message
        )

        # Run assistant using the preset ID
        run = client.beta.threads.runs.create(
            thread_id=thread_id,
            assistant_id=CODE_CHAT_ASSISTANT_ID
        )

        # Wait for completion
        while True:
            status = client.beta.threads.runs.retrieve(
                thread_id=thread_id,
                run_id=run.id
            )
            if status.status == "completed":
                break
            time.sleep(1)

        # Get final assistant response
        messages = client.beta.threads.messages.list(thread_id=thread_id)
        response = messages.data[0].content[0].text.value

        return jsonify({"response": response})

    except Exception as e:
        print(f"[ERROR] Chat in thread failed: {e}")
        return jsonify({"error": "Chat failed"}), 500    

# ------------------------
# Compete Mode Related
# ------------------------

@app.route('/generate_ai_challenger_code', methods=['POST'])
def generate_ai_challenger_code():
    try:
        data = request.get_json()
        code = data.get("code", "")

        if not code.strip():
            return jsonify({"error": "No code provided"}), 400

        prompt = f"""
        You are a junior developer tasked with reimplementing a code snippet that someone wrote.

        Step 1: First detect what programming language this is (it may not be Python).
        Step 2: Reimplement the code in the **same language** with these guidelines:
        - The logic must still work correctly.
        - It must be **better than the original** (e.g. fixed bugs, improved logic, less code smell).
        - But write it like a junior developer, include one or more minor falloffs like: less modularity, short unclear variable names, some duplication, inconsistent formatting, and no comments or docstrings.

        Output only the reimplemented code — no backticks like ```python, java, etc., no explanation.

        Original Code:
        {code}
        """

        client = openai.OpenAI()
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are an inexperienced but improving junior developer."},
                {"role": "user", "content": prompt}
            ]
        )

        flawed_code = response.choices[0].message.content.strip()

        return jsonify({
            "ai_challenger_code": flawed_code
        })

    except Exception as e:
        print(f"[ERROR] AI Challenger Generation Failed: {e}")
        return jsonify({"error": "Unable to generate AI challenger code"}), 500
    
@app.route('/judge_code_competition', methods=['POST'])
def judge_code_competition():
    try:
        data = request.get_json()
        user_code = data.get("user_code", "")
        ai_code = data.get("ai_code", "")

        if not user_code.strip() or not ai_code.strip():
            return jsonify({"error": "Both user_code and ai_code are required"}), 400

        prompt = f"""
            You are a senior software engineer evaluating two implementations of the same function.
            They may be in any language — your job is to recognize the language and review accordingly.

            ### Evaluation Criteria:
            - Naming conventions
            - Duplicated logic
            - Modularity / readability
            - Bugs or inefficiencies
            - Maintainability
            - Scalability
            - Other common code smells

            ### Your Task:
            Determine which implementation is better, and provide a structured breakdown.

            Return your response in EXACTLY the following format. Do NOT change the words.

            Winner: <User or AI>

            UserPros:
            --- <bullet 1>
            --- <bullet 2>
            --- <bullet 3> (Optional)

            UserCons:
            --- <bullet 1>
            --- <bullet 2>
            --- <bullet 3> (Optional)

            AIPros:
            --- <bullet 1>
            --- <bullet 2>
            --- <bullet 3> (Optional)

            AICons:
            --- <bullet 1>
            --- <bullet 2>
            --- <bullet 3> (Optional)

            Reason:
            <Final 1–2 sentence reason why one is better>

            ----------

            User Code:
            {user_code}

            AI Code:
            {ai_code}
            """

        client = openai.OpenAI()
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a veteran software architect acting as a code competition judge."},
                {"role": "user", "content": prompt}
            ]
        )

        content = response.choices[0].message.content.strip()

        # Parsing each section using regex
        winner = re.search(r"Winner:\s*(User|AI)", content)

        user_pros = re.search(r"UserPros:\s*(.*?)(?:\n\s*UserCons:)", content, re.DOTALL)
        user_cons = re.search(r"UserCons:\s*(.*?)(?:\n\s*AIPros:)", content, re.DOTALL)
        ai_pros   = re.search(r"AIPros:\s*(.*?)(?:\n\s*AICons:)", content, re.DOTALL)
        ai_cons   = re.search(r"AICons:\s*(.*?)(?:\n\s*Reason:)", content, re.DOTALL)
        reason    = re.search(r"Reason:\s*(.*)", content, re.DOTALL)


        # Return nicely structured JSON
        return jsonify({
            "winner": winner.group(1) if winner else "Unknown",
            "user_pros": user_pros.group(1).strip() if user_pros else "[Missing]",
            "user_cons": user_cons.group(1).strip() if user_cons else "[Missing]",
            "ai_pros": ai_pros.group(1).strip() if ai_pros else "[Missing]",
            "ai_cons": ai_cons.group(1).strip() if ai_cons else "[Missing]",
            "reason": reason.group(1).strip() if reason else "[Missing]",
            #"raw_response": content
        })

    except Exception as e:
        print(f"[ERROR] Judge Evaluation Failed: {e}")
        return jsonify({"error": "Unable to judge code comparison"}), 500
    
if __name__ == '__main__':
    print("[INFO] Flask server is running on http://127.0.0.1:5000")
    app.run(debug=True)