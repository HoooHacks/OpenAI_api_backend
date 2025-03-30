from flask import Flask, request, jsonify
import openai
from dotenv import load_dotenv
import os
import json
import time
import re

# Load environment variables and set OpenAI API key
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")
# Initialize Flask app
app = Flask(__name__)

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


if __name__ == '__main__':
    print("[INFO] Flask server is running on http://127.0.0.1:5000")
    app.run(debug=True)