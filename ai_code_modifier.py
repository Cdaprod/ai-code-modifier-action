import os
import requests
import subprocess
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, ValidationError

# Load environment variables
GITHUB_TOKEN = os.getenv('GITHUB_TOKEN')
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
REPO_OWNER = "Cdaprod"
REPO_NAME = "repocate"

# GitHub API Headers
headers = {
    "Authorization": f"token {GITHUB_TOKEN}",
    "Accept": "application/vnd.github.v3+json"
}

# Define the Output Model for Pydantic
class AISuggestion(BaseModel):
    modified_code: str

# Setup the AI model and output parser
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.0, api_key=OPENAI_API_KEY)
parser = PydanticOutputParser(pydantic_object=AISuggestion)

def create_or_switch_branch(branch_name):
    """Create or switch to a new branch for AI suggestions."""
    subprocess.run(["git", "fetch"], check=True)
    branches = subprocess.run(["git", "branch"], capture_output=True, text=True).stdout.splitlines()
    if branch_name not in branches:
        subprocess.run(["git", "checkout", "-b", branch_name], check=True)
    else:
        subprocess.run(["git", "checkout", branch_name], check=True)

def get_failed_workflow_run_logs(run_id):
    """Fetch workflow run logs for a failed run."""
    run_url = f"https://api.github.com/repos/{REPO_OWNER}/{REPO_NAME}/actions/runs/{run_id}"
    run_response = requests.get(run_url, headers=headers)

    if run_response.status_code == 200:
        run_data = run_response.json()
        if run_data['conclusion'] == 'failure':
            logs_url = run_data['logs_url']
            logs_response = requests.get(logs_url, headers=headers)
            if logs_response.status_code == 200:
                return logs_response.content
    return None

def parse_failure_logs(logs):
    """Parse logs to extract error messages."""
    error_pattern = re.compile(r"error[:\s](.*)", re.IGNORECASE)
    errors = error_pattern.findall(logs.decode('utf-8'))
    return errors

def generate_ai_suggestions(errors):
    """Generate code suggestions using OpenAI based on errors."""
    # Prepare dynamic content for the prompt
    error_text = "\n".join(errors)

    # Use the formatted prompt template
    chat_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "You are a helpful assistant. Respond with JSON format as {\"modified_code\": \"<fixed code here>\"}."),
            MessagesPlaceholder(variable_name="conversation"),
            ("user", f"The following errors were encountered during a build step:\n\n{error_text}\n\nProvide the minimal code changes required to fix these issues.")
        ]
    )
    
    formatted_prompt = chat_prompt.format_messages(conversation=[{"role": "user", "content": error_text}])
    response = llm.invoke(formatted_prompt)

    try:
        suggestion = parser.parse(response)
        return suggestion
    except ValidationError as e:
        print(f"Validation Error while parsing AI response: {e}")
        return None

def apply_ai_suggestions(suggestion):
    """Apply the suggested code changes to the repository."""
    if suggestion and suggestion.modified_code:
        with open("modified_code.py", "w") as file:
            file.write(suggestion.modified_code)
        # Assuming changes are ready to be staged
        commit_changes("ai-suggestions", "Apply AI suggestions")

def commit_changes(branch_name, message="Apply AI suggestions"):
    """Commit changes to the specified branch."""
    subprocess.run(["git", "add", "."], check=True)
    subprocess.run(["git", "commit", "-m", message], check=True)
    subprocess.run(["git", "push", "-u", "origin", branch_name], check=True)

# Example Workflow Run ID
run_id = os.getenv('RUN_ID')  # Adjust as needed
logs = get_failed_workflow_run_logs(run_id)
if logs:
    errors = parse_failure_logs(logs)
    if errors:
        ai_response = generate_ai_suggestions(errors)
        if ai_response:
            print("Suggested modified code:\n", ai_response.modified_code)
            create_or_switch_branch("ai-suggestions")
            apply_ai_suggestions(ai_response)
        else:
            print("No valid suggestion was generated.")
    else:
        print("No errors found in logs.")
else:
    print("Failed to retrieve logs for the workflow run.")