import os
import json
import urllib.error
import urllib.request

OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434").rstrip("/")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen3.5:9b")
_ACTIVE_OLLAMA_HOST = None


def read(file):
    with open(file, 'r', encoding="utf8") as f:
        text = f.read()
    return text

def write_answers(file, output):
    with open(file, 'a', encoding='utf-8', errors='ignore') as f:
        f.write(output)

def read_json_file(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
    return data

def write_json_file(data, file_path):
    with open(file_path, 'w') as file:
        json.dump(data, file, indent=4)

def _normalize_ollama_host(host):
    host = host.strip().rstrip("/")
    if not host.startswith(("http://", "https://")):
        host = f"http://{host}"
    return host

def _is_wsl():
    try:
        with open("/proc/version", "r", encoding="utf-8") as file:
            version = file.read().lower()
        return "microsoft" in version or "wsl" in version
    except OSError:
        return False

def _windows_host_from_wsl():
    try:
        with open("/etc/resolv.conf", "r", encoding="utf-8") as file:
            for line in file:
                parts = line.strip().split()
                if len(parts) >= 2 and parts[0] == "nameserver":
                    return parts[1]
    except OSError:
        return None
    return None

def _ollama_host_candidates():
    if os.getenv("OLLAMA_HOST"):
        return [_normalize_ollama_host(os.getenv("OLLAMA_HOST"))]

    candidates = [_normalize_ollama_host(OLLAMA_HOST), "http://127.0.0.1:11434"]
    if _is_wsl():
        windows_host = _windows_host_from_wsl()
        if windows_host:
            candidates.append(f"http://{windows_host}:11434")
        candidates.append("http://host.docker.internal:11434")

    deduped = []
    for host in candidates:
        if host not in deduped:
            deduped.append(host)
    return deduped

def _request_json(host, path, payload=None, timeout=300):
    data = None
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")

    request = urllib.request.Request(
        f"{host}{path}",
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST" if payload is not None else "GET",
    )

    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))

def _active_ollama_host():
    global _ACTIVE_OLLAMA_HOST
    if _ACTIVE_OLLAMA_HOST:
        return _ACTIVE_OLLAMA_HOST

    errors = []
    for host in _ollama_host_candidates():
        try:
            _request_json(host, "/api/tags", timeout=3)
            _ACTIVE_OLLAMA_HOST = host
            return host
        except urllib.error.URLError as exc:
            errors.append(f"{host} ({exc.reason})")

    tried = ", ".join(errors) or ", ".join(_ollama_host_candidates())
    raise RuntimeError(
        f"Could not reach Ollama. Tried: {tried}. "
        "If Ollama is running in Windows PowerShell and this script runs in WSL, "
        "configure Windows Ollama to listen on 0.0.0.0:11434 and restart it."
    )

def _uses_default_temperature_only(model):
    normalized = model.lower().replace("_", "-").replace(" ", "-")
    return "gpt-5" in normalized or "gpt5" in normalized

def ollama_chat(messages, model=OLLAMA_MODEL, temperature=0.5, max_tokens=None):
    options = {}
    if temperature is not None and not _uses_default_temperature_only(model):
        options["temperature"] = temperature
    if max_tokens is not None:
        options["num_predict"] = max_tokens

    payload = {
        "model": model,
        "messages": messages,
        "stream": False,
        "think": False,
    }
    if options:
        payload["options"] = options

    host = _active_ollama_host()
    try:
        return _request_json(host, "/api/chat", payload=payload, timeout=300)
    except urllib.error.URLError as exc:
        raise RuntimeError(
            f"Could not reach Ollama at {host}. "
            f"Make sure Ollama is running and the {model} model is available."
        ) from exc

def make_prompt(title):
    if title == "basic":
        prompt  = " You are a knowledgeable local AI language model. Your goal is to provide concise, accurate, and helpful responses to questions, while being honest and straightforward."
    elif title == "concise":
        prompt  = " You are a local AI language model. Instructions: Answer factual questions concisely."
    elif title == "direction":
        prompt = "you are an expert life coach and need to classify the action to take from your client's statement. Answer with only the action to take, replace the 'xxx' with a title but no explanation. answer with one of these: create goal xxx, create value xxx, add activity xxx to goal 'goal'.  the client statement is:"
    elif title == "get_json":
        prompt = "you are an expert life coach and need to classify the action to take from your client's statement. Answer with only the action to take, replace the 'xxx' with a title but no explanation. answer with one of these: create goal xxx, create value xxx, add activity xxx to goal 'goal'.  the client statement is:"
    else:
        prompt = " Your function is to generate human-like text based on the inputs given and to assist users in generating informative, helpful and engaging responses to questions and requests. Whenever asking the user for his input as a question, only ask one question, wait for the answer, provide feedback and move on to the next question if applicable.  Please provide a detailed response with lists, where applicable. If you are not sure of the meaning of the question, ask clarifying questions.  "
    return prompt
  
def converse(request, conversation):
    model = conversation.get('model', conversation.get('engine', OLLAMA_MODEL))
    temperature = conversation['temperature']
    requestStr =  json.dumps(request)
    conversationStr =  json.dumps(conversation)
    prompt = [
        {"role": "system", "content": requestStr},
        {"role": "user", "content": conversationStr}
                      ]
    response = ollama_chat(prompt, model=model, temperature=temperature)
    answer = response["message"]["content"]
    # try to make the answer a json object
    try:
        answer = json.loads(answer)
    except:
        answer = {"nextCommunication": 'done'}
    return answer


    instructions = make_prompt(prompt_title)
    history = " our chat history: " + history 
    instructions + "  " + question   
    prompt = [
        {"role": "system", "content": instructions},
        {"role": "system", "content": history},
        {"role": "user", "content": question}
        ]
    response = ollama_chat(prompt, model=model, temperature=temperature, max_tokens=1500)
    return response["message"]["content"]