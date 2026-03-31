#ask us if apikey is needed
apikey_team = "*************"

import json
import os
import re
import time
import anthropic
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import defaultdict

# ── CONFIG ────────────────────────────────────────────────────────────────────
API_KEY = apikey_team

json_dir       = r"C:\Users\Rohank\Documents\UC DAVIS\HACKATHON\personas_json"
test_json_path = r"C:\Users\Rohank\Documents\UC DAVIS\HACKATHON\final_test_questions.json"
output_path    = r"C:\Users\Rohank\Documents\UC DAVIS\HACKATHON\predictions.json"

MODEL = "claude-sonnet-4-20250514"

MAX_WORKERS = 4
REQUESTS_PER_MIN = 100   # slightly higher since batching reduces load
BATCH_SIZE = 5
MAX_RETRIES = 5

client = anthropic.Anthropic(api_key=apikey_team)

# ── RATE LIMITER ──────────────────────────────────────────────────────────────
rate_lock = threading.Lock()
last_call_time = [0]
DELAY = 60 / REQUESTS_PER_MIN

def throttle():
    with rate_lock:
        now = time.time()
        elapsed = now - last_call_time[0]
        if elapsed < DELAY:
            time.sleep(DELAY - elapsed)
        last_call_time[0] = time.time()

# ── SAFE API CALL ─────────────────────────────────────────────────────────────
def safe_call(messages, max_tokens=200):
    for attempt in range(MAX_RETRIES):
        try:
            throttle()
            return client.messages.create(
                model=MODEL,
                max_tokens=max_tokens,
                messages=messages
            )
        except Exception as e:
            if "rate_limit" in str(e):
                time.sleep(2 ** attempt)
            else:
                raise e
    return None

# ── HELPERS ───────────────────────────────────────────────────────────────────
def get_answer_text(question):
    answers = question.get("Answers", {})
    if not answers:
        return None

    if "Text" in answers:
        t = answers["Text"]
        if isinstance(t, list):
            return " | ".join(list(x.values())[0] for x in t if x)
        return str(t)

    selected = answers.get("SelectedText")
    if isinstance(selected, list):
        return str(selected[0]) if selected else None
    return str(selected) if selected else None


def build_profile(data):
    lines = []
    for block in data:
        for q in block.get("Questions", []):
            if q.get("is_descriptive") or q.get("QuestionType") == "DB":
                continue
            ans = get_answer_text(q)
            qtext = q.get("QuestionText", "")[:60]
            if ans:
                lines.append(f"{qtext}: {ans}")
    return "\n".join(lines[:50])


def compress_persona(profile_text):
    prompt = f"""
Summarize this person into a short behavioral profile (max 120 words).

PROFILE:
{profile_text}
"""
    msg = safe_call([{"role": "user", "content": prompt}], max_tokens=150)
    return msg.content[0].text.strip() if msg else ""


# ── BATCH PREDICTION ──────────────────────────────────────────────────────────
def predict_batch(profile, batch_questions):
    prompt_parts = [f"PERSON:\n{profile}\n"]

    for i, q in enumerate(batch_questions):
        options_str = "\n".join(q["options"])
        prompt_parts.append(f"""
Q{i+1}: {q['question_text']}

Options:
{options_str}
""")

    prompt_parts.append("""
Answer ALL questions.

Return ONLY in this format:
1: <option number>
2: <option number>
3: <option number>
...
""")

    prompt = "\n".join(prompt_parts)

    msg = safe_call([{"role": "user", "content": prompt}], max_tokens=200)

    if not msg:
        return [None] * len(batch_questions)

    text = msg.content[0].text.strip()

    answers = []
    for i in range(len(batch_questions)):
        match = re.search(rf"{i+1}:\s*(\d+)", text)
        answers.append(match.group(1) if match else None)

    return answers


# ── MAIN ──────────────────────────────────────────────────────────────────────
def run():

    print("Loading data...")
    with open(test_json_path) as f:
        test_questions = json.load(f)

    personas = {}
    for filename in os.listdir(json_dir):
        if filename.endswith("_persona.json"):
            pid = filename.replace("_persona.json", "")
            with open(os.path.join(json_dir, filename)) as f:
                personas[pid] = json.load(f)

    # ── BUILD COMPRESSED PROFILES ─────────────────────────────────────────────
    print("Compressing personas...")
    profile_cache = {}

    for pid in set(q["person_id"] for q in test_questions):
        if pid not in personas:
            continue

        raw = build_profile(personas[pid])
        profile_cache[pid] = compress_persona(raw)

    print(f"✅ {len(profile_cache)} personas ready")

    # ── GROUP QUESTIONS BY PERSON ─────────────────────────────────────────────
    grouped = defaultdict(list)
    for idx, tq in enumerate(test_questions):
        grouped[tq["person_id"]].append((idx, tq))

    results = [None] * len(test_questions)
    lock = threading.Lock()
    counter = [0]
    total = len(test_questions)

    def worker(person_id, items):
        profile = profile_cache.get(person_id)
        if not profile:
            return

        # batch per person
        for i in range(0, len(items), BATCH_SIZE):
            batch = items[i:i+BATCH_SIZE]

            questions = [x[1] for x in batch]
            indices = [x[0] for x in batch]

            preds = predict_batch(profile, questions)

            for idx, tq, pred in zip(indices, questions, preds):
                if pred:
                    full = next(
                        (o for o in tq["options"] if o.startswith(pred)),
                        pred
                    )
                    tq["predicted_answer"] = full
                else:
                    tq["predicted_answer"] = None

                results[idx] = tq

                with lock:
                    counter[0] += 1
                    if counter[0] % 100 == 0:
                        print(f"[{counter[0]}/{total}]")

    print("Running batched predictions...")

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futures = [
            ex.submit(worker, pid, items)
            for pid, items in grouped.items()
        ]
        for f in as_completed(futures):
            f.result()

    # ── SAVE ──────────────────────────────────────────────────────────────────
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    print("✅ Done")


# ── RUN ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    run()
