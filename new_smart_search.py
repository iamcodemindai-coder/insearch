import json
import os
import re
import time
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI


# ============================================================
# CONFIG
# ============================================================

load_dotenv()

API_KEY = os.getenv("OPENAI_API_KEY")
BASE_URL = os.getenv("OPENAI_BASE_URL")
MODEL = os.getenv("OPENAI_MODEL", "gpt-5")

if not API_KEY:
    raise RuntimeError("OPENAI_API_KEY is missing in .env")


client_config = {"api_key": API_KEY}

if BASE_URL:
    client_config["base_url"] = BASE_URL

client = OpenAI(**client_config)


# ============================================================
# SYSTEM PROMPT
# ============================================================

SYSTEM_PROMPT = """
You are a high-precision semantic similarity evaluator.

Your task is to compare ONE CURRENT ISSUE SUMMARY against
MULTIPLE HISTORICAL NOTIFICATIONS.

Return one score for EVERY notification.

The score represents semantic relevance between the complete
meaning of the current issue and the complete meaning of the
historical notification.

============================================================
IMPORTANT: GENERIC VS SPECIFIC INFORMATION
============================================================

A notification can contain:

1. Generic information
2. Specific technical/problem information

Example:

"damaged material (damaged lead screw)"

Generic:
"damaged material"

Specific:
"damaged lead screw"

The generic part should NOT dominate the score.

However, do NOT completely ignore the generic part either.

Use the COMPLETE TEXT.

============================================================
CRITICAL GENERIC CASE
============================================================

Current issue:

"damaged material"

Historical notifications:

"damaged material (misaligned hook talons)"
"damaged material (soldering issues)"
"damaged material (high SMA wire resistance)"
"damaged material (damaged lead screw)"

The current issue is generic-only.

Therefore:

- Do NOT assume that any specific historical subtype is the
  actual current problem.
- Do NOT give these records 90+ merely because they contain
  "damaged material".
- BUT also do NOT force every record to exactly the same score.
- Evaluate the complete semantic content of each notification.
- Preserve meaningful differences in semantic closeness.
- Generic category similarity contributes to relevance.
- Specific information refines the score.

The goal is RELATIVE semantic relevance, not keyword counting.

============================================================
VERY IMPORTANT: DO NOT FLATTEN SCORES
============================================================

Do NOT return:

25, 25, 25, 25

just because all notifications contain the same generic phrase.

If the complete meanings have different semantic closeness,
their scores should be different.

For example, a reasonable result may look like:

65
61
68
73

rather than:

25
25
25
25

Exact values are your judgement.

============================================================
SPECIFIC MATCH
============================================================

If both records contain the same specific problem:

Current:
"damaged material (soldering issue)"

Historical:
"damaged material (soldering problem)"

=> HIGH score.

If specific problems are different:

Current:
"damaged material (soldering issue)"

Historical:
"damaged material (battery corrosion)"

=> LOW/MODERATE score.

The shared generic phrase alone is not enough for a high score.

============================================================
SEMANTIC UNDERSTANDING
============================================================

Understand:

- synonyms
- paraphrases
- technical terminology
- abbreviations
- equivalent descriptions
- word order
- grammar
- complete technical meaning

Example:

"high SMA wire resistance"

and

"SMA wire has high resistance"

are highly similar.

============================================================
SCORING GUIDELINES
============================================================

95-100:
Almost identical complete meaning.

85-94:
Very strong semantic match.

70-84:
Strong semantic match.

55-69:
Moderate-to-strong semantic relationship.

40-54:
Moderate relationship.

25-39:
Weak relationship / mostly generic relationship.

10-24:
Very weak relationship.

0-9:
Essentially unrelated.

IMPORTANT:

These are guidelines, NOT fixed buckets.

Do not automatically assign the same score to records in
the same category.

Do not automatically assign 25%.

============================================================
EMBEDDING-LIKE RELATIVE BEHAVIOUR
============================================================

The historical records may all share a generic phrase.

Still compare their complete text and preserve relative
semantic closeness.

If one notification is semantically closer to the current
issue than another, its score should normally be higher.

The purpose is to produce a meaningful relevance ranking.

Do NOT try to reproduce a specific embedding cosine value.

For example:

Embedding scores:

0.64
0.64
0.67
0.73

The LLM does NOT need to output exactly:

64
64
67
73

But it should preserve meaningful relative differences when
the text supports them.

============================================================
MISSING INFORMATION
============================================================

Never invent information.

If the current issue does not mention a specific component,
do not assume that component is the current problem.

============================================================
BLANK NOTIFICATIONS
============================================================

If pcm_inv_notif is:

null
""
whitespace
or missing completely

the score MUST be exactly 0.

Do not fail.

============================================================
OUTPUT
============================================================

Return ONLY valid JSON:

{
    "results": [
        {
            "pcm_inv_notif_id": "123",
            "score": 72
        }
    ]
}

Rules:

- one result for every notification
- preserve exact notification ID
- score must be integer 0-100
- blank/missing notification = 0
- no explanation
- no markdown
- no additional fields
"""


# ============================================================
# HELPERS
# ============================================================

def normalize(text):
    if text is None:
        return ""

    return re.sub(
        r"\s+",
        " ",
        str(text).strip()
    )


def load_json(path):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    except FileNotFoundError:
        raise RuntimeError(f"Input file not found: {path}")

    except json.JSONDecodeError as e:
        raise RuntimeError(
            f"Invalid JSON: {e}"
        )


def save_json(data, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(
            data,
            f,
            indent=4,
            ensure_ascii=False
        )


# ============================================================
# EXTRACT NOTIFICATIONS
# ============================================================

def get_notifications(search_results):

    notifications = []

    for index, item in enumerate(search_results):

        if not isinstance(item, dict):
            print(
                f"COMMENT: search_results[{index}] "
                f"is invalid. Skipping."
            )
            continue

        notif_id = item.get("pcm_inv_notif_id")

        if notif_id is not None:
            notif_id = normalize(notif_id)

        text = normalize(
            item.get("pcm_inv_notif")
        )

        notifications.append({
            "index": index,
            "id": notif_id,
            "text": text,
            "missing": not bool(text)
        })

        if not text:
            print(
                f"COMMENT: Notification missing/blank "
                f"for ID {notif_id}. Score = 0."
            )

    return notifications


# ============================================================
# LLM SIMILARITY
# ============================================================

def calculate_similarity(
    issue_summary,
    notifications
):

    valid_notifications = [
        x for x in notifications
        if x["text"] and x["id"]
    ]

    # --------------------------------------------------------
    # If nothing valid exists, no LLM call required
    # --------------------------------------------------------

    if not valid_notifications:

        return {
            "results": [
                {
                    "pcm_inv_notif_id": x["id"],
                    "score": 0
                }
                for x in notifications
                if x["id"]
            ]
        }, 0.0

    # --------------------------------------------------------
    # Prepare records
    # --------------------------------------------------------

    records = []

    for x in valid_notifications:

        records.append({
            "pcm_inv_notif_id": x["id"],
            "pcm_inv_notif": x["text"]
        })

    user_prompt = f"""
CURRENT ISSUE SUMMARY:

{issue_summary}


HISTORICAL NOTIFICATIONS:

{json.dumps(
    records,
    indent=2,
    ensure_ascii=False
)}


TASK:

Evaluate EVERY historical notification independently against
the current issue.

Then calibrate the scores across the complete set so that
meaningful differences between notifications are preserved.

IMPORTANT:

If the current issue is:

"damaged material"

and notifications contain:

"damaged material (misaligned hook talons)"
"damaged material (soldering issues)"
"damaged material (high SMA wire resistance)"
"damaged material (damaged lead screw)"

do NOT return identical scores simply because
"damaged material" is repeated.

The generic category should contribute to similarity, while
the complete notification text should refine the score.

Do not assume any specific subtype is the actual current issue.

Return one score for every notification.

Return ONLY JSON.
"""

    start = time.perf_counter()

    try:

        response = client.chat.completions.create(
            model=MODEL,
            messages=[
                {
                    "role": "system",
                    "content": SYSTEM_PROMPT
                },
                {
                    "role": "user",
                    "content": user_prompt
                }
            ],
            response_format={
                "type": "json_object"
            }
        )

    except Exception as e:

        raise RuntimeError(
            f"LLM API call failed: {e}"
        )

    llm_time = time.perf_counter() - start

    # --------------------------------------------------------
    # Read response
    # --------------------------------------------------------

    content = (
        response
        .choices[0]
        .message
        .content
    )

    if not content:
        raise RuntimeError(
            "LLM returned empty response."
        )

    content = content.strip()

    # Remove accidental markdown fences
    content = re.sub(
        r"^```(?:json)?",
        "",
        content,
        flags=re.IGNORECASE
    )

    content = re.sub(
        r"```$",
        "",
        content
    ).strip()

    try:
        result = json.loads(content)

    except json.JSONDecodeError as e:

        raise RuntimeError(
            f"Invalid JSON returned by LLM:\n{content}"
        ) from e

    if not isinstance(result, dict):
        raise RuntimeError(
            "LLM response must be an object."
        )

    results = result.get("results")

    if not isinstance(results, list):
        raise RuntimeError(
            "LLM response must contain 'results' list."
        )

    # --------------------------------------------------------
    # Validate IDs and scores
    # --------------------------------------------------------

    expected_ids = {
        x["id"]
        for x in valid_notifications
    }

    received = {}

    for item in results:

        if not isinstance(item, dict):
            raise RuntimeError(
                "Invalid result object returned by LLM."
            )

        notif_id = normalize(
            item.get("pcm_inv_notif_id")
        )

        if not notif_id:
            raise RuntimeError(
                "LLM returned result without notification ID."
            )

        if notif_id not in expected_ids:
            raise RuntimeError(
                f"Unexpected notification ID: {notif_id}"
            )

        if notif_id in received:
            raise RuntimeError(
                f"Duplicate notification ID: {notif_id}"
            )

        try:
            score = int(
                round(
                    float(item.get("score"))
                )
            )

        except (
            TypeError,
            ValueError
        ):

            raise RuntimeError(
                f"Invalid score for {notif_id}"
            )

        if not 0 <= score <= 100:
            raise RuntimeError(
                f"Score out of range for {notif_id}: {score}"
            )

        received[notif_id] = score

    # --------------------------------------------------------
    # Make sure every valid notification got score
    # --------------------------------------------------------

    missing = expected_ids - set(received)

    if missing:
        raise RuntimeError(
            f"LLM missed notification IDs: {missing}"
        )

    # --------------------------------------------------------
    # Rebuild results in ORIGINAL order
    # --------------------------------------------------------

    final_results = []

    for x in notifications:

        notif_id = x["id"]

        if not notif_id:
            continue

        if x["missing"]:

            score = 0

        else:

            score = received.get(
                notif_id,
                0
            )

        final_results.append({
            "pcm_inv_notif_id": notif_id,
            "score": score
        })

    return {
        "results": final_results
    }, llm_time


# ============================================================
# ADD SCORE TO ORIGINAL JSON
# ============================================================

def add_scores(
    original_data,
    similarity_result
):

    score_map = {
        str(x["pcm_inv_notif_id"]): int(x["score"])
        for x in similarity_result["results"]
        if x.get("pcm_inv_notif_id") is not None
    }

    output = dict(original_data)

    updated = []

    for item in original_data.get(
        "search_results",
        []
    ):

        if not isinstance(item, dict):
            updated.append(item)
            continue

        item_copy = {}
        notif_id = normalize(
            item.get("pcm_inv_notif_id")
        )

        score = score_map.get(
            notif_id,
            0
        )

        inserted = False

        for key, value in item.items():

            item_copy[key] = value

            if (
                key == "pcm_inv_notif"
                and not inserted
            ):

                item_copy["score"] = (
                    f"{score}% match"
                )

                inserted = True

        if not inserted:

            item_copy["score"] = (
                f"{score}% match"
            )

        updated.append(item_copy)

    output["search_results"] = updated

    return output


# ============================================================
# MAIN PROCESS
# ============================================================

def process(data):

    if not isinstance(data, dict):
        raise ValueError(
            "Input JSON must be an object."
        )

    issue_summary = normalize(
        data.get("pcm_issue_summary")
    )

    if not issue_summary:
        raise ValueError(
            "pcm_issue_summary is missing/empty."
        )

    search_results = data.get(
        "search_results"
    )

    if not isinstance(
        search_results,
        list
    ):
        raise ValueError(
            "search_results must be a list."
        )

    notifications = get_notifications(
        search_results
    )

    print("\n" + "=" * 70)
    print("INPUT")
    print("=" * 70)

    print(
        f"Issue Summary : {issue_summary}"
    )

    print(
        f"Records       : {len(notifications)}"
    )

    # --------------------------------------------------------
    # LLM
    # --------------------------------------------------------

    similarity_result, llm_time = (
        calculate_similarity(
            issue_summary,
            notifications
        )
    )

    # --------------------------------------------------------
    # Add scores
    # --------------------------------------------------------

    output = add_scores(
        data,
        similarity_result
    )

    return output, llm_time


# ============================================================
# MAIN
# ============================================================

def main():

    input_file = Path(
        "example_input.json"
    )

    output_file = Path(
        "output.json"
    )

    print("=" * 70)
    print("GENAI SEMANTIC SIMILARITY")
    print("=" * 70)

    total_start = time.perf_counter()

    data = load_json(
        input_file
    )

    output, llm_time = process(
        data
    )

    save_json(
        output,
        output_file
    )

    total_time = (
        time.perf_counter()
        - total_start
    )

    # --------------------------------------------------------
    # PRINT SCORES
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("SCORES")
    print("=" * 70)

    for item in output.get(
        "search_results",
        []
    ):

        print(
            f"{item.get('pcm_inv_notif_id')} "
            f"-> {item.get('score', '0% match')}"
        )

    print("\n" + "=" * 70)
    print("PERFORMANCE")
    print("=" * 70)

    print(
        f"LLM response time : {llm_time:.3f} sec"
    )

    print(
        f"Total time        : {total_time:.3f} sec"
    )

    print(
        f"Output saved      : {output_file}"
    )

    print("\n" + "=" * 70)
    print("FINAL JSON")
    print("=" * 70)

    print(
        json.dumps(
            output,
            indent=4,
            ensure_ascii=False
        )
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()
