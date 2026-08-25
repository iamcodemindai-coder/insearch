import json
import os
import re
import time
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI


# ============================================================
# LOAD ENVIRONMENT
# ============================================================

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-5")


# ============================================================
# VALIDATE CONFIG
# ============================================================

if not OPENAI_API_KEY:
    raise RuntimeError(
        "OPENAI_API_KEY is not configured in .env file."
    )


# ============================================================
# OPENAI CLIENT
# ============================================================

client_config = {
    "api_key": OPENAI_API_KEY
}

if OPENAI_BASE_URL:
    client_config["base_url"] = OPENAI_BASE_URL

client = OpenAI(**client_config)


# ============================================================
# SYSTEM PROMPT
# ============================================================

SYSTEM_PROMPT = """
You are a HIGH-PRECISION semantic similarity evaluator.

Your task is ONLY to compare:

CURRENT ISSUE SUMMARY

against

ONE HISTORICAL NOTIFICATION.

This is NOT keyword matching.

This is NOT category matching.

This is NOT medical diagnosis.

You must determine how much of the actual problem meaning
is shared between the two texts.

============================================================
MOST IMPORTANT RULE
============================================================

GENERIC CATEGORY MUST NOT DOMINATE THE SCORE.

Many records may contain the same generic phrase.

Example:

Current:
"damaged material"

Historical:
"damaged material (battery corrosion)"

Historical:
"damaged material (extra spring)"

Historical:
"damaged material (ratchet gear)"

Historical:
"damaged material (soldering issue)"

The phrase:

"damaged material"

is GENERIC.

It does NOT identify the actual failure/problem.

The historical records contain specific information:

battery corrosion
extra spring
ratchet gear
soldering issue

The current issue does NOT contain any of those specific details.

Therefore, NONE of these records should receive a high score.

============================================================
GENERIC-ONLY CURRENT ISSUE
============================================================

This is a CRITICAL RULE.

If the current issue contains only a generic category such as:

"damaged material"

then a historical notification containing:

"damaged material (battery corrosion)"

must NOT receive 50%, 60%, 70%, 80%, 90% or 100%.

The correct interpretation is:

The category is related, but the actual problem is unknown.

Therefore use LOW similarity.

CALIBRATION:

Generic category only:
approximately 20-35

Generic category + unrelated/different specific subtype:
approximately 15-30

Generic category + same specific subtype:
HIGH ONLY if the current issue actually contains
that same specific information.

Do NOT invent missing information.

============================================================
EXAMPLE 1
============================================================

CURRENT:

"damaged material"

HISTORICAL:

"damaged material (battery corrosion)"

Score should be LOW.

Reason:

Both mention damaged material, but current issue does not
specify battery corrosion.

Do NOT assume battery corrosion is the issue.

============================================================
EXAMPLE 2
============================================================

CURRENT:

"damaged material"

HISTORICAL:

"damaged material (soldering issue)"

Score should be LOW.

The common phrase is generic.

The specific soldering problem is not present in the
current issue.

============================================================
EXAMPLE 3
============================================================

CURRENT:

"damaged material (soldering issue)"

HISTORICAL:

"damaged material (soldering problem)"

Score should be VERY HIGH.

The generic category matches AND the specific problem
has the same meaning.

============================================================
EXAMPLE 4
============================================================

CURRENT:

"damaged material (soldering issue)"

HISTORICAL:

"damaged material (battery corrosion)"

Score should be LOW.

The generic category matches but the specific failure
is different.

============================================================
EXAMPLE 5
============================================================

CURRENT:

"high SMA wire resistance"

HISTORICAL:

"SMA wire has high resistance"

Score should be VERY HIGH.

These have the same specific technical meaning.

============================================================
SPECIFICITY HAS PRIORITY
============================================================

When comparing two records, identify:

1. Generic category
2. Specific problem
3. Affected component
4. Failure mode
5. Cause
6. Symptom
7. Technical condition
8. Other distinctive technical information

Specific information is much more important than generic
information.

If specific information conflicts, similarity should be LOW
even when the generic category is identical.

============================================================
SCORING CALIBRATION
============================================================

95-100
Almost identical complete meaning and same specific issue.

85-94
Very strong semantic match with the same specific issue.

70-84
Strong semantic match with closely related specific issue.

50-69
Moderate similarity ONLY when meaningful specific information
is shared.

30-49
Weak relationship or partially shared information.

20-29
Generic/context relationship but no meaningful specific match.

10-19
Very weak relationship.

0-9
Essentially unrelated.

IMPORTANT:

Do NOT automatically use 50 as a default score.

Do NOT give the same score to every notification simply because
they share the same generic phrase.

============================================================
GENERIC-ONLY CURRENT SCORE RULE
============================================================

If CURRENT ISSUE SUMMARY is generic-only, for example:

"damaged material"

and HISTORICAL NOTIFICATION is:

"damaged material (battery corrosion)"

then score should normally remain in approximately:

20-35

range.

If the historical notification has a completely different
specific subtype, prefer the lower part of that range.

If the historical notification is also only:

"damaged material"

then it may receive a somewhat higher generic match,
but still do NOT treat it as a 90+ specific match.

============================================================
SEMANTIC MATCHING
============================================================

Understand:

- synonyms
- paraphrases
- abbreviations
- technical terminology
- equivalent phrases
- word order
- grammatical differences
- singular/plural differences
- complete technical meaning

Example:

"high resistance in SMA wire"

and

"SMA wire resistance is high"

are highly similar.

But:

"damaged material (soldering issue)"

and

"damaged material (battery corrosion)"

are NOT highly similar.

============================================================
DO NOT INVENT INFORMATION
============================================================

Only use information explicitly present in the texts.

If current issue does not mention:

battery
soldering
lead screw
ratchet
spring
sensor
etc.

DO NOT assume that any of these are the current issue.

============================================================
BLANK / MISSING NOTIFICATION
============================================================

If historical notification is:

null

""

whitespace

or the field is missing,

the score MUST be exactly:

0

Do not fail.

Do not generate an error.

============================================================
OUTPUT
============================================================

Return ONLY valid JSON.

Required format:

{
    "score": 27
}

The score must be an integer from 0 to 100.

Do not return:

%
explanation
reason
markdown
additional fields
additional text

============================================================
FINAL INTERNAL CHECK
============================================================

Before returning the score verify:

1. Did I compare actual meaning?
2. Did I identify generic vs specific information?
3. Did I avoid generic keyword dominance?
4. Did I avoid inventing missing information?
5. If current issue is generic-only, did I keep score low?
6. If specific problems are different, did I keep score low?
7. If specific problems are equivalent, did I give high score?
8. Is the score between 0 and 100?

Return ONLY:

{
    "score": integer
}
"""


# ============================================================
# TEXT NORMALIZATION
# ============================================================

def normalize_text(text):
    """
    Normalize text for internal processing only.
    Original JSON is never modified.
    """

    if text is None:
        return ""

    text = str(text).strip()

    text = re.sub(
        r"\s+",
        " ",
        text
    )

    return text


# ============================================================
# VALIDATE INPUT
# ============================================================

def validate_input_data(data):

    if not isinstance(data, dict):
        raise ValueError(
            "Input JSON must be a JSON object."
        )

    if "search_results" not in data:
        raise ValueError(
            "'search_results' is missing."
        )

    if not isinstance(
        data["search_results"],
        list
    ):
        raise ValueError(
            "'search_results' must be a list."
        )

    if "pcm_issue_summary" not in data:
        raise ValueError(
            "'pcm_issue_summary' is missing."
        )

    issue_summary = normalize_text(
        data.get("pcm_issue_summary")
    )

    if not issue_summary:
        raise ValueError(
            "'pcm_issue_summary' cannot be empty."
        )


# ============================================================
# FIND NOTIFICATIONS
# ============================================================

def find_notifications(search_results):

    notifications = []

    for index, item in enumerate(search_results):

        if not isinstance(item, dict):

            print(
                f"WARNING: search_results[{index}] "
                f"is not an object. Skipping."
            )

            continue

        notification_id = item.get(
            "pcm_inv_notif_id"
        )

        if notification_id is None:

            print(
                f"COMMENT: search_results[{index}] "
                f"has no pcm_inv_notif_id. "
                f"Cannot assign score."
            )

            notifications.append({
                "index": index,
                "pcm_inv_notif_id": None,
                "pcm_inv_notif": "",
                "missing_notification": True,
                "missing_id": True
            })

            continue

        notification_id = str(
            notification_id
        ).strip()

        if not notification_id:

            print(
                f"COMMENT: search_results[{index}] "
                f"has blank pcm_inv_notif_id."
            )

            notifications.append({
                "index": index,
                "pcm_inv_notif_id": None,
                "pcm_inv_notif": "",
                "missing_notification": True,
                "missing_id": True
            })

            continue

        # ----------------------------------------------------
        # Get notification
        # ----------------------------------------------------

        if "pcm_inv_notif" not in item:

            print(
                f"COMMENT: Notification missing for ID "
                f"{notification_id}. Score = 0."
            )

            notifications.append({
                "index": index,
                "pcm_inv_notif_id": notification_id,
                "pcm_inv_notif": "",
                "missing_notification": True,
                "missing_id": False
            })

            continue

        notification = normalize_text(
            item.get("pcm_inv_notif")
        )

        if not notification:

            print(
                f"COMMENT: Notification blank for ID "
                f"{notification_id}. Score = 0."
            )

            notifications.append({
                "index": index,
                "pcm_inv_notif_id": notification_id,
                "pcm_inv_notif": "",
                "missing_notification": True,
                "missing_id": False
            })

            continue

        notifications.append({
            "index": index,
            "pcm_inv_notif_id": notification_id,
            "pcm_inv_notif": notification,
            "missing_notification": False,
            "missing_id": False
        })

    return notifications


# ============================================================
# CALCULATE ONE NOTIFICATION SCORE
# ============================================================

def calculate_one_score(
    issue_summary,
    notification
):
    """
    Calculate similarity for ONE notification.

    Important:
    Every notification gets its own LLM evaluation.
    This prevents score anchoring caused by multiple
    notifications sharing the same generic phrase.
    """

    # --------------------------------------------------------
    # Blank notification = ZERO
    # --------------------------------------------------------

    if not notification:

        return 0, 0.0

    user_prompt = f"""
CURRENT ISSUE SUMMARY:

{issue_summary}


HISTORICAL NOTIFICATION:

{notification}


TASK:

Calculate the semantic similarity between the current issue
summary and this ONE historical notification.

Remember:

- Generic category is NOT the actual problem.
- Specific technical meaning is more important.
- Shared phrase such as "damaged material" alone must NOT
  create a high similarity.
- If current issue is only "damaged material", and the
  historical notification contains a specific subtype such as
  battery corrosion, extra spring, ratchet gear, sensor issue,
  electrical failure, soldering issue, lead screw damage etc.,
  the score must remain LOW.
- Do not invent the missing subtype in the current issue.
- If the specific problem meanings are equivalent, score high.
- If specific problem meanings are different, score low.

Return ONLY:

{{
    "score": integer
}}
"""

    start_time = time.perf_counter()

    try:

        response = client.chat.completions.create(
            model=OPENAI_MODEL,

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

    except Exception as error:

        raise RuntimeError(
            f"LLM API call failed: {error}"
        ) from error

    end_time = time.perf_counter()

    response_time = (
        end_time - start_time
    )

    # --------------------------------------------------------
    # Extract content
    # --------------------------------------------------------

    try:

        content = (
            response
            .choices[0]
            .message
            .content
        )

    except Exception as error:

        raise RuntimeError(
            "Unable to extract LLM response."
        ) from error

    if not content:

        raise RuntimeError(
            "LLM returned empty response."
        )

    content = content.strip()

    # --------------------------------------------------------
    # Remove markdown if accidentally returned
    # --------------------------------------------------------

    if content.startswith("```"):

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
        )

        content = content.strip()

    # --------------------------------------------------------
    # Parse JSON
    # --------------------------------------------------------

    try:

        result = json.loads(content)

    except json.JSONDecodeError as error:

        raise RuntimeError(
            "LLM returned invalid JSON.\n"
            f"Response: {content}"
        ) from error

    # --------------------------------------------------------
    # Validate score
    # --------------------------------------------------------

    if not isinstance(result, dict):

        raise RuntimeError(
            "LLM result must be a JSON object."
        )

    score = result.get("score")

    if score is None:

        raise RuntimeError(
            "LLM response does not contain score."
        )

    try:

        score = float(score)

    except (
        TypeError,
        ValueError
    ):

        raise RuntimeError(
            f"Invalid score returned by LLM: {score}"
        )

    if score < 0 or score > 100:

        raise RuntimeError(
            f"LLM score outside valid range: {score}"
        )

    score = int(round(score))

    return score, response_time


# ============================================================
# CALCULATE ALL SCORES
# ============================================================

def calculate_similarity(
    issue_summary,
    notifications
):
    """
    Calculate similarity for every notification.

    Blank/missing notifications receive 0 without
    calling the LLM.

    Valid notifications are evaluated independently.
    """

    results = []

    total_llm_time = 0.0

    valid_count = 0
    missing_count = 0

    for number, item in enumerate(
        notifications,
        start=1
    ):

        notification_id = (
            item["pcm_inv_notif_id"]
        )

        notification = (
            item["pcm_inv_notif"]
        )

        print("\n")
        print("-" * 70)

        print(
            f"Processing {number}/"
            f"{len(notifications)}"
        )

        print(
            f"ID: {notification_id}"
        )

        # ----------------------------------------------------
        # Missing / blank
        # ----------------------------------------------------

        if item["missing_notification"]:

            print(
                "COMMENT: Notification is blank/missing."
            )

            print(
                "Score: 0% match"
            )

            results.append({
                "pcm_inv_notif_id":
                    notification_id,

                "score":
                    0
            })

            missing_count += 1

            continue

        # ----------------------------------------------------
        # Valid notification
        # ----------------------------------------------------

        print(
            f"Notification: {notification}"
        )

        score, response_time = calculate_one_score(
            issue_summary=issue_summary,
            notification=notification
        )

        total_llm_time += response_time

        valid_count += 1

        results.append({
            "pcm_inv_notif_id":
                notification_id,

            "score":
                score
        })

        print(
            f"Score: {score}% match"
        )

        print(
            f"LLM time: {response_time:.3f} seconds"
        )

    print("\n")
    print("=" * 70)
    print("SIMILARITY PROCESSING SUMMARY")
    print("=" * 70)

    print(
        f"Total notifications : {len(notifications)}"
    )

    print(
        f"Valid notifications  : {valid_count}"
    )

    print(
        f"Blank/missing        : {missing_count}"
    )

    print(
        f"Total LLM time       : "
        f"{total_llm_time:.3f} seconds"
    )

    print("=" * 70)

    return {
        "results": results
    }, total_llm_time


# ============================================================
# ADD SCORES TO ORIGINAL JSON
# ============================================================

def add_scores(
    original_data,
    similarity_result
):

    score_map = {}

    for result in similarity_result.get(
        "results",
        []
    ):

        notification_id = result.get(
            "pcm_inv_notif_id"
        )

        score = result.get(
            "score"
        )

        if notification_id is None:
            continue

        if score is None:
            continue

        try:

            score = int(
                round(
                    float(score)
                )
            )

        except (
            TypeError,
            ValueError
        ):

            score = 0

        score = max(
            0,
            min(
                100,
                score
            )
        )

        score_map[
            str(notification_id)
        ] = f"{score}% match"

    # --------------------------------------------------------
    # Preserve original JSON
    # --------------------------------------------------------

    final_data = dict(
        original_data
    )

    updated_search_results = []

    for item in original_data.get(
        "search_results",
        []
    ):

        if not isinstance(item, dict):

            updated_search_results.append(
                item
            )

            continue

        updated_item = {}

        notification_id = item.get(
            "pcm_inv_notif_id"
        )

        if notification_id is not None:

            notification_id = str(
                notification_id
            ).strip()

        score_value = score_map.get(
            notification_id,
            "0% match"
        )

        # ----------------------------------------------------
        # Preserve original order
        # ----------------------------------------------------

        score_inserted = False

        for key, value in item.items():

            updated_item[key] = value

            if (
                key == "pcm_inv_notif"
                and not score_inserted
            ):

                updated_item["score"] = (
                    score_value
                )

                score_inserted = True

        # ----------------------------------------------------
        # If pcm_inv_notif itself is missing
        # ----------------------------------------------------

        if not score_inserted:

            updated_item["score"] = (
                score_value
            )

        updated_search_results.append(
            updated_item
        )

    final_data[
        "search_results"
    ] = updated_search_results

    return final_data


# ============================================================
# PROCESS DATA
# ============================================================

def process_data(
    incoming_data
):

    total_start_time = time.perf_counter()

    # --------------------------------------------------------
    # Validate
    # --------------------------------------------------------

    validate_input_data(
        incoming_data
    )

    issue_summary = normalize_text(
        incoming_data.get(
            "pcm_issue_summary"
        )
    )

    search_results = incoming_data.get(
        "search_results"
    )

    # --------------------------------------------------------
    # Find notifications
    # --------------------------------------------------------

    notifications = find_notifications(
        search_results
    )

    print("\n")
    print("=" * 70)
    print("INPUT ANALYSIS")
    print("=" * 70)

    print(
        f"Current issue summary : {issue_summary}"
    )

    print(
        f"Total search records  : "
        f"{len(search_results)}"
    )

    print(
        f"Notification records  : "
        f"{len(notifications)}"
    )

    missing_count = sum(
        1
        for item in notifications
        if item["missing_notification"]
    )

    print(
        f"Blank/missing notif   : "
        f"{missing_count}"
    )

    print("=" * 70)

    # --------------------------------------------------------
    # No search results
    # --------------------------------------------------------

    if not notifications:

        print(
            "COMMENT: No notification records found."
        )

        final_result = add_scores(
            original_data=incoming_data,

            similarity_result={
                "results": []
            }
        )

        total_end_time = time.perf_counter()

        return (
            final_result,
            0.0,
            total_end_time - total_start_time
        )

    # --------------------------------------------------------
    # Check whether at least one valid notification exists
    # --------------------------------------------------------

    valid_notifications = [
        item
        for item in notifications
        if not item["missing_notification"]
    ]

    # --------------------------------------------------------
    # ALL notifications missing/blank
    # --------------------------------------------------------

    if not valid_notifications:

        print("\n")
        print(
            "COMMENT: All notifications are blank/missing."
        )

        print(
            "COMMENT: LLM call skipped."
        )

        print(
            "COMMENT: All notification scores = 0."
        )

        similarity_result = {
            "results": [
                {
                    "pcm_inv_notif_id":
                        item["pcm_inv_notif_id"],

                    "score":
                        0
                }

                for item in notifications
                if item["pcm_inv_notif_id"] is not None
            ]
        }

        final_result = add_scores(
            original_data=incoming_data,
            similarity_result=similarity_result
        )

        total_end_time = time.perf_counter()

        return (
            final_result,
            0.0,
            total_end_time - total_start_time
        )

    # --------------------------------------------------------
    # LLM similarity
    # --------------------------------------------------------

    (
        similarity_result,
        llm_response_time
    ) = calculate_similarity(
        issue_summary=issue_summary,
        notifications=notifications
    )

    # --------------------------------------------------------
    # Add scores
    # --------------------------------------------------------

    final_result = add_scores(
        original_data=incoming_data,
        similarity_result=similarity_result
    )

    # --------------------------------------------------------
    # Total processing time
    # --------------------------------------------------------

    total_end_time = time.perf_counter()

    total_processing_time = (
        total_end_time
        - total_start_time
    )

    return (
        final_result,
        llm_response_time,
        total_processing_time
    )


# ============================================================
# LOAD JSON
# ============================================================

def load_json_file(
    file_path
):

    try:

        with open(
            file_path,
            "r",
            encoding="utf-8"
        ) as file:

            return json.load(file)

    except FileNotFoundError as error:

        raise RuntimeError(
            f"Input file not found: {file_path}"
        ) from error

    except json.JSONDecodeError as error:

        raise RuntimeError(
            f"Invalid JSON in file: {file_path}\n"
            f"Error: {error}"
        ) from error


# ============================================================
# SAVE JSON
# ============================================================

def save_json_file(
    data,
    file_path
):

    with open(
        file_path,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            data,
            file,
            indent=4,
            ensure_ascii=False
        )


# ============================================================
# PRINT SCORE SUMMARY
# ============================================================

def print_score_summary(
    final_result
):

    print("\n")
    print("=" * 70)
    print("SCORE SUMMARY")
    print("=" * 70)

    for item in final_result.get(
        "search_results",
        []
    ):

        notification_id = item.get(
            "pcm_inv_notif_id",
            "N/A"
        )

        notification = item.get(
            "pcm_inv_notif",
            ""
        )

        score = item.get(
            "score",
            "0% match"
        )

        print(
            f"\nID    : {notification_id}"
        )

        print(
            f"Text  : "
            f"{notification or '[BLANK / MISSING]'}"
        )

        print(
            f"Score : {score}"
        )

    print("=" * 70)


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("GENAI SEMANTIC SIMILARITY")
    print("=" * 70)

    print(
        f"Model : {OPENAI_MODEL}"
    )

    input_file = Path(
        "example_input.json"
    )

    output_file = Path(
        "output.json"
    )

    print(
        f"Input  : {input_file}"
    )

    print(
        f"Output : {output_file}"
    )

    program_start_time = time.perf_counter()

    try:

        # ----------------------------------------------------
        # Load
        # ----------------------------------------------------

        incoming_data = load_json_file(
            input_file
        )

        # ----------------------------------------------------
        # Process
        # ----------------------------------------------------

        (
            final_result,
            llm_response_time,
            total_processing_time
        ) = process_data(
            incoming_data
        )

        # ----------------------------------------------------
        # Save
        # ----------------------------------------------------

        save_json_file(
            final_result,
            output_file
        )

        # ----------------------------------------------------
        # Program timer
        # ----------------------------------------------------

        program_end_time = time.perf_counter()

        total_program_time = (
            program_end_time
            - program_start_time
        )

        # ----------------------------------------------------
        # Performance
        # ----------------------------------------------------

        print("\n")
        print("=" * 70)
        print("PERFORMANCE")
        print("=" * 70)

        print(
            f"Number of notifications : "
            f"{len(final_result.get('search_results', []))}"
        )

        print(
            f"LLM total response time : "
            f"{llm_response_time:.3f} seconds"
        )

        print(
            f"Total processing time   : "
            f"{total_processing_time:.3f} seconds"
        )

        print(
            f"Total program time      : "
            f"{total_program_time:.3f} seconds"
        )

        print("=" * 70)

        # ----------------------------------------------------
        # Scores
        # ----------------------------------------------------

        print_score_summary(
            final_result
        )

        # ----------------------------------------------------
        # Final JSON
        # ----------------------------------------------------

        print("\n")
        print("=" * 70)
        print("FINAL RESULT")
        print("=" * 70)

        print(
            json.dumps(
                final_result,
                indent=4,
                ensure_ascii=False
            )
        )

        print("\n")

        print(
            f"Output saved to: {output_file}"
        )

    except Exception as error:

        print("\n")
        print("=" * 70)
        print("ERROR")
        print("=" * 70)

        print(
            str(error)
        )

        print("=" * 70)

        raise


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()
