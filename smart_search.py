import json
import os
import re
import time
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI


# ============================================================
# LOAD ENVIRONMENT VARIABLES
# ============================================================

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-5")


# ============================================================
# VALIDATE CONFIGURATION
# ============================================================

if not OPENAI_API_KEY:
    raise RuntimeError(
        "OPENAI_API_KEY is not configured in .env file."
    )


# ============================================================
# CREATE OPENAI CLIENT
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
You are a precision semantic similarity evaluator.

Your ONLY task is to calculate similarity between:

1. ONE CURRENT ISSUE SUMMARY
2. MULTIPLE HISTORICAL NOTIFICATIONS

This is NOT a medical diagnosis task.

Do not:
- diagnose anything
- recommend treatment
- recommend medication
- provide medical advice
- invent information
- modify input text

============================================================
MOST IMPORTANT RULE: GENERIC VS SPECIFIC CONTENT
============================================================

Many records may contain a common generic phrase.

Example:

Current:
"damaged material (soldering issue)"

Historical:
"damaged material (soldering issues)"

Historical:
"damaged material (damaged lead screw)"

Historical:
"damaged material (battery corrosion)"

Here:

"damaged material"

is GENERIC / COMMON CONTEXT.

The text inside parentheses is the SPECIFIC CONTENT.

The specific content is much more important for similarity.

Therefore:

DO NOT give a high similarity score simply because both records
contain the same generic phrase.

Repeated generic/common words must NOT dominate the score.

============================================================
SPECIFIC CONTENT HAS HIGHER WEIGHT
============================================================

When specific information is available, use approximately:

- Specific content: 80% importance
- Generic/common context: 20% importance

The exact weighting can be adjusted based on meaning.

Example:

Current:
"damaged material (soldering issues)"

Historical:
"damaged material (soldering problem)"

Expected:
Very high similarity.

Example:

Current:
"damaged material (soldering issues)"

Historical:
"damaged material (damaged lead screw)"

Expected:
Low similarity.

Example:

Current:
"damaged material (electrical failure)"

Historical:
"damaged material (electrical component failure)"

Expected:
High similarity.

Example:

Current:
"damaged material (battery corrosion)"

Historical:
"damaged material (soldering issues)"

Expected:
Low similarity.

============================================================
GENERIC-ONLY CURRENT ISSUE
============================================================

If the current issue is only:

"damaged material"

and historical records are:

"damaged material (soldering issues)"
"damaged material (damaged lead screw)"
"damaged material (battery corrosion)"

then DO NOT give all records 85-95%.

The current issue does not contain enough specific information
to establish strong similarity with the specific historical
sub-types.

The common phrase "damaged material" alone should produce only
a LIMITED similarity contribution.

In such a situation, prefer a lower/moderate score rather than
artificially giving every record a high score.

============================================================
SEMANTIC SIMILARITY
============================================================

Do not rely only on exact keyword matching.

Understand:
- synonyms
- equivalent phrases
- paraphrases
- abbreviations
- technical terminology
- word order differences
- singular/plural differences
- grammatical differences

Example:

"high SMA wire resistance"

and

"SMA wire has high resistance"

should be highly similar.

Example:

"soldering issue"

and

"problem during soldering"

should be highly similar.

But:

"soldering issue"

and

"damaged lead screw"

should be low similarity.

============================================================
SCORING
============================================================

95-100:
Almost identical specific meaning.

85-94:
Very high similarity in specific content.

70-84:
Strong similarity in specific content.

50-69:
Moderate similarity.

30-49:
Weak similarity.

10-29:
Very low similarity.

0-9:
Essentially unrelated.

IMPORTANT:

If only the generic/common phrase matches but the specific
content is different, DO NOT give a high score.

A shared generic category alone should generally remain below
50 unless the overall meaning independently supports a higher
score.

============================================================
IMPORTANT OUTPUT RULES
============================================================

Return EXACTLY one result for every historical notification.

The pcm_inv_notif_id MUST exactly match the input ID.

The score MUST be an integer between 0 and 100.

Do not include "%" in the score.

Return ONLY valid JSON.

Required format:

{
    "results": [
        {
            "pcm_inv_notif_id": "123",
            "score": 92
        }
    ]
}

Do not add explanations.
Do not add markdown.
Do not add comments.
"""


# ============================================================
# TEXT NORMALIZATION
# ============================================================

def normalize_text(text):
    """
    Basic text normalization.

    This does NOT change the original JSON.
    It is only used internally for analysis.
    """

    if text is None:
        return ""

    text = str(text).strip()

    # Normalize spaces
    text = re.sub(r"\s+", " ", text)

    return text


# ============================================================
# EXTRACT GENERIC AND SPECIFIC CONTENT
# ============================================================

def analyze_text_structure(text):
    """
    Detects text inside parentheses.

    Example:

    damaged material (soldering issues)

    becomes:

    {
        "full_text": "...",
        "generic_part": "damaged material",
        "specific_part": "soldering issues"
    }

    If parentheses do not exist, the whole text is treated
    as the main/general text.
    """

    text = normalize_text(text)

    if not text:
        return {
            "full_text": "",
            "generic_part": "",
            "specific_part": ""
        }

    # Find content inside parentheses
    match = re.search(r"\((.*?)\)", text)

    if match:
        specific_part = match.group(1).strip()

        # Remove the parentheses portion
        generic_part = re.sub(
            r"\s*\(.*?\)",
            "",
            text
        ).strip()

        return {
            "full_text": text,
            "generic_part": generic_part,
            "specific_part": specific_part
        }

    # No parentheses.
    return {
        "full_text": text,
        "generic_part": text,
        "specific_part": ""
    }


# ============================================================
# FIND CURRENT ISSUE SUMMARY
# ============================================================

def find_issue_summary(data):
    """
    Finds pcm_issue_summary.

    Supports the current JSON structure:

    {
        "search_results": [...],
        "pcm_issue_summary": "...",
        ...
    }

    Also supports a list of dictionaries for backward compatibility.
    """

    # Normal expected structure
    if isinstance(data, dict):

        summary = data.get("pcm_issue_summary")

        if summary is not None:

            summary = normalize_text(summary)

            if summary:
                return summary

    # Backward compatibility
    if isinstance(data, list):

        for item in data:

            if not isinstance(item, dict):
                continue

            if "pcm_issue_summary" not in item:
                continue

            summary = normalize_text(
                item.get("pcm_issue_summary")
            )

            if summary:
                return summary

    raise ValueError(
        "No valid 'pcm_issue_summary' found in input JSON."
    )


# ============================================================
# FIND ALL PCM INV NOTIFICATIONS
# ============================================================

def find_notifications(search_results):
    """
    Finds every valid notification object.

    Only these fields are required:

        pcm_inv_notif_id
        pcm_inv_notif

    All other fields are preserved in original JSON but ignored
    for similarity calculation.
    """

    if not isinstance(search_results, list):
        raise ValueError(
            "'search_results' must be a list."
        )

    notifications = []

    for index, item in enumerate(search_results):

        if not isinstance(item, dict):
            print(
                f"WARNING: search_results[{index}] is not an object. "
                f"Skipping."
            )
            continue

        notification = item.get("pcm_inv_notif")

        if notification is None:
            print(
                f"WARNING: search_results[{index}] does not contain "
                f"'pcm_inv_notif'. Skipping."
            )
            continue

        notification = normalize_text(notification)

        if not notification:
            print(
                f"WARNING: search_results[{index}] has empty "
                f"'pcm_inv_notif'. Skipping."
            )
            continue

        notification_id = item.get("pcm_inv_notif_id")

        if notification_id is None:
            print(
                f"WARNING: search_results[{index}] does not contain "
                f"'pcm_inv_notif_id'. Skipping."
            )
            continue

        notification_id = str(notification_id).strip()

        if not notification_id:
            print(
                f"WARNING: search_results[{index}] has empty "
                f"'pcm_inv_notif_id'. Skipping."
            )
            continue

        structure = analyze_text_structure(
            notification
        )

        notifications.append(
            {
                "pcm_inv_notif_id": notification_id,
                "pcm_inv_notif": notification,
                "generic_part": structure["generic_part"],
                "specific_part": structure["specific_part"]
            }
        )

    return notifications


# ============================================================
# FIND COMMON GENERIC CONTEXT
# ============================================================

def find_common_generic_context(issue_summary, notifications):
    """
    Finds the generic context shared by the current issue and
    historical notifications.

    This is mainly for helping the LLM understand that a repeated
    phrase should not dominate similarity.

    Example:

        damaged material
        damaged material (soldering issues)
        damaged material (lead screw)

    Common context:

        damaged material
    """

    issue_structure = analyze_text_structure(
        issue_summary
    )

    issue_generic = issue_structure["generic_part"].lower().strip()

    if not issue_generic:
        return ""

    common_count = 0

    for notification in notifications:

        generic_part = (
            notification["generic_part"]
            .lower()
            .strip()
        )

        if generic_part == issue_generic:
            common_count += 1

    # Consider it common if it appears in at least one historical
    # notification. The prompt will tell the LLM to down-weight it.
    if common_count > 0:
        return issue_structure["generic_part"]

    return ""


# ============================================================
# BUILD ANALYSIS DATA
# ============================================================

def build_analysis_data(issue_summary, notifications):
    """
    Creates structured information for the LLM.

    Original input is NOT modified.
    """

    issue_structure = analyze_text_structure(
        issue_summary
    )

    common_generic_context = find_common_generic_context(
        issue_summary,
        notifications
    )

    historical_cases = []

    for notification in notifications:

        historical_cases.append(
            {
                "pcm_inv_notif_id": notification[
                    "pcm_inv_notif_id"
                ],
                "pcm_inv_notif": notification[
                    "pcm_inv_notif"
                ],
                "generic_part": notification[
                    "generic_part"
                ],
                "specific_part": notification[
                    "specific_part"
                ]
            }
        )

    analysis_data = {
        "current_issue": {
            "full_text": issue_summary,
            "generic_part": issue_structure[
                "generic_part"
            ],
            "specific_part": issue_structure[
                "specific_part"
            ]
        },
        "common_generic_context": common_generic_context,
        "historical_notifications": historical_cases
    }

    return analysis_data


# ============================================================
# VALIDATE LLM RESULT
# ============================================================

def validate_similarity_result(
    result,
    notifications
):
    """
    Strictly validates the LLM response.

    Checks:

    1. results exists
    2. results is a list
    3. every input ID exists exactly once
    4. score is numeric
    5. score is between 0 and 100
    """

    if not isinstance(result, dict):
        raise RuntimeError(
            "GenAI response must be a JSON object."
        )

    if "results" not in result:
        raise RuntimeError(
            "GenAI response does not contain 'results'."
        )

    results = result["results"]

    if not isinstance(results, list):
        raise RuntimeError(
            "'results' must be a list."
        )

    expected_ids = [
        str(item["pcm_inv_notif_id"])
        for item in notifications
    ]

    received_ids = []

    for item in results:

        if not isinstance(item, dict):
            raise RuntimeError(
                "Each result must be a JSON object."
            )

        notification_id = item.get(
            "pcm_inv_notif_id"
        )

        if notification_id is None:
            raise RuntimeError(
                "A result is missing 'pcm_inv_notif_id'."
            )

        notification_id = str(
            notification_id
        )

        score = item.get("score")

        if score is None:
            raise RuntimeError(
                f"Score missing for notification "
                f"{notification_id}."
            )

        try:
            score = float(score)
        except (TypeError, ValueError):

            raise RuntimeError(
                f"Invalid score for notification "
                f"{notification_id}: {score}"
            )

        if score < 0 or score > 100:

            raise RuntimeError(
                f"Score must be between 0 and 100. "
                f"Received {score} for {notification_id}."
            )

        received_ids.append(
            notification_id
        )

    # Check duplicate IDs
    if len(received_ids) != len(set(received_ids)):

        raise RuntimeError(
            "GenAI returned duplicate notification IDs."
        )

    # Check missing IDs
    missing_ids = set(expected_ids) - set(received_ids)

    if missing_ids:

        raise RuntimeError(
            "GenAI did not return results for all "
            f"notifications. Missing IDs: {missing_ids}"
        )

    # Check unexpected IDs
    unexpected_ids = set(received_ids) - set(expected_ids)

    if unexpected_ids:

        raise RuntimeError(
            "GenAI returned unexpected notification IDs: "
            f"{unexpected_ids}"
        )

    # Normalize scores
    normalized_results = []

    for item in results:

        notification_id = str(
            item["pcm_inv_notif_id"]
        )

        score = float(
            item["score"]
        )

        score = max(
            0,
            min(
                100,
                score
            )
        )

        normalized_results.append(
            {
                "pcm_inv_notif_id": notification_id,
                "score": int(round(score))
            }
        )

    return {
        "results": normalized_results
    }


# ============================================================
# CALCULATE SIMILARITY USING GENAI
# ============================================================

def calculate_similarity(
    issue_summary,
    notifications
):
    """
    Calculates semantic similarity between the current issue
    and every historical notification.

    Uses ONE LLM call for the complete batch.

    Returns:

        similarity_result
        llm_response_time
    """

    analysis_data = build_analysis_data(
        issue_summary=issue_summary,
        notifications=notifications
    )

    # --------------------------------------------------------
    # Create user prompt
    # --------------------------------------------------------

    user_prompt = f"""
CURRENT ISSUE:

{json.dumps(
    analysis_data["current_issue"],
    ensure_ascii=False,
    indent=2
)}


COMMON GENERIC CONTEXT:

{json.dumps(
    analysis_data["common_generic_context"],
    ensure_ascii=False
)}


HISTORICAL NOTIFICATIONS:

{json.dumps(
    analysis_data["historical_notifications"],
    ensure_ascii=False,
    indent=2
)}


TASK:

Compare the CURRENT ISSUE with EVERY historical notification.

IMPORTANT:

The current issue and historical notifications may share a
generic phrase.

Do NOT give a high score simply because the generic phrase is
the same.

Pay much more attention to the specific content.

If the text inside parentheses is present, it usually represents
the important specific issue.

For example:

"damaged material (soldering issues)"

should be evaluated primarily using:

"soldering issues"

rather than simply:

"damaged material"

If the current issue has no specific content and contains only
the generic phrase, do not artificially give all historical
records a high score.

Return exactly one result for every historical notification.

Return ONLY valid JSON in this format:

{{
    "results": [
        {{
            "pcm_inv_notif_id": "123",
            "score": 92
        }}
    ]
}}
"""

    # ========================================================
    # START LLM TIMER
    # ========================================================

    llm_start_time = time.perf_counter()

    # ========================================================
    # OPENAI / GENAI CALL
    # ========================================================

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

    # ========================================================
    # END LLM TIMER
    # ========================================================

    llm_end_time = time.perf_counter()

    llm_response_time = (
        llm_end_time - llm_start_time
    )

    # ========================================================
    # EXTRACT RESPONSE
    # ========================================================

    try:

        raw_result = (
            response
            .choices[0]
            .message
            .content
            .strip()
        )

    except Exception as error:

        raise RuntimeError(
            "Unable to extract content from LLM response."
        ) from error

    if not raw_result:

        raise RuntimeError(
            "LLM returned an empty response."
        )

    # ========================================================
    # REMOVE MARKDOWN FENCES IF ANY
    # ========================================================

    if raw_result.startswith("```"):

        raw_result = re.sub(
            r"^```(?:json)?",
            "",
            raw_result,
            flags=re.IGNORECASE
        )

        raw_result = re.sub(
            r"```$",
            "",
            raw_result
        )

        raw_result = raw_result.strip()

    # ========================================================
    # PARSE JSON
    # ========================================================

    try:

        result = json.loads(
            raw_result
        )

    except json.JSONDecodeError as error:

        raise RuntimeError(
            "GenAI returned invalid JSON.\n\n"
            f"Model response:\n{raw_result}"
        ) from error

    # ========================================================
    # VALIDATE RESULT
    # ========================================================

    result = validate_similarity_result(
        result=result,
        notifications=notifications
    )

    return result, llm_response_time


# ============================================================
# ADD SCORES TO ORIGINAL JSON
# ============================================================

def add_scores(
    original_data,
    similarity_result
):
    """
    Adds scores ONLY to search_results.

    All other original JSON fields remain unchanged.
    """

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

            score = float(score)

        except (
            TypeError,
            ValueError
        ):

            continue

        score = max(
            0,
            min(
                100,
                score
            )
        )

        score = int(
            round(score)
        )

        score_map[
            str(notification_id)
        ] = f"{score}% match"

    # --------------------------------------------------------
    # Copy original JSON
    # --------------------------------------------------------

    final_data = dict(
        original_data
    )

    # --------------------------------------------------------
    # Update search_results
    # --------------------------------------------------------

    update_search_results = []

    for item in original_data.get(
        "search_results",
        []
    ):

        # Preserve complete original object
        updated_item = dict(item)

        notification_id = updated_item.get(
            "pcm_inv_notif_id"
        )

        if notification_id is not None:

            notification_id = str(
                notification_id
            )

            if notification_id in score_map:

                updated_item["score"] = (
                    score_map[notification_id]
                )

        update_search_results.append(
            updated_item
        )

    final_data["search_results"] = (
        update_search_results
    )

    return final_data


# ============================================================
# VALIDATE INPUT JSON
# ============================================================

def validate_input_data(incoming_data):
    """
    Validates the expected JSON structure.
    """

    if not isinstance(
        incoming_data,
        dict
    ):

        raise ValueError(
            "Input JSON must be a JSON object."
        )

    search_results = incoming_data.get(
        "search_results"
    )

    if not isinstance(
        search_results,
        list
    ):

        raise ValueError(
            "'search_results' must be a list."
        )

    issue_summary = incoming_data.get(
        "pcm_issue_summary"
    )

    if issue_summary is None:

        raise ValueError(
            "No 'pcm_issue_summary' found in input JSON."
        )

    issue_summary = normalize_text(
        issue_summary
    )

    if not issue_summary:

        raise ValueError(
            "'pcm_issue_summary' cannot be empty."
        )

    return True


# ============================================================
# MAIN BUSINESS LOGIC
# ============================================================

def process_data(incoming_data):
    """
    Main reusable business function.

    Input:

    {
        "search_results": [...],
        "pcm_issue_summary": "...",
        "pcm_issue_id": "...",
        "automation_id": "..."
    }

    Returns:

        final_result
        llm_response_time
        total_processing_time
    """

    # ========================================================
    # START TOTAL TIMER
    # ========================================================

    total_start_time = time.perf_counter()

    # ========================================================
    # VALIDATE INPUT
    # ========================================================

    validate_input_data(
        incoming_data
    )

    # ========================================================
    # GET SEARCH RESULTS
    # ========================================================

    search_results = incoming_data.get(
        "search_results"
    )

    # ========================================================
    # GET CURRENT ISSUE
    # ========================================================

    issue_summary = incoming_data.get(
        "pcm_issue_summary"
    )

    issue_summary = normalize_text(
        issue_summary
    )

    # ========================================================
    # FIND HISTORICAL NOTIFICATIONS
    # ========================================================

    notifications = find_notifications(
        search_results
    )

    if not notifications:

        raise ValueError(
            "No valid 'pcm_inv_notif' records found "
            "inside 'search_results'."
        )

    # ========================================================
    # PRINT ANALYSIS INFORMATION
    # ========================================================

    print("\n")
    print("=" * 70)
    print("INPUT ANALYSIS")
    print("=" * 70)

    print(
        f"Current issue summary : {issue_summary}"
    )

    current_structure = analyze_text_structure(
        issue_summary
    )

    print(
        f"Generic part          : "
        f"{current_structure['generic_part']}"
    )

    print(
        f"Specific part         : "
        f"{current_structure['specific_part'] or '[NONE]'}"
    )

    common_context = find_common_generic_context(
        issue_summary,
        notifications
    )

    print(
        f"Common generic part   : "
        f"{common_context or '[NONE]'}"
    )

    print(
        f"Notifications         : "
        f"{len(notifications)}"
    )

    print("=" * 70)

    # ========================================================
    # GENAI SIMILARITY
    # ========================================================

    similarity_result, llm_response_time = (
        calculate_similarity(
            issue_summary=issue_summary,
            notifications=notifications
        )
    )

    # ========================================================
    # ADD SCORES
    # ========================================================

    final_result = add_scores(
        original_data=incoming_data,
        similarity_result=similarity_result
    )

    # ========================================================
    # END TOTAL TIMER
    # ========================================================

    total_end_time = time.perf_counter()

    total_processing_time = (
        total_end_time -
        total_start_time
    )

    return (
        final_result,
        llm_response_time,
        total_processing_time
    )


# ============================================================
# LOAD JSON FILE
# ============================================================

def load_json_file(file_path):

    try:

        with open(
            file_path,
            "r",
            encoding="utf-8"
        ) as file:

            return json.load(file)

    except json.JSONDecodeError as error:

        raise RuntimeError(
            f"Invalid JSON in file: {file_path}\n"
            f"Error: {error}"
        ) from error

    except FileNotFoundError as error:

        raise RuntimeError(
            f"Input file not found: {file_path}"
        ) from error


# ============================================================
# SAVE JSON FILE
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

def print_score_summary(final_result):

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
            "N/A"
        )

        score = item.get(
            "score",
            "N/A"
        )

        print(
            f"\nID      : {notification_id}"
        )

        print(
            f"Text    : {notification}"
        )

        print(
            f"Score   : {score}"
        )

    print("=" * 70)


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("GENAI SPECIFIC SEMANTIC SIMILARITY")
    print("=" * 70)

    print(
        f"Model       : {OPENAI_MODEL}"
    )

    # --------------------------------------------------------
    # Input / Output files
    # --------------------------------------------------------

    input_file = Path(
        "example_input.json"
    )

    output_file = Path(
        "output.json"
    )

    print(
        f"\nInput  : {input_file}"
    )

    print(
        f"Output : {output_file}"
    )

    # ========================================================
    # TOTAL PROGRAM TIMER
    # ========================================================

    program_start_time = time.perf_counter()

    try:

        # ----------------------------------------------------
        # LOAD INPUT
        # ----------------------------------------------------

        incoming_data = load_json_file(
            input_file
        )

        # ----------------------------------------------------
        # PROCESS
        # ----------------------------------------------------

        (
            final_result,
            llm_response_time,
            total_processing_time
        ) = process_data(
            incoming_data
        )

        # ----------------------------------------------------
        # SAVE OUTPUT
        # ----------------------------------------------------

        save_json_file(
            final_result,
            output_file
        )

        # ====================================================
        # END PROGRAM TIMER
        # ====================================================

        program_end_time = time.perf_counter()

        total_program_time = (
            program_end_time -
            program_start_time
        )

        # ====================================================
        # PERFORMANCE REPORT
        # ====================================================

        print("\n")
        print("=" * 70)
        print("PERFORMANCE")
        print("=" * 70)

        print(
            f"Number of notifications : "
            f"{len(final_result.get('search_results', []))}"
        )

        print(
            f"LLM response time       : "
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

        # ====================================================
        # SCORE SUMMARY
        # ====================================================

        print_score_summary(
            final_result
        )

        # ====================================================
        # FINAL RESULT
        # ====================================================

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
