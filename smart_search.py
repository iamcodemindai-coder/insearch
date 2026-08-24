import json
import os
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

OPENAI_TEMPERATURE = float(
    os.getenv("OPENAI_TEMPERATURE", "0")
)


# ============================================================
# VALIDATE CONFIGURATION
# ============================================================

if not OPENAI_API_KEY:
    raise RuntimeError(
        "OPENAI_API_KEY is not configured."
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
You are a semantic similarity evaluator.

Your ONLY task is to compare ONE current issue summary
with multiple historical patient notifications.

This is NOT a medical diagnosis task.

Do NOT:
- diagnose the patient
- recommend treatment
- recommend medication
- provide clinical advice
- make medical decisions
- invent information
- modify the input text

Compare only the semantic meaning of the information provided.

Consider:
- symptoms
- events
- conditions
- relevant context
- overall meaning

Do NOT rely only on exact keyword matching.

SCORING:

95-100 = Almost identical meaning
85-94  = Very high similarity
70-84  = Strong similarity
50-69  = Moderate similarity
30-49  = Weak similarity
0-29   = Very low or no meaningful similarity

Example:

Current issue:
"Patient is experiencing fever and body pain."

Historical notification:
"Patient previously reported high temperature and body ache."

This should receive a very high similarity score.

Another example:

Current issue:
"Patient is experiencing fever and body pain."

Historical notification:
"Patient visited hospital for a minor leg injury."

This should receive a low similarity score.

IMPORTANT:

Return EXACTLY one result for every historical
pcm_inv_notif.

The pcm_inv_notif_id in the output MUST exactly match
the corresponding input pcm_inv_notif_id.

The score MUST be an integer from 0 to 100.

The score represents percentage similarity.

For example:
92 means 92% similarity.
75 means 75% similarity.
20 means 20% similarity.

DO NOT include the "%" symbol in the score.

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
"""


# ============================================================
# FIND CURRENT ISSUE SUMMARY
# ============================================================

def find_issue_summary(data):
    """
    Finds pcm_issue_summary from the input JSON.

    No dependency on pcm_issue_id or any other field.
    """

    for item in data:

        if not isinstance(item, dict):
            continue

        if "pcm_issue_summary" not in item:
            continue

        summary = item.get(
            "pcm_issue_summary"
        )

        if summary is None:
            continue

        summary = str(summary).strip()

        if summary:
            return summary

    raise ValueError(
        "No valid 'pcm_issue_summary' found in input JSON."
    )


# ============================================================
# FIND ALL PCM INV NOTIFICATIONS
# ============================================================

def find_notifications(data):
    """
    Finds every object containing pcm_inv_notif.

    Only these fields are used for similarity:

        pcm_inv_notif_id
        pcm_inv_notif

    All other fields are ignored.
    """

    notifications = []

    for item in data:

        if not isinstance(item, dict):
            continue

        if "pcm_inv_notif" not in item:
            continue

        notification = item.get(
            "pcm_inv_notif"
        )

        if notification is None:
            continue

        notification = str(
            notification
        ).strip()

        if not notification:
            continue

        notifications.append(
            {
                "pcm_inv_notif_id": item.get(
                    "pcm_inv_notif_id"
                ),
                "pcm_inv_notif": notification
            }
        )

    return notifications


# ============================================================
# CALCULATE SIMILARITY USING GENAI
# ============================================================

def calculate_similarity(
    issue_summary,
    notifications
):
    """
    Sends current issue summary and all notifications
    to GenAI.

    Also measures the exact time taken by the LLM API call.

    Returns:
        similarity_result
        llm_response_time
    """

    historical_cases = []

    for notification in notifications:

        historical_cases.append(
            {
                "pcm_inv_notif_id": notification[
                    "pcm_inv_notif_id"
                ],
                "pcm_inv_notif": notification[
                    "pcm_inv_notif"
                ]
            }
        )

    # --------------------------------------------------------
    # Create user prompt
    # --------------------------------------------------------

    user_prompt = f"""
CURRENT ISSUE SUMMARY:

{issue_summary}


HISTORICAL NOTIFICATIONS:

{json.dumps(
    historical_cases,
    ensure_ascii=False,
    indent=2
)}


TASK:

Compare the CURRENT ISSUE SUMMARY with EVERY
pcm_inv_notif.

Return exactly one score for every notification.

The score must be between 0 and 100.

Return ONLY valid JSON.
"""

    # ========================================================
    # START LLM TIMER
    # ========================================================

    llm_start_time = time.perf_counter()

    # ========================================================
    # OPENAI / GENAI CALL
    # ========================================================

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
        temperature=OPENAI_TEMPERATURE
    )

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

    raw_result = (
        response.choices[0]
        .message
        .content
        .strip()
    )

    # --------------------------------------------------------
    # Remove markdown fences if returned by model
    # --------------------------------------------------------

    if raw_result.startswith("```"):

        raw_result = raw_result.replace(
            "```json",
            ""
        )

        raw_result = raw_result.replace(
            "```",
            ""
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
    # VALIDATE RESPONSE
    # ========================================================

    if not isinstance(
        result,
        dict
    ):

        raise RuntimeError(
            "GenAI response must be a JSON object."
        )

    if "results" not in result:

        raise RuntimeError(
            "GenAI response does not contain 'results'."
        )

    if not isinstance(
        result["results"],
        list
    ):

        raise RuntimeError(
            "'results' must be a list."
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
    Adds ONLY:

        "score": "92% match"

    to objects containing pcm_inv_notif.

    Every other existing field and value
    remains unchanged.
    """

    score_map = {}

    # --------------------------------------------------------
    # Create score mapping
    # --------------------------------------------------------

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

        # ----------------------------------------------------
        # Keep score between 0 and 100
        # ----------------------------------------------------

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

        # ----------------------------------------------------
        # Convert:
        #
        # 92
        #
        # to:
        #
        # "92% match"
        # ----------------------------------------------------

        formatted_score = (
            f"{score}% match"
        )

        score_map[
            str(notification_id)
        ] = formatted_score

    # --------------------------------------------------------
    # Create final JSON
    # --------------------------------------------------------

    final_data = []

    for item in original_data:

        # Preserve complete original object
        updated_item = dict(item)

        # Only notification objects are touched
        if "pcm_inv_notif" in updated_item:

            notification_id = updated_item.get(
                "pcm_inv_notif_id"
            )

            if notification_id is not None:

                notification_id = str(
                    notification_id
                )

                if notification_id in score_map:

                    updated_item["score"] = (
                        score_map[
                            notification_id
                        ]
                    )

        final_data.append(
            updated_item
        )

    return final_data


# ============================================================
# MAIN BUSINESS LOGIC
# ============================================================

def process_data(incoming_data):
    """
    Main reusable business function.

    Returns:

        final_result
        llm_response_time
        total_processing_time
    """

    # ========================================================
    # START TOTAL TIMER
    # ========================================================

    total_start_time = time.perf_counter()

    # --------------------------------------------------------
    # Validate input
    # --------------------------------------------------------

    if not isinstance(
        incoming_data,
        list
    ):

        raise ValueError(
            "Input JSON must be a list of objects."
        )

    # --------------------------------------------------------
    # Find current issue
    # --------------------------------------------------------

    issue_summary = find_issue_summary(
        incoming_data
    )

    # --------------------------------------------------------
    # Find historical notifications
    # --------------------------------------------------------

    notifications = find_notifications(
        incoming_data
    )

    if not notifications:

        raise ValueError(
            "No 'pcm_inv_notif' records found."
        )

    # --------------------------------------------------------
    # GenAI similarity
    # --------------------------------------------------------

    similarity_result, llm_response_time = (
        calculate_similarity(
            issue_summary=issue_summary,
            notifications=notifications
        )
    )

    # --------------------------------------------------------
    # Add scores to original JSON
    # --------------------------------------------------------

    final_result = add_scores(
        original_data=incoming_data,
        similarity_result=similarity_result
    )

    # ========================================================
    # END TOTAL TIMER
    # ========================================================

    total_end_time = time.perf_counter()

    total_processing_time = (
        total_end_time - total_start_time
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

    with open(
        file_path,
        "r",
        encoding="utf-8"
    ) as file:

        return json.load(file)


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
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("GENAI SEMANTIC SIMILARITY")
    print("=" * 70)

    print(
        f"Model       : {OPENAI_MODEL}"
    )

    print(
        f"Temperature : {OPENAI_TEMPERATURE}"
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
    # TOTAL EXECUTION TIMER
    # ========================================================

    program_start_time = time.perf_counter()

    # --------------------------------------------------------
    # Load input JSON
    # --------------------------------------------------------

    incoming_data = load_json_file(
        input_file
    )

    # --------------------------------------------------------
    # Process
    # --------------------------------------------------------

    (
        final_result,
        llm_response_time,
        total_processing_time
    ) = process_data(
        incoming_data
    )

    # --------------------------------------------------------
    # Save output
    # --------------------------------------------------------

    save_json_file(
        final_result,
        output_file
    )

    # ========================================================
    # END PROGRAM TIMER
    # ========================================================

    program_end_time = time.perf_counter()

    total_program_time = (
        program_end_time - program_start_time
    )

    # ========================================================
    # PERFORMANCE REPORT
    # ========================================================

    print("\n")
    print("=" * 70)
    print("PERFORMANCE")
    print("=" * 70)

    print(
        f"Number of notifications : "
        f"{len(find_notifications(incoming_data))}"
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

    # ========================================================
    # FINAL RESULT
    # ========================================================

    print("\nFINAL RESULT")
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


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()