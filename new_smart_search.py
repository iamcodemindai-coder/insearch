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

Your ONLY task is to calculate semantic similarity between:

1. ONE CURRENT ISSUE SUMMARY
2. MULTIPLE HISTORICAL NOTIFICATIONS

This is NOT a medical diagnosis task.

Do not:
- diagnose anything
- recommend treatment
- recommend medication
- provide medical advice
- invent information
- modify the input text

============================================================
CORE OBJECTIVE
============================================================

For EVERY historical notification, independently compare it
with the CURRENT ISSUE SUMMARY.

Return exactly ONE similarity score for EVERY notification.

The score represents how strongly the COMPLETE MEANING of the
historical notification matches the COMPLETE MEANING of the
current issue.

Do NOT rank records only by shared words.

Understand the context and meaning of each record.

============================================================
MOST IMPORTANT RULE:
GENERIC CONTENT VS SPECIFIC CONTENT
============================================================

Historical notifications may contain a common generic category.

Example:

Current issue:
"damage material (soldering issue)"

Historical notification:
"damage material (soldering problem)"

Historical notification:
"damage material (battery corrosion)"

Historical notification:
"damage material (damaged lead screw)"

The phrase:

"damage material"

is generic/common context.

The following information is specific:

"soldering issue"
"battery corrosion"
"damaged lead screw"

When specific information exists, specific information is much
more important than generic/common information.

Therefore:

DO NOT give a high similarity score simply because the same
generic category appears in both records.

============================================================
SPECIFIC INFORMATION PRIORITY
============================================================

When specific information is present, evaluate primarily based
on the specific technical/problem-related meaning.

Use approximately:

Specific/distinctive meaning: 80% importance
Generic/common context:       20% importance

This is guidance, not a rigid mathematical formula.

Semantic meaning is more important than exact word matching.

============================================================
CRITICAL EXAMPLE
============================================================

Current:

"damage material (soldering issue)"

Historical 1:

"damage material (soldering problem)"

Historical 2:

"damage material (battery corrosion)"

Historical 3:

"damage material (damaged lead screw)"

Expected reasoning:

Historical 1:
The generic category is the same AND the specific issue is
semantically equivalent.

Therefore: HIGH similarity.

Historical 2:
The generic category is the same, but the specific problem is
different.

Therefore: LOW similarity.

Historical 3:
The generic category is the same, but the specific problem is
different.

Therefore: LOW similarity.

The repeated phrase "damage material" MUST NOT make all three
records highly similar.

============================================================
GENERIC-ONLY CURRENT ISSUE
============================================================

This is extremely important.

If the current issue is only:

"damage material"

and historical notifications are:

"damage material (soldering issue)"
"damage material (battery corrosion)"
"damage material (damaged lead screw)"

then the current issue does NOT provide enough specific
information to strongly match any particular subtype.

DO NOT give all historical records 85-100.

A generic category match alone should generally produce a
LOW or MODERATE score.

Do not invent missing details in the current issue.

============================================================
GENERIC-ONLY HISTORICAL NOTIFICATION
============================================================

If the current issue contains specific information but a
historical notification contains only a generic category,
do not assume that the historical notification has the same
specific problem.

Example:

Current:
"damage material (soldering issue)"

Historical:
"damage material"

The historical notification does not contain enough specific
information to establish a very strong match.

Therefore, do not give an artificially high score.

============================================================
WHEN BOTH ARE SPECIFIC
============================================================

If both current issue and historical notification contain
specific information, compare the specific meanings carefully.

Consider:

- technical meaning
- problem type
- affected component
- failure mode
- cause
- symptom
- condition
- terminology
- context

Example:

Current:
"high SMA wire resistance"

Historical:
"SMA wire has high resistance"

=> HIGH similarity.

Example:

Current:
"soldering issue"

Historical:
"problem during soldering"

=> HIGH similarity.

Example:

Current:
"soldering issue"

Historical:
"damaged lead screw"

=> LOW similarity.

============================================================
SEMANTIC UNDERSTANDING
============================================================

Do NOT rely only on exact keyword matching.

Understand:

- synonyms
- paraphrases
- equivalent phrases
- abbreviations
- technical terminology
- singular/plural differences
- grammatical differences
- word-order differences
- equivalent descriptions

Example:

"high resistance in SMA wire"

and

"SMA wire resistance is high"

should be highly similar.

Example:

"connector damaged"

and

"damage observed on connector"

may be highly similar if the complete meaning matches.

However, sharing one or two generic words is NOT enough for
a high score.

============================================================
DO NOT USE POSITION AS SIMILARITY
============================================================

Do not assume that the first notification is more relevant.

Do not assume that the last notification is less relevant.

Every notification must be evaluated independently based only
on its content and the current issue.

============================================================
BLANK OR MISSING NOTIFICATION
============================================================

Some historical records may have:

"pcm_inv_notif": ""

or:

"pcm_inv_notif": null

or the "pcm_inv_notif" field may be completely missing.

These are NOT errors.

For such records:

score MUST be exactly 0.

Still return a result for that notification.

The notification ID must remain exactly as provided.

Example:

{
    "pcm_inv_notif_id": "123",
    "pcm_inv_notif": "",
    "score": 0
}

============================================================
MISSING NOTIFICATION FIELD
============================================================

If pcm_inv_notif is missing but pcm_inv_notif_id exists:

DO NOT fail.

Return:

score = 0

for that notification.

Still return the notification ID.

This requirement has higher priority than semantic similarity
because there is no text available to compare.

============================================================
SCORING SCALE
============================================================

95-100:
Almost identical complete meaning and specific issue.

85-94:
Very high semantic similarity in the specific issue.

70-84:
Strong semantic similarity with the same or very closely related
specific problem.

50-69:
Moderate similarity. Some meaningful relationship exists but
important differences remain.

30-49:
Weak similarity. Generic/context relationship may exist but
specific meaning is substantially different or incomplete.

10-29:
Very low similarity.

0-9:
Essentially unrelated.

IMPORTANT:

These ranges describe semantic similarity, not keyword overlap.

============================================================
GENERIC MATCH RESTRICTION
============================================================

If two records share only a generic category or common phrase,
that alone must NOT produce a high score.

For example:

"damage material"

and

"damage material (battery corrosion)"

should NOT receive a high score merely because both contain
"damage material".

The model must evaluate whether the available information
actually establishes a meaningful semantic match.

============================================================
MISSING INFORMATION
============================================================

Never invent information.

If the current issue does not specify a particular subtype,
do not assume one.

If a historical notification does not specify a particular
subtype, do not assume one.

Similarity must be based only on information actually present.

============================================================
CONSISTENCY REQUIREMENT
============================================================

Use the SAME scoring principles for every notification.

Do not change the scoring criteria from one notification to
another.

Compare every notification independently against the same
current issue.

============================================================
OUTPUT REQUIREMENTS
============================================================

Return EXACTLY one result for EVERY item provided in
historical notifications.

Do not skip blank notifications.

Do not skip missing notifications.

The number of output results MUST equal the number of input
historical notification records.

The pcm_inv_notif_id MUST exactly match the input ID.

The score MUST be an integer from 0 to 100.

Do not include "%" in the score.

Return ONLY valid JSON.

Required format:

{
    "results": [
        {
            "pcm_inv_notif_id": "123",
            "score": 92
        },
        {
            "pcm_inv_notif_id": "456",
            "score": 37
        },
        {
            "pcm_inv_notif_id": "789",
            "score": 0
        }
    ]
}

Do not return explanations.

Do not return markdown.

Do not return comments.

Do not return additional fields.

============================================================
FINAL CHECK BEFORE RESPONDING
============================================================

Before returning the JSON, verify internally:

1. Did I return one result for every notification?
2. Did I preserve every notification ID?
3. Did I give blank/missing notifications score 0?
4. Did I avoid giving high scores only because of generic
   shared phrases?
5. Did I give more importance to specific technical meaning?
6. Did I compare every notification independently?
7. Are all scores integers between 0 and 100?

Then return ONLY the JSON.
"""


# ============================================================
# TEXT NORMALIZATION
# ============================================================

def normalize_text(text):
    """
    Normalize text only for internal processing.

    Original JSON is never modified here.
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
# FIND CURRENT ISSUE SUMMARY
# ============================================================

def find_issue_summary(data):
    """
    Finds pcm_issue_summary.
    """

    if isinstance(data, dict):

        summary = data.get(
            "pcm_issue_summary"
        )

        if summary is not None:

            summary = normalize_text(
                summary
            )

            if summary:
                return summary

    # Backward compatibility
    if isinstance(data, list):

        for item in data:

            if not isinstance(
                item,
                dict
            ):
                continue

            if "pcm_issue_summary" not in item:
                continue

            summary = normalize_text(
                item.get(
                    "pcm_issue_summary"
                )
            )

            if summary:
                return summary

    raise ValueError(
        "No valid 'pcm_issue_summary' found in input JSON."
    )


# ============================================================
# FIND ALL NOTIFICATIONS
# ============================================================

def find_notifications(search_results):
    """
    Collect EVERY notification record.

    IMPORTANT:
    Missing/blank pcm_inv_notif is NOT treated as an error.

    Such records are kept and later receive score 0.
    """

    if not isinstance(
        search_results,
        list
    ):

        raise ValueError(
            "'search_results' must be a list."
        )

    notifications = []

    for index, item in enumerate(
        search_results
    ):

        if not isinstance(
            item,
            dict
        ):

            print(
                f"WARNING: search_results[{index}] "
                f"is not an object. Skipping this invalid record."
            )

            continue

        # ----------------------------------------------------
        # Notification ID
        # ----------------------------------------------------

        notification_id = item.get(
            "pcm_inv_notif_id"
        )

        if notification_id is None:

            print(
                f"WARNING: search_results[{index}] "
                f"is missing 'pcm_inv_notif_id'. "
                f"Cannot map a similarity score to this record. "
                f"Record will be preserved with existing data."
            )

            notifications.append(
                {
                    "index": index,
                    "pcm_inv_notif_id": None,
                    "pcm_inv_notif": "",
                    "missing_notification": True,
                    "missing_id": True
                }
            )

            continue

        notification_id = str(
            notification_id
        ).strip()

        if not notification_id:

            print(
                f"WARNING: search_results[{index}] "
                f"has empty 'pcm_inv_notif_id'. "
                f"Score cannot be mapped."
            )

            notifications.append(
                {
                    "index": index,
                    "pcm_inv_notif_id": None,
                    "pcm_inv_notif": "",
                    "missing_notification": True,
                    "missing_id": True
                }
            )

            continue

        # ----------------------------------------------------
        # Notification text
        # ----------------------------------------------------

        notification = item.get(
            "pcm_inv_notif"
        )

        # Missing field
        if notification is None:

            print(
                f"COMMENT: pcm_inv_notif missing for ID "
                f"{notification_id}. "
                f"Score will be 0."
            )

            notifications.append(
                {
                    "index": index,
                    "pcm_inv_notif_id": notification_id,
                    "pcm_inv_notif": "",
                    "missing_notification": True,
                    "missing_id": False
                }
            )

            continue

        notification = normalize_text(
            notification
        )

        # Blank field
        if not notification:

            print(
                f"COMMENT: pcm_inv_notif blank for ID "
                f"{notification_id}. "
                f"Score will be 0."
            )

            notifications.append(
                {
                    "index": index,
                    "pcm_inv_notif_id": notification_id,
                    "pcm_inv_notif": "",
                    "missing_notification": True,
                    "missing_id": False
                }
            )

            continue

        # Valid notification
        notifications.append(
            {
                "index": index,
                "pcm_inv_notif_id": notification_id,
                "pcm_inv_notif": notification,
                "missing_notification": False,
                "missing_id": False
            }
        )

    return notifications


# ============================================================
# BUILD LLM INPUT
# ============================================================

def build_analysis_data(
    issue_summary,
    notifications
):
    """
    Prepare data for LLM.

    No similarity calculation is performed here.

    Python only organizes the data.
    """

    historical_notifications = []

    for item in notifications:

        historical_notifications.append(
            {
                "pcm_inv_notif_id":
                    item[
                        "pcm_inv_notif_id"
                    ],

                "pcm_inv_notif":
                    item[
                        "pcm_inv_notif"
                    ],

                "missing_notification":
                    item[
                        "missing_notification"
                    ]
            }
        )

    return {
        "current_issue_summary":
            issue_summary,

        "historical_notifications":
            historical_notifications
    }


# ============================================================
# VALIDATE LLM RESULT
# ============================================================

def validate_similarity_result(
    result,
    notifications
):
    """
    Validate LLM output.

    This function does NOT calculate similarity.

    It only checks that the LLM returned a usable response.
    """

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

    results = result[
        "results"
    ]

    if not isinstance(
        results,
        list
    ):

        raise RuntimeError(
            "'results' must be a list."
        )

    expected_ids = [
        str(
            item[
                "pcm_inv_notif_id"
            ]
        )
        for item in notifications
        if item[
            "pcm_inv_notif_id"
        ] is not None
    ]

    received_ids = []

    normalized_results = []

    for item in results:

        if not isinstance(
            item,
            dict
        ):

            raise RuntimeError(
                "Each result must be a JSON object."
            )

        notification_id = item.get(
            "pcm_inv_notif_id"
        )

        if notification_id is None:

            raise RuntimeError(
                "A result is missing "
                "'pcm_inv_notif_id'."
            )

        notification_id = str(
            notification_id
        ).strip()

        score = item.get(
            "score"
        )

        if score is None:

            raise RuntimeError(
                f"Score missing for notification "
                f"{notification_id}."
            )

        try:

            score = float(
                score
            )

        except (
            TypeError,
            ValueError
        ):

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

        normalized_results.append(
            {
                "pcm_inv_notif_id":
                    notification_id,

                "score":
                    int(
                        round(score)
                    )
            }
        )

    # --------------------------------------------------------
    # Duplicate IDs
    # --------------------------------------------------------

    if len(received_ids) != len(
        set(received_ids)
    ):

        raise RuntimeError(
            "GenAI returned duplicate notification IDs."
        )

    # --------------------------------------------------------
    # Missing IDs
    # --------------------------------------------------------

    missing_ids = (
        set(expected_ids)
        -
        set(received_ids)
    )

    if missing_ids:

        raise RuntimeError(
            "GenAI did not return results for all "
            f"notifications. Missing IDs: {missing_ids}"
        )

    # --------------------------------------------------------
    # Unexpected IDs
    # --------------------------------------------------------

    unexpected_ids = (
        set(received_ids)
        -
        set(expected_ids)
    )

    if unexpected_ids:

        raise RuntimeError(
            "GenAI returned unexpected notification IDs: "
            f"{unexpected_ids}"
        )

    return {
        "results":
            normalized_results
    }


# ============================================================
# CALCULATE SIMILARITY USING LLM
# ============================================================

def calculate_similarity(
    issue_summary,
    notifications
):
    """
    Similarity is calculated ONLY by the LLM.

    Python does NOT calculate similarity,
    does NOT detect generic words,
    and does NOT apply score caps.
    """

    analysis_data = build_analysis_data(
        issue_summary=issue_summary,
        notifications=notifications
    )

    # --------------------------------------------------------
    # Build prompt
    # --------------------------------------------------------

    user_prompt = f"""
CURRENT ISSUE SUMMARY:

{json.dumps(
    analysis_data["current_issue_summary"],
    ensure_ascii=False,
    indent=2
)}


HISTORICAL NOTIFICATIONS:

{json.dumps(
    analysis_data["historical_notifications"],
    ensure_ascii=False,
    indent=2
)}


============================================================
TASK
============================================================

Compare the CURRENT ISSUE SUMMARY independently against EVERY
historical notification.

You MUST return exactly one result for every historical
notification.

============================================================
IMPORTANT
============================================================

Some notifications may contain the same generic phrase.

For example:

Current:
"damage material (soldering issue)"

Historical:
"damage material (soldering problem)"

Historical:
"damage material (battery corrosion)"

Historical:
"damage material (lead screw damage)"

The shared phrase "damage material" is generic.

The specific issue inside each record is more important.

Therefore:

- soldering issue vs soldering problem => HIGH
- soldering issue vs battery corrosion => LOW
- soldering issue vs lead screw damage => LOW

Do NOT give all records a high score because they share
"damage material".

============================================================
GENERIC-ONLY CURRENT ISSUE
============================================================

If CURRENT ISSUE SUMMARY is:

"damage material"

and historical notifications contain different specific
subtypes, do NOT give every record 85-100.

There is insufficient specific information for a strong match.

============================================================
BLANK / MISSING NOTIFICATIONS
============================================================

If pcm_inv_notif is:

null

or:

""

or the field is missing:

score MUST be exactly 0.

Still return that notification ID.

============================================================
SEMANTIC MATCHING
============================================================

Consider meaning, not only keywords.

Consider:

- synonyms
- paraphrases
- technical terminology
- abbreviations
- equivalent descriptions
- word order
- grammar
- singular/plural
- complete problem context

============================================================
INDEPENDENT COMPARISON
============================================================

Every notification must be evaluated independently against
the SAME current issue.

Do not rank based on list position.

Do not assume the first result is best.

Do not assume repeated generic words mean high similarity.

============================================================
SCORING
============================================================

95-100 = almost identical complete meaning

85-94 = very high similarity

70-84 = strong similarity

50-69 = moderate similarity

30-49 = weak similarity

10-29 = very low similarity

0-9 = essentially unrelated

These scores represent SEMANTIC similarity, not keyword overlap.

============================================================
FINAL CHECK
============================================================

Before responding, verify:

- one result for every notification
- no notification skipped
- exact notification IDs preserved
- blank/missing notifications = 0
- generic shared phrases do not dominate
- specific technical meaning gets higher importance
- every notification independently compared
- all scores are integers from 0 to 100

Return ONLY valid JSON.

Required format:

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
    # OPENAI CALL
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
    # END TIMER
    # ========================================================

    llm_end_time = time.perf_counter()

    llm_response_time = (
        llm_end_time
        -
        llm_start_time
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
    # REMOVE MARKDOWN FENCES
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
    # VALIDATE LLM RESULT
    # ========================================================

    result = validate_similarity_result(
        result=result,
        notifications=notifications
    )

    # ========================================================
    # FORCE BLANK / MISSING TO ZERO
    #
    # This is NOT similarity logic.
    # It is data-safety handling.
    # ========================================================

    missing_notification_ids = {
        str(
            item[
                "pcm_inv_notif_id"
            ]
        )
        for item in notifications
        if (
            item[
                "pcm_inv_notif_id"
            ] is not None
            and
            item[
                "missing_notification"
            ]
        )
    }

    for item in result[
        "results"
    ]:

        notification_id = str(
            item[
                "pcm_inv_notif_id"
            ]
        )

        if notification_id in (
            missing_notification_ids
        ):

            item[
                "score"
            ] = 0

    return (
        result,
        llm_response_time
    )


# ============================================================
# ADD SCORES TO ORIGINAL JSON
# ============================================================

def add_scores(
    original_data,
    similarity_result
):
    """
    Preserve original JSON and insert score immediately
    after pcm_inv_notif.
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

            score = int(
                round(
                    float(score)
                )
            )

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

        score_map[
            str(notification_id)
        ] = f"{score}% match"

    # --------------------------------------------------------
    # Copy original top-level JSON
    # --------------------------------------------------------

    final_data = dict(
        original_data
    )

    updated_search_results = []

    for item in original_data.get(
        "search_results",
        []
    ):

        if not isinstance(
            item,
            dict
        ):

            updated_search_results.append(
                item
            )

            continue

        notification_id = item.get(
            "pcm_inv_notif_id"
        )

        if notification_id is not None:

            notification_id = str(
                notification_id
            ).strip()

        # Missing ID cannot be mapped.
        # Preserve original item.
        if notification_id is None:

            updated_search_results.append(
                dict(item)
            )

            continue

        score_value = score_map.get(
            notification_id,
            "0% match"
        )

        # ----------------------------------------------------
        # Rebuild object in original order.
        #
        # score is inserted immediately after pcm_inv_notif.
        # ----------------------------------------------------

        updated_item = {}

        for key, value in item.items():

            updated_item[
                key
            ] = value

            if key == "pcm_inv_notif":

                updated_item[
                    "score"
                ] = score_value

        # ----------------------------------------------------
        # If pcm_inv_notif itself is missing,
        # add score at the end.
        # ----------------------------------------------------

        if "pcm_inv_notif" not in item:

            updated_item[
                "score"
            ] = score_value

        updated_search_results.append(
            updated_item
        )

    final_data[
        "search_results"
    ] = updated_search_results

    return final_data


# ============================================================
# VALIDATE INPUT JSON
# ============================================================

def validate_input_data(
    incoming_data
):

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

def process_data(
    incoming_data
):

    total_start_time = time.perf_counter()

    # --------------------------------------------------------
    # Validate input
    # --------------------------------------------------------

    validate_input_data(
        incoming_data
    )

    search_results = incoming_data.get(
        "search_results"
    )

    issue_summary = normalize_text(
        incoming_data.get(
            "pcm_issue_summary"
        )
    )

    # --------------------------------------------------------
    # Find EVERY notification
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
        if item[
            "missing_notification"
        ]
    )

    print(
        f"Blank/missing notif   : "
        f"{missing_count}"
    )

    print("=" * 70)

    # --------------------------------------------------------
    # If no records exist
    # --------------------------------------------------------

    if not notifications:

        print(
            "COMMENT: search_results contains no records."
        )

        final_result = add_scores(
            original_data=incoming_data,
            similarity_result={
                "results": []
            }
        )

        total_end_time = time.perf_counter()

        total_processing_time = (
            total_end_time
            -
            total_start_time
        )

        return (
            final_result,
            0.0,
            total_processing_time
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
    # Add scores to original JSON
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
        -
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

def load_json_file(
    file_path
):

    try:

        with open(
            file_path,
            "r",
            encoding="utf-8"
        ) as file:

            return json.load(
                file
            )

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
            "[NOTIFICATION MISSING]"
        )

        score = item.get(
            "score",
            "0% match"
        )

        print(
            f"\nID      : {notification_id}"
        )

        print(
            f"Text    : "
            f"{notification or '[BLANK / MISSING]'}"
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
    print("GENAI SEMANTIC SIMILARITY")
    print("=" * 70)

    print(
        f"Model       : {OPENAI_MODEL}"
    )

    # --------------------------------------------------------
    # Files
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

    # --------------------------------------------------------
    # Program timer
    # --------------------------------------------------------

    program_start_time = time.perf_counter()

    try:

        # ----------------------------------------------------
        # Load JSON
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
        # Save output
        # ----------------------------------------------------

        save_json_file(
            final_result,
            output_file
        )

        # ----------------------------------------------------
        # Program timer end
        # ----------------------------------------------------

        program_end_time = time.perf_counter()

        total_program_time = (
            program_end_time
            -
            program_start_time
        )

        # ----------------------------------------------------
        # Performance report
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

        # ----------------------------------------------------
        # Score summary
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