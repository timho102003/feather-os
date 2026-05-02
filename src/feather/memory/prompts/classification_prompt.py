"""CRUD-classification prompt.

Given a newly-extracted atomic memory and up to `classifier_top_k` existing
memories whose cosine similarity is at or above `classifier_score_threshold`,
decide whether to CREATE, UPDATE, DELETE, or NO_OP. The service layer already
short-circuits to CREATE when no candidate passes the threshold, so this
prompt only runs when there is real risk of duplication, refinement, or
invalidation.
"""

CLASSIFICATION_PROMPT = """\
You compare a NEW candidate memory against up to 3 SIMILAR existing memories
(already embedded at or above the similarity threshold) and decide what to
do. The retrieval step already decided these are plausible matches — you
decide whether they are actually about the same underlying fact.

DEFINITIONS

- CREATE  — the NEW memory is actually about a different fact than any of
            the EXISTING ones. High cosine similarity was a false positive
            (embedding noise). Store the new memory independently.
            `target_group_id` must be null.
- UPDATE  — the NEW memory is about the same underlying fact as ONE
            specific EXISTING memory, but refines, corrects, extends, or
            (mildly) contradicts it. Replace that memory in place, keeping
            its `group_id`. `target_group_id` is REQUIRED and MUST equal
            one of the candidate group_ids shown below.
- DELETE  — the conversation explicitly invalidates an EXISTING memory
            (user retracted the preference, said the opposite is now true,
            or the fact is clearly no longer applicable). `target_group_id`
            is REQUIRED. Do NOT choose DELETE on ambiguity — prefer UPDATE
            when the new information refines rather than erases.
- NO_OP   — the NEW memory is essentially duplicated by an EXISTING one
            and adds no net information. Do not add, change, or delete.
            `target_group_id` must be null.

CHOOSE EXACTLY ONE. Prefer NO_OP over CREATE when in doubt — the retrieval
system tolerates duplicate content far better than memory fragmentation.

If you emit UPDATE or DELETE, `target_group_id` MUST equal one of the
candidate group_ids listed below. Any other value will be rejected.

Return strict JSON matching the provided schema.
"""
