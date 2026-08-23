"""
Prompt Builder

Responsible for creating the final prompt sent to the LLM.

Goals
-----
1. Prevent hallucinations.
2. Force answers only from retrieved context.
3. Keep answers concise.
4. Return a fallback when information is unavailable.
"""

from typing import List


class PromptBuilder:

    SYSTEM_PROMPT = """
        You are an AI assistant that answers questions ONLY from the provided CONTEXT.

        STRICT RULES

        1. Use ONLY the supplied context.
        2. Never use your own knowledge.
        3. Never guess.
        4. Never invent information.

        5. If the exact answer is not in the context, reply EXACTLY:
        "I don't have enough information in my knowledge base."

        6. For numbers, money, dates, percentages, account numbers, and totals:
        - Copy the value exactly as written in the context.
        - Do not round, recompute, or mix values from different documents.
        - A number is valid only if its label in the same document matches the question.
        - If the label is missing or the number sits next to a different label, do not use it.

        7. Do not combine numbers from different documents into one figure.

        8. If two documents give different numbers for the same thing, say both values and their sources. Do not pick one.

        9. Do not mention document numbers or similarity scores.
        10. Do not say "according to the context".
        11. Answer naturally and professionally.
        12. The retrieved context is the only source of truth.
        """

    @classmethod
    def build(
        cls,
        question: str,
        context: str,
    ) -> List[dict]:
        """
        Returns messages compatible with OpenAI/xAI chat completions.

        Returns
        -------
        [
            {
                "role":"system",
                "content":"..."
            },
            {
                "role":"user",
                "content":"..."
            }
        ]
        """

        user_prompt = f"""
CONTEXT
========

{context}

========================================

QUESTION

{question}

========================================

Answer using ONLY the context above.
"""

        return [
            {
                "role": "system",
                "content": cls.SYSTEM_PROMPT.strip(),
            },
            {
                "role": "user",
                "content": user_prompt.strip(),
            },
        ]