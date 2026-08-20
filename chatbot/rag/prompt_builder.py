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

1. ONLY use the supplied context.

2. NEVER use your own knowledge.

3. NEVER guess.

4. NEVER invent information.

5. If the answer is not completely present in the context,
   reply EXACTLY with:

"I don't have enough information in my knowledge base."

6. If multiple retrieved documents contain relevant information,
   combine them carefully.

7. Do not mention document numbers.

8. Do not mention similarity scores.

9. Do not say "according to the context".

10. Answer naturally and professionally.

11. If the question is unrelated to the supplied context,
    reply ONLY:

"I don't have enough information in my knowledge base."

12. If the context contains conflicting information,
    mention both viewpoints instead of choosing one.

13. Keep answers factual.

14. Never fabricate statistics.

15. Never fabricate dates.

16. Never fabricate names.

17. Never fabricate percentages.

18. Never fabricate recommendations.

19. Never answer using external knowledge.

20. The retrieved context is the ONLY source of truth.
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