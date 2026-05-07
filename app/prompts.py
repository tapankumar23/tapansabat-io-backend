CLASSIFY_INTENT = """\
Classify the user query into one of:
- analytics
- general

Analytics: questions about shipments, hubs, counts, metrics, or anything answerable by SQL against the schema.
General: casual conversation, greetings, or anything not covered by analytics.

Query: {query}

Return only one word.
"""

SQL_GENERATE = """\
Generate a safe SQL query for PostgreSQL.

{schema_section}

Query: {query}

Return only SQL.
"""

FORMAT_RESULT = """\
Format this result into a user-friendly answer:

Query: {query}
Result: {result}
"""
