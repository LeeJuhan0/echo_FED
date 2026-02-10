import openai
from concurrent.futures import ThreadPoolExecutor
import tiktoken
import os

# Add your own OpenAI API key
openai_api_key = os.getenv("OPENAI_API_KEY")

sum_prompt = """
Given the raw minutes text, generate a structured summary that highlights the key elements under the following flexible categories. Only include a category if it is relevant:

MONETARY_POLICY_DIRECTION: The stance or change in monetary policy (e.g., hawkish, dovish, neutral).
KEY_DISCUSSION_TOPICS: Main themes discussed (e.g., inflation, employment, external risks).
ECONOMIC_INDICATORS_MENTIONED: Specific indicators referenced (e.g., CPI, GDP, exports).
INFLATION_OUTLOOK: Any projections or perspectives on inflation.
GROWTH_OUTLOOK: Growth expectations or concerns raised.
RISKS_AND_CONCERNS: Notable uncertainties, both domestic and global.
DISSENTING_OPINIONS: If any committee members disagreed or expressed differing views.
POLICY_TRIGGERS: What conditions or data trends would prompt future policy changes.
MARKET_IMPLICATIONS: Hints or signals toward financial market expectations.
NOTABLE_QUOTES: Any standout quotes from the minutes that encapsulate sentiment.
SUMMARY_TONE: Overall sentiment/tone (e.g., cautious optimism, uncertainty, confidence).

The output should be formatted as:
CATEGORY: Summary content (1–2 sentences, precise and professional)
Only include relevant categories for the given input. Avoid generic repetition or over-explanation.
"""

def call_openai_api(chunk):
    response = openai.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": sum_prompt},
            {"role": "user", "content": f" {chunk}"},
        ],
        max_tokens=500,
        n=1,
        stop=None,
        temperature=0.5,
    )
    return response.choices[0].message.content

def split_into_chunks(text, tokens=500):
    encoding = tiktoken.encoding_for_model('gpt-4o')
    words = encoding.encode(text)
    chunks = []
    for i in range(0, len(words), tokens):
        chunks.append(' '.join(encoding.decode(words[i:i + tokens])))
    return chunks   

def process_chunks(content):
    chunks = split_into_chunks(content)

    # Processes chunks in parallel
    with ThreadPoolExecutor() as executor:
        responses = list(executor.map(call_openai_api, chunks))
    # print(responses)
    return responses

if __name__ == "__main__":
    content = "sth you wanna test"
    process_chunks(content)

# Can take up to a few minutes to run depending on the size of your data input