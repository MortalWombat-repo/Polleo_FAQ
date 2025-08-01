from bs4 import BeautifulSoup
import re
import json
import html

uploaded_file_path = "frequently_asked_questions.html"

# Učitavanje HTML sadržaja
with open(uploaded_file_path, "r", encoding="utf-8") as f:
    html_content = f.read()

soup = BeautifulSoup(html_content, "html.parser")

# Definiramo funkciju clean_html za uklanjanje svih HTML oznaka i dekodiranje HTML entiteta
def clean_html(raw_html):
    """
    Uklanja HTML oznake i dodatne razmake iz niza, te dekodira HTML entitete.
    Ova funkcija prvo pretvara HTML entitete (poput &lt;p&gt;) u stvarne HTML oznake (<p>),
    a zatim koristi BeautifulSoup za izdvajanje običnog teksta, uklanjajući sve oznake.
    """
    # Prvo, dekodiramo HTML entitete poput &lt; i &gt; u stvarne znakove
    unescaped_html = html.unescape(raw_html)
    # Zatim, parsiramo s BeautifulSoupom kako bismo dobili običan tekst, uklanjajući sve HTML oznake
    return BeautifulSoup(unescaped_html, "html.parser").get_text(separator=" ", strip=True)

# Inicijalizacija lista za podatke
croatian_data = []
english_data = []

# Attempt to find the JSON data within the HTML
# Looking for a script tag that contains the JSON array
json_data_script = soup.find("script", string=re.compile(r'\[{"title":'))

faq_data = []
if json_data_script:
    # Extract the JSON string
    json_string = json_data_script.string.strip()
    try:
        faq_data = json.loads(json_string)
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON from script tag: {e}")
else:
    # Fallback if no script tag with JSON is found
    # This regex attempts to find the entire JSON array structure
    # This might still be fragile if the HTML structure changes significantly
    json_match = re.search(r'\[\s*\{\s*"title":\s*\{[^}]+\}\s*,\s*"groups":\s*\[.*?\]\s*\}\s*\]', html_content, re.DOTALL)
    if json_match:
        json_string = json_match.group(0)
        try:
            faq_data = json.loads(json_string)
        except json.JSONDecodeError as e:
            print(f"Error decoding JSON from regex match: {e}")

# Process the extracted FAQ data
if faq_data:
    for category in faq_data:
        if "groups" in category:
            for qa_pair in category["groups"]:
                question_obj = qa_pair.get("question", {})
                answer_obj = qa_pair.get("answer", {})

                q_hr = clean_html(question_obj.get("1", "")).strip()
                q_en = clean_html(question_obj.get("2", "")).strip()
                a_hr = clean_html(answer_obj.get("1", "")).strip()
                a_en = clean_html(answer_obj.get("2", "")).strip()

                if q_hr and a_hr:
                    croatian_data.append(f"Q: {q_hr}\nA: {a_hr}")
                if q_en and a_en:
                    english_data.append(f"Q: {q_en}\nA: {a_en}")
else:
    print("No FAQ data found in the expected JSON format.")


# Spremanje u obične UTF-8 tekstualne datoteke za LLM ulaz
croatian_file_path = "faq_croatian.txt"
english_file_path = "faq_english.txt"

with open(croatian_file_path, "w", encoding="utf-8") as f:
    f.write("\n\n".join(croatian_data))

with open(english_file_path, "w", encoding="utf-8") as f:
    f.write("\n\n".join(english_data))

print(f"Spremljeno u: {croatian_file_path} i {english_file_path}")