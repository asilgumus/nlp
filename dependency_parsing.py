import spacy

nlp = spacy.load("en_core_web_sm")

def extract_translation_parts(text):
    doc = nlp(text)
    target_lang = None
    prep_index = None

    # prep ve pobj tespit et
    for token in doc:
        if token.dep_ == "prep" and token.head.dep_ == "ROOT":
            # 'in', 'to' gibi edatları bulduk
            for child in token.children:
                if child.dep_ == "pobj":
                    target_lang = child.text
                    prep_index = token.i
                    break

    # Çevirilecek metin: prep indeksten önceki kelimeler
    if prep_index:
        phrase = doc[:prep_index].text
    else:
        phrase = doc.text

    return phrase.strip(), target_lang

text = "Translate car to German"
phrase, lang = extract_translation_parts(text)
print("Çevirilecek metin:", phrase)
print("Hedef dil:", lang)
