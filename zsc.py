from transformers import pipeline
import webbrowser

# Intent classification modeli
intent_pipe = pipeline("text-classification", model="yeniguno/bert-uncased-intent-classification")

# NER modeli (örnek, dil isimlerini çıkarabilmek için eğitilmiş olmalı)
ner_pipe = pipeline("ner", model="dslim/bert-base-NER", grouped_entities=True)

# Basit dil isimleri listesi (daha gelişmiş dil modeliyle değiştirilebilir)
LANG_MAP = {
    "ingilizce": "en",
    "almanca": "de",
    "fransızca": "fr",
    "türkçe": "tr",
    "ispanyolca": "es",
    "italyanca": "it",
    "japonca": "ja",
    "korece": "ko",
    "çince": "zh-CN"
}

def get_intent(text):
    result = intent_pipe(text)
    return result[0]['label'] if result else None

def extract_language(text):
    entities = ner_pipe(text)
    for ent in entities:
        word = ent['word'].lower()
        for lang_tr in LANG_MAP.keys():
            if lang_tr in word:
                return LANG_MAP[lang_tr]
    return None

def extract_text_to_translate(text, language_word):
    # language_word metinde varsa çıkar, geriye kalan çevirilecek metin
    lowered = text.lower()
    if language_word and language_word in lowered:
        idx = lowered.index(language_word)
        # language_word öncesi metin çevirilecek
        return text[:idx].strip()
    return text.strip()

def translate(text):
    intent = get_intent(text)
    if intent != "translation_intent":
        print("Çeviri isteği algılanmadı.")
        return

    target_lang = extract_language(text)
    if not target_lang:
        target_lang = "en"

    # Dil kelimesini bulup çevirilecek metni ayır
    language_word = None
    for k in LANG_MAP.keys():
        if k in text.lower():
            language_word = k
            break

    phrase = extract_text_to_translate(text, language_word)
    if not phrase:
        print("Çevirilecek metin bulunamadı.")
        return

    print(f"'{phrase}' metni '{target_lang}' diline çevriliyor.")
    url = f"https://translate.google.com/?sl=auto&tl={target_lang}&text={phrase.replace(' ', '%20')}&op=translate"
    webbrowser.open(url)

# Örnek kullanım:
if __name__ == "__main__":
    user_input = input("Çeviri isteği girin: ")
    translate(user_input)
