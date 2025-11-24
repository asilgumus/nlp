from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline
import webbrowser
import re
import time
import threading
import os
import datetime
import random
import spacy

# ===================== 🔧 MODELLER ===================== #
intent_model = AutoModelForSequenceClassification.from_pretrained("yeniguno/bert-uncased-turkish-intent-classification")
intent_tokenizer = AutoTokenizer.from_pretrained("yeniguno/bert-uncased-turkish-intent-classification")
intent_pipe = pipeline("text-classification", model=intent_model, tokenizer=intent_tokenizer)
ner_pipe = pipeline("ner", model="savasy/bert-base-turkish-ner-cased", grouped_entities=True)

import stanza
stanza.download('tr')  # sadece 1 kez
nlp = stanza.Pipeline('tr')

# ===================== 🧠 INTENT ===================== #
def get_intent(text):
    result = intent_pipe(text)
    return result[0]["label"] if result else None

# ===================== 🎵 MÜZİK ===================== #
def extract_song_name(text):
    
    entities = ner_pipe(text)
    print(entities)
    for ent in entities:
        if ent["entity_group"] in ["MISC", "PER", "ORG"] and ent["score"] > 0.80:
            word = ent["word"]
            if " by " in word.lower():
                word = word.split(" by ")[0]
            return word.strip()

    # NER yakalayamazsa, yedek çözüm: "Shape of You dinlemek istiyorum"
    doc = nlp(text)
    keywords = []
    for token in doc:
        if token.ent_type_ == "" and token.pos_ in ["PROPN", "NOUN"]:
            keywords.append(token.text)
    return " ".join(keywords).strip() if keywords else None


def open_in_spotify(song_name):
    print(f"🎵 Spotify'da aratılıyor: {song_name}")
    url = "https://open.spotify.com/search/" + song_name.replace(" ", "%20")
    webbrowser.open(url)

def handle_play_music(text):
    song_name = extract_song_name(text)
    if song_name:
        open_in_spotify(song_name)
    else:
        print("🎤 Hangi şarkıyı dinlemek istersiniz?")
        while True:
            user_input = input("👤: ")
            if user_input.strip().lower() in ["fikrimi değiştirdim", "neyse", "vazgeçtim"]:
                print("🚫 İşlem iptal edildi.")
                break
            song_name = extract_song_name(user_input)
            if song_name:
                open_in_spotify(song_name)
                break
            elif len(user_input.strip().split()) <= 6:
                open_in_spotify(user_input.strip())
                break
            else:
                print("⏳ Anlayamadım, tekrar deneyin ya da sadece şarkı adını yazın.")

# ===================== 🌤️ HAVA DURUMU ===================== #
def extract_location(text):
    entities = ner_pipe(text)
    for ent in entities:
        if ent["entity_group"] == "LOC" and ent["score"] > 0.80:
            return ent["word"].strip()
    return None

def give_weather_info(text):
    location = extract_location(text)
    if location:
        print(f"📍 {location} için hava durumu aranıyor...")
        query = f"{location} hava durumu"
        webbrowser.open("https://www.google.com/search?q=" + query.replace(" ", "+"))
    else:
        user_input = input("📍 Hangi şehir için?")
        location = extract_location(user_input)
        if location:
            webbrowser.open("https://www.google.com/search?q=" + f"{location} hava durumu".replace(" ", "+"))

# ===================== ⏰ ZAMANLAYICI ===================== #
def extract_timer_duration(text):
    text = text.lower()
    pattern = r'(\d+)\s*(seconds?|saniye|minutes?|dakika)'
    matches = re.findall(pattern, text)
    total_seconds = 0
    for value, unit in matches:
        value = int(value)
        if "minute" in unit or "dakika" in unit:
            total_seconds += value * 60
        elif "second" in unit or "saniye" in unit:
            total_seconds += value
    return total_seconds if total_seconds > 0 else None

def start_timer(seconds):
    def countdown():
        print(f"⏳ {seconds} saniyelik zamanlayıcı başlatıldı...")
        time.sleep(seconds)
        print("🔔 Zamanlayıcı süresi doldu!")
    threading.Thread(target=countdown).start()

def handle_timer(text):
    duration = extract_timer_duration(text)
    if duration:
        start_timer(duration)
    else:
        print("❓ Süreyi anlayamadım. Örn: '5 dakika'")
        user_input = input("👤: ")
        duration = extract_timer_duration(user_input)
        if duration:
            start_timer(duration)

# ===================== 🔍 GOOGLE ARAMA ===================== #
def handle_search(text):
    print(f"🔍 Google'da aranıyor: {text}")
    webbrowser.open("https://www.google.com/search?q=" + text.replace(" ", "+"))

# ===================== 📅 SAAT / ŞAKA / YARDIM ===================== #
def tell_time():
    print("🕓 Şu an saat:", datetime.datetime.now().strftime("%H:%M"))

def tell_joke():
    jokes = [
        "RAM'ler neden kavga etti? Çünkü biri diğerinin alanını işgal etti!",
        "Python yılan değildir, kod yazanın ta kendisidir!",
    ]
    print("😂", random.choice(jokes))

def show_help():
    print("""🧠 Yapabileceklerim:
- Müzik çal (play_music)
- Hava durumu (get_weather)
- Zamanlayıcı (create_timer)
- Google'da ara (information_query)
- Şaka yap (fun_fact)
- Saat söyle (time_query)
- Şehir sorgula (get_location)
- Rezervasyon ayarla (set_reservation)
- Alarm oluştur (create_alarm)
- Hatırlatıcı oluştur (create_reminder)
- Yardım iste (help_query)
""")

# ===================== 🗣️ ÇEVİRİ İŞLEMİ ===================== #
def clean_translation_input(text):
    """
    cümledeki ROOT (komut fiil) token'ını çıkarır
    """
    doc = nlp(text)
    root_token = None
    for token in doc:
        if token.dep_ == "ROOT":
            root_token = token
            break
    if root_token:
        # ROOT kelimesi dışında kalan kelimeler
        tokens = [t.text for t in doc if t.i != root_token.i]
        return " ".join(tokens).strip()
    return text

def extract_target_language(text):
    """
    cümledeki hedef dili bulmaya çalışır.
    Basitçe ner_pipe ile LOC veya MISC içinde dil isimleri aranabilir.
    Burada İngilizce, Almanca, Türkçe gibi kelimeleri örnek aldım.
    """
    languages = ["english", "german", "turkish", "french", "spanish", "italian", "japanese", "korean", "chinese", "arabic", "russian"]
    doc = nlp(text.lower())
    for token in doc:
        if token.text in languages:
            return token.text
    # Ayrıca ner_pipe'den de bakabiliriz
    entities = ner_pipe(text)
    for ent in entities:
        if ent["entity_group"] in ["LOC", "MISC"] and ent["word"].lower() in languages:
            return ent["word"].lower()
    return None

def handle_translation(text):
    cleaned_text = clean_translation_input(text)
    target_lang = extract_target_language(text)
    if not target_lang:
        print("❓ Hangi dile çevirmek istediğinizi anlayamadım.")
        target_lang = input("👤 Lütfen hedef dili yazınız (ör: german, english): ").strip().lower()
    if not cleaned_text:
        print("❓ Çevrilecek metin bulunamadı.")
        return
    print(f"🌐 '{cleaned_text}' metni {target_lang} diline çevriliyor...")
    # Google Translate araması yapalım
    query = f"translate {cleaned_text} to {target_lang}"
    url = "https://translate.google.com/?sl=auto&tl=" + target_lang + "&text=" + cleaned_text.replace(" ", "%20") + "&op=translate"
    webbrowser.open(url)

# ===================== ANA AKIŞ ===================== #
def main():
    while True:
        user_text = input("🎙️ Ne yapmak istersiniz?\n👤: ")
        if user_text.strip().lower() in ["çık", "çıkış", "exit", "quit"]:
            print("👋 Görüşmek üzere!")
            break
        intent = get_intent(user_text)
        print(f"📌 Algılanan intent: {intent}")

        if intent in ["play_music", "music_query"]:
            handle_play_music(user_text)
        elif intent in ["get_weather", "weather_query"]:
            give_weather_info(user_text)
        elif intent in ["create_timer", "add_timer"]:
            handle_timer(user_text)
        elif intent in ["information_query", "search_event", "search_creative_work"]:
            handle_search(user_text)
        elif intent == "time_query":
            tell_time()
        elif intent == "fun_fact":
            tell_joke()
        elif intent == "help_query":
            show_help()
        elif intent in ["get_location", "location_query"]:
            print("📍 Konum bilgisi: İzmir, Türkiye (örnek veridir)")
        elif intent == "create_alarm":
            print("⏰ Alarm oluşturuluyor... (detaylı işlem eklenebilir)")
        elif intent == "create_reminder":
            print("📌 Hatırlatıcı ayarlanıyor... (detaylı işlem eklenebilir)")
        elif intent == "set_reservation":
            print("📅 Rezervasyon işlemi başlatıldı... (detaylı işlem eklenebilir)")
        elif intent == "translation_intent":
            handle_translation(user_text)
        else:
            print("🤖 Komut anlaşılamadı, yardım almak için 'yardım' yazabilirsiniz.")


if __name__ == "__main__":
    main()
