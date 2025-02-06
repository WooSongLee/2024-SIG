from nlp_processor import load_KoBART, generate_text, extract_keysents

voice_text = ""

text_style = ""

diary_text = generate_text(load_KoBART(), voice_text, text_style)

finals_text = extract_keysents(voice_text)