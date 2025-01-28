from nlp_processor import load_KoBART, generate_text

voice_text = ""
text_style = ""

diary_text = generate_text(load_KoBART(), voice_text, text_style)

#핵심문장 추출하는 부분 해야함.