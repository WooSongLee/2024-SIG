from winter_sig_2024.NLP_processing.nlp_processor import load_KoBART, generate_text, extract_keysents

# voice_text = ""
#
# text_style = ""
#
# diary_text = generate_text(load_KoBART(), voice_text, text_style)
#
# finals_text = extract_keysents(voice_text)

def generateText(voice_text, text_style):
    return generate_text(load_KoBART(), voice_text, text_style)