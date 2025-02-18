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

text = """
오늘은 평범하지만 소소한 즐거움이 있는 하루였다. 아침에 일어나 창문을 열어보니 약간 쌀쌀한 공기가 방 안으로 스며들었다. 따뜻한 커피 한 잔을 마시며 하루를 시작했다.

하루 종일 이것저것 배우고, 고민하고, 작은 성취를 느끼는 순간들이 있었다. 코드를 짜다가 막히면 답답했지만, 해결했을 때의 뿌듯함이 더 컸다.

저녁엔 잠시 쉬면서 좋아하는 음악을 들었는데, 하루의 피로가 싹 가시는 기분이었다. 내일은 오늘보다 조금 더 나아지길 바라며 하루를 마무리한다.

"조금씩 나아가면 결국 도착하게 된다." 오늘 나에게 했던 말 중 가장 마음에 남는 한마디다. 
"""

print(generateText(text, '문어체'))
print(generateText(text, 'informal'))
print(generateText(text, 'android'))
print(generateText(text, 'azae'))
print(generateText(text, 'chat'))
print(generateText(text, 'choding'))
print(generateText(text, 'emoticon'))
print(generateText(text, 'enfp'))
print(generateText(text, 'gentle'))
print(generateText(text, 'halbae'))
print(generateText(text, 'halmae'))
print(generateText(text, 'joongding'))
print(generateText(text, 'king'))
print(generateText(text, 'naruto'))
print(generateText(text, 'seonbi'))
print(generateText(text, 'sosim'))
print(generateText(text, 'translator'))
