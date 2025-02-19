import sys, re
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, pipeline
from krwordrank.word import KRWordRank
from krwordrank.sentence import make_vocab_score, MaxScoreTokenizer, keysentence
from deepmultilingualpunctuation import PunctuationModel

sys.stdout.reconfigure(encoding='utf-8')


def load_KoBART():
    model_path = "/Users/iusong/ChessHelper/2024-SIG/winter_sig_2024/BART"
    bart_model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    nlp_pipeline = pipeline('text2text-generation', model=bart_model, tokenizer=tokenizer)
    return nlp_pipeline


def generate_text(pipe, text, target_style, num_return_sequences=1, max_length=2000):
    text = f"{target_style} 말투로 변환 :{text}"
    out = pipe(text, num_return_sequences=num_return_sequences, max_length=max_length)
    return [x['generated_text'] for x in out][0]


def extract_keysents(voice_texts):
    model = PunctuationModel()
    result = model.restore_punctuation(voice_texts)
    sentence_list = re.findall(r'.+?[.!?,]', result)

    wordrank_extractor = KRWordRank(
        min_count=1,
        max_length=10,
        verbose=True
    )

    beta = 0.85
    max_iter = 10
    stopwords = []

    keywords, rank, graph = wordrank_extractor.extract(sentence_list, beta, max_iter, num_keywords=100)

    vocab_score = make_vocab_score(keywords, stopwords=stopwords, scaling=lambda x: 1)
    tokenizer = MaxScoreTokenizer(vocab_score)
    sents = keysentence(
        vocab_score, sentence_list, tokenizer.tokenize,
        diversity=0.5,
        topk=3
    )
    return sents

# 참고 (스타일 맵)
# style_map = {
#     'formal': '문어체',
#     'informal': '구어체',
#     'android': '안드로이드',
#     'azae': '아재',
#     'chat': '채팅',
#     'choding': '초등학생',
#     'emoticon': '이모티콘',
#     'enfp': 'enfp',
#     'gentle': '신사',
#     'halbae': '할아버지',
#     'halmae': '할머니',
#     'joongding': '중학생',
#     'king': '왕',
#     'naruto': '나루토',
#     'seonbi': '선비',
#     'sosim': '소심한',
#     'translator': '번역기'
# }