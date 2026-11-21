# [AI 상담원 서버] 고객 목소리를 듣고(STT) → 생각하고(LLM, 필요하면 전산실 조회) → 목소리로 답하는(TTS) 본체입니다 (9000번).
# 왜 있나: 대회에서 우리가 만들어야 하는 핵심이에요. 말 끝 판단, 문지기, 끝 신호(turn.end) 안전망까지 맡아요.
# 매뉴얼(말투·규칙)은 같은 폴더의 prompt.md + data/brand/ 에, API 키는 같은 폴더의 .env (또는 맨 위 폴더의 .env) 에 적습니다.
"""DIRECT:ON 2026 AI 상담원 서버 (WebSocket, 기본 9000번 포트).

전화 한 통 = 연결 한 번:
  듣기(말 끝 판단, 목소리는 Soniox 로 실시간 전달) → 받아쓰기 확정 → 생각(LLM + 모의 함수 서버) → 말하기(TTS) → turn.end

대회 지정 모델:  STT = Soniox stt-rt-v6 / LLM = gpt-5.4-mini / TTS = Gemini 3.8 Flash TTS (gemini-3.8-flash-tts, 운영진 공지로 2.5 에서 변경)
실행:  python agent_server/server.py --port 9000
준비:  pip install -r requirements.txt
       agent_server/.env 에 BAZE_API_KEY (대회 게이트웨이 키 하나로 셋 다) 또는 OPENAI_API_KEY, SONIOX_API_KEY, GEMINI_API_KEY 를 적는다 (.env.example 을 복사해서 사용)
"""
import argparse, asyncio, contextvars, difflib, hashlib, json, math, os, random, re, time
from collections import deque
from contextlib import suppress
from dataclasses import dataclass, field
from pathlib import Path

import httpx, numpy as np, websockets
from dotenv import load_dotenv
from google import genai
from google.genai import errors as genai_errors
from google.genai import types
from openai import AsyncOpenAI
from scipy.signal import resample_poly

ROOT = Path(__file__).resolve().parents[1]
for env in (Path(__file__).with_name(".env"), ROOT / ".env"):     # agent_server/.env 와 맨 위 폴더 .env 를 모두 읽는다 (제출할 땐 맨 위 폴더). 먼저 읽은 값, 이미 설정된 환경변수가 우선. 메모장 BOM 도 OK
    load_dotenv(env, encoding="utf-8-sig")
import jev                                                # Jev(말 끝 판정·업무 분류). .env 를 읽은 뒤에 불러와야 키·설정이 반영된다
BAZE_KEY = os.getenv("BAZE_API_KEY")                     # 대회 BAZE API Gateway 키. 있으면 LLM·STT·TTS 를 모두 이 키 하나로 부른다 (없으면 아래 세 키로 각 서비스를 직접 부른다)
BAZE_URL = os.getenv("BAZE_BASE_URL", "https://api.baze.co/v1/gateway").rstrip("/")
missing = [] if BAZE_KEY else [k for k in ("OPENAI_API_KEY", "SONIOX_API_KEY", "GEMINI_API_KEY") if not os.getenv(k)]
if missing:
    raise SystemExit(f"API 키가 없습니다: {', '.join(missing)}\n→ agent_server/.env 파일(제출할 땐 맨 위 폴더의 .env)에 한 줄씩 적어 주세요. 예) BAZE_API_KEY=키값  (또는 OPENAI_API_KEY, SONIOX_API_KEY, GEMINI_API_KEY 세 개)")

# ── 설정 ────────────────────────────────────────────────────────────
LLM_MODEL = os.getenv("LLM_MODEL", "gpt-5.4-mini")
LLM_EFFORT = os.getenv("LLM_EFFORT")                      # 생각 강도 none / low / medium ... (비우면 모델 기본값=none)
SONIOX_KEY = BAZE_KEY or os.environ["SONIOX_API_KEY"]
SONIOX_URL = os.getenv("SONIOX_URL") or (f"wss://{BAZE_URL.split('://', 1)[1]}/soniox/transcribe-websocket" if BAZE_KEY else "wss://stt-rt.soniox.com/transcribe-websocket")
SONIOX_CONFIG = {"model": os.getenv("SONIOX_MODEL", "stt-rt-v6"), "audio_format": "pcm_s16le", "sample_rate": 16000, "num_channels": 1,
                 "language_hints": ["ko"]}               # 말 끝은 우리 눈치가 정하고 finalize 를 보낸다 (Soniox 자체 endpoint 감지는 끔)
STT_TIMEOUT_S = 5                                         # finalize 뒤 확정 글자를 기다리는 최대 시간
TTS_MODEL = os.getenv("TTS_MODEL", "gemini-3.8-flash-tts")   # 운영진 공지: Gemini 2.5 Flash TTS 가 아니라 Gemini 3.8 Flash 모델을 쓴다 (게이트웨이 모델 목록에도 이 이름으로 있음)
TTS_VOICE = os.getenv("TTS_VOICE", "Kore")
TTS_STYLE = os.getenv("TTS_STYLE", "" if TTS_MODEL.startswith("gemini-3.8") else "Say: ")   # 읽을 글 앞에 붙이는 지시. 예전 모델(2.5·flash-lite)은 없으면 글을 '질문'으로 착각해 실패한다. 3.8 은 지시어를 소리 내어 읽어 버려서(직접 확인) 붙이지 않는다
TTS_KO_STYLE = "다음 문장을 그대로 읽어 주세요: "                # 위 지시가 실패했을 때 쓰는 한국어 지시 (직접 시험했을 때 성공한 문구)
TTS_RETRY_STYLE = "Read the following text aloud exactly as written: "   # 첫 시도가 실패했을 때 쓰는 더 분명한 지시
TTS_STREAM = os.getenv("TTS_STREAM", "0" if BAZE_KEY else "1") == "1"            # 1 이면 음성을 조각으로 받는 대로 바로 내보낸다 (첫 소리가 빨라짐). 안 되면 자동으로 통째 방식으로 돌아간다
TTS_PER_SENTENCE = os.getenv("TTS_PER_SENTENCE", "1" if BAZE_KEY else "0") == "1"   # 0 이면 답 전체를 한 번에 읽힌다 (요청 수가 줄어 한도에 유리)
TTS_RATE = 24000                                          # Gemini TTS 출력은 24kHz 16-bit mono. 목소리가 이상하면 여기부터 확인
STYLES = ("",) if BAZE_KEY or TTS_MODEL.startswith("gemini-3.8") else (TTS_STYLE, TTS_KO_STYLE, TTS_RETRY_STYLE)   # 게이트웨이와 3.8 모델은 읽을 글 앞에 지시를 붙이면 그 지시까지 읽어 버려서 붙이지 않는다
QUOTA_RETRY_MAX_S = 8                                     # Gemini 분당 호출 한도(429)에 걸렸을 때, '이 시간 뒤에 다시'가 이 초 이하면 기다렸다가 한 번 더 시도한다
PHRASE_DIR = Path(__file__).with_name("phrases")          # 고정 문구 음성을 저장해 두는 폴더. 있으면 서버를 켤 때 TTS 를 부르지 않는다 (제출물에 함께 넣을 것)

RATE, FRAME = 16000, 640            # 대회 규격: 16kHz 16-bit mono, 20ms = 640바이트
SPEECH_RMS = int(os.getenv("SPEECH_RMS", 500))            # 볼륨이 이보다 크면 말소리
END_SILENCE_MS = int(os.getenv("END_SILENCE_MS", 1000))   # 문장부호 없는 말: 이만큼 조용하면 말이 끝난 것 (0.8초는 연습 음성의 긴 문장 중간 쉼에 걸림. 첫마디가 바로 나가서 1.0초여도 첫 소리 1.5초 안)
PUNCT_SILENCE_MS = int(os.getenv("PUNCT_SILENCE_MS", 1000))          # 문장부호로 끝난 말 (측정)
LIVE_PUNCT_SILENCE_MS = int(os.getenv("LIVE_PUNCT_SILENCE_MS", 800))  # 문장부호로 끝난 말 (시험기·시연)
NOUN_SILENCE_MS = int(os.getenv("NOUN_SILENCE_MS", 2000))            # 서술어 없는 명사로 끝난 짧은 말 (측정)
LIVE_NOUN_SILENCE_MS = int(os.getenv("LIVE_NOUN_SILENCE_MS", 1000))
COMMA_SILENCE_MS = int(os.getenv("COMMA_SILENCE_MS", 2400))          # 쉼표·말줄임으로 끝난 말
PREAMBLE_SILENCE_MS = int(os.getenv("PREAMBLE_SILENCE_MS", 1600))    # 인사·머뭇거림뿐인 말
SENT_END = re.compile(r"[.?!。？！]\s*$")
VERB_END = re.compile(r"(요|다|까|죠|니다|세요|어|아|해|네|군|지|래|나|가|고|서|데|라|자|게|걸|냐|니)$")
PREAMBLE_WORDS = {"안녕하세요", "안녕하십니까", "여보세요", "저기요", "저기", "네", "예", "넵", "음", "어", "그", "아", "저", "수고하십니다", "상담원님"}
CORRECTING_ONLY = re.compile(r"^(?:(?:음|어|아|저|네|그)[\s,.…]*)?(?:잘못\s*(?:말|얘기|이야기)\S*|아니(?:\s*아니)*|아냐|잠깐(?:만요|만)?|잠시만(?:요)?|"
                             r"다시\s*(?:말할게요|말씀드릴게요|말할게|불러\s*드릴게요)|그게\s*아니(?:라|고)?)[\s,.…!?~]*$")
MIN_SPEECH_MS = 200                 # 이보다 짧은 소리(잡음)는 말로 치지 않는다
LONG_SILENCE_MS = int(os.getenv("LONG_SILENCE_MS", 1800))  # 말이 '~는데', '~하고' 처럼 이어질 모양으로 멈췄을 땐 이만큼 더 기다린다 (END_SILENCE_MS 와 같게 두면 끔)
CONTINUING = ("는데", "은데", "인데", "하고", "했고", "였고", "이고", "있고", "없고", "싶고", "해서", "어서", "아서", "라서",
              "하면", "니까", "지만", "으며", "하며", "그리고", "그런데", "근데", "그래서")
CORRECT_SILENCE_MS = int(os.getenv("CORRECT_SILENCE_MS", 2500))   # '잘못 얘기했는데'·'아니'·'잠깐' 처럼 말을 고치려는 짧은 시작 말 뒤에는 생각하느라 오래 쉬니 이만큼 기다린다
CORRECTING = re.compile(r"^(?:[아어음저네그][,.\s]*)?(?:잘못|아니(?=[\s,])|아냐|잠깐|잠시|다시|그게\s*아니)")   # 말을 고치는 시작 말 (대답인 "아니요"는 제외. 받아쓰기가 한 박자 늦어 "아니"만 보일 수 있어, 뒤에 띄어쓰기가 올 때만 센다)
DIGITS_TAIL = re.compile(r"([0-9][0-9\s\-.]*)$")          # 받아쓴 글이 숫자로 끝나는 부분 (번호를 말하다 쉬는 중인지 볼 때 쓴다)
MAX_LISTEN_MS = 30_000              # 아무리 길어도 여기서 듣기를 멈춘다
# ── 말하는 속도에 맞춘 말 끝 대기 (pace.md): 느리게 말하는 사람은 문장 사이 쉼도 길어서 기본 대기(END_SILENCE_MS)만으로는 말 도중에 끼어든다.
# 그래서 말이 느리거나 쉼이 긴 사람에게만 기본 대기를 조금 늘린다. 줄이지는 않는다(줄이면 끼어들어 속도 0점). 이미 1.8~2.5초 기다리는 규칙에는 적용하지 않는다.
PACE_ADAPT = os.getenv("PACE_ADAPT", "live")              # off 끔 / live 시험기·시연(session.start 에 debug 가 있을 때)만 / all 측정 포함 (측정 음성은 합성이라 속도와 쉼이 무관하니 all 은 pace.md 9절을 확인한 뒤에)
PACE_REF_CPS = float(os.getenv("PACE_REF_CPS", 5.3))      # 기준 발음 속도(글자/초). 이보다 느리면 대기를 늘린다. 합성 음성의 중앙값이라 사람 목소리로 다시 맞출 것 (pace.md 7절)
PACE_MAX_FACTOR = float(os.getenv("PACE_MAX_FACTOR", 1.2))   # 기본 대기를 늘리는 최대 배율 (1.0 이면 늘리지 않음)
PACE_PAUSE_MARGIN, PACE_MIN_PAUSE_MS = 1.25, 250          # 본인이 말 중간에 쉰 시간의 1.25배는 기다린다 / 250ms 미만의 끊김은 쉼으로 치지 않는다
PACE_EXTRA_LIVE_MS, PACE_EXTRA_MEASURE_MS = 600, 300      # 본인 쉼 때문에 기본 대기에서 늘어나는 최대 폭(시연 / 측정). 속도 배율 상한(1.2배)과 따로 둔다: 같이 묶으면 0.75초 쉬던 사람이 다음 턴에도 또 잘린다 (tts_speed_and_pace_handoff.md)
PACE_LEARN_MAX_MS = int(os.getenv("PACE_LEARN_MAX_MS", 2000))   # 말 끝으로 확정한 뒤 고객이 이어 말하기까지 이 시간 안이면 '끼어들기에서 배운 쉼'으로 기억한다 (이보다 늦은 건 답하는 말을 끊은 것이라 배우지 않는다)
PACE_PLAY = os.getenv("PACE_PLAY", "1") == "1"            # 느리게 말하는 사람에게는 상담원 음성도 음높이를 유지한 채 느리게 들려준다 (시험기·시연만, 0.85~1.0배)
PACE_PLAY_MIN_SPEED = float(os.getenv("PACE_PLAY_MIN_SPEED", 0.85))   # 가장 느린 재생 속도. 이보다 낮으면 변환 울림이 들릴 수 있어 직접 들어 보고 낮출 것
PACE_PLAY_GAP_MS = 150                                    # 느린 재생일 때 문장 사이에 넣는 쉼 (원래 쉼 ÷ 재생 속도)
PACE_PROFILES = {}                                        # 통화 번호(call_id) → Profile. 우리 서버는 말 한 번마다 연결이 새로 열려서, 같은 통화를 알아보는 값(call_id)이 있을 때만 기록한다. 채점기는 call_id 를 안 보내므로 구조적으로 발동하지 않는다 (서버를 다시 켜면 비워짐)
PLAY_SPEED = contextvars.ContextVar("play_speed", default=1.0)   # 이번 답의 재생 속도 (답을 시작할 때 정하고, 그 답을 말하는 작업들이 읽는다)
LEAD_RMS, LEAD_KEEP_MS = 300, 40    # 말하기 음성 앞 무음을 자를 때: 이 볼륨부터 소리로 보고, 그 40ms 전부터 남긴다
DEADLINE_S = float(os.getenv("DEADLINE_S", 18))           # 말 끝 이후 이 안에 마무리 (25초 안에 turn.end 필요)
MAX_TOOL_ROUNDS = 3                 # LLM이 함수를 연달아 부를 수 있는 최대 횟수
VERBOSE = os.getenv("VERBOSE", "1") == "1"                # 측정 때는 0으로 (고객 발화 내용을 출력하지 않음)

NOT_HEARD = "잘 못 들었습니다. 다시 한번 말씀해 주시겠어요?"
SORRY = "죄송합니다. 잠시 문제가 생겨 바로 답변드리기 어렵습니다."
# 자주 쓰는 말은 서버를 켤 때 미리 음성으로 만들어 둔다 (Gemini 가 목소리를 만드는 데 몇 초가 걸려서).
# LLM 이 아래 문장을 그대로 말하면 만드는 시간 없이 바로 나간다. 매뉴얼에 정해진 문구가 있으면 여기를 고칠 것.
GREETING_REPLY = "네 안녕하세요, 무엇을 도와드릴까요?"          # 손님이 처음에 인사만 했을 때
CLOSING_REPLY = "네 감사합니다. 좋은 하루 보내시길 바랍니다."      # 상담이 끝나 갈 때 손님이 감사·끝인사만 했을 때
PHRASES = {
    "손님이 처음에 인사만 했을 때 ('안녕하세요')": GREETING_REPLY,
    "손님이 상담 끝에 감사·끝인사만 했을 때 ('감사합니다', '수고하세요')": CLOSING_REPLY,
    "손님이 방금 한 말을 고치려 하거나('잘못 얘기했는데', '아니 그게 아니라') 말을 이어 가려는 듯 아직 덜 한 것 같을 때": "네, 말씀하세요.",
    "주문번호나 전화번호가 필요한데 [통화 정보]에도 이전 대화에도 없을 때": "확인 도와드릴게요. 주문번호나 주문하신 분 전화번호를 말씀해 주시겠어요?",
    "손님이 몸이 불편하다·부작용이 있다고 말해 상담원 연결이 필요할 때 (브랜드 정보에 해당 안내 문구가 있으면 그 문구를 따른다)": "걱정되시겠어요. 바로 상담원 연결을 도와드리겠습니다.",
    "매뉴얼과 조회 결과에 없는 내용이라 답할 수 없을 때 (조회 결과가 '없음'인 경우는 해당하지 않으며, 그때는 '주문번호 ○○로는 조회되지 않습니다. 다시 확인해 주시겠어요?'처럼 짧게 알린다)":
        "죄송하지만 그 부분은 제가 정확히 안내드리기 어렵습니다. 상담원 연결을 도와드리겠습니다.",
}
# 함수를 부를 때(조회에 1~2초 걸림) 서버가 먼저 말하는 한마디. 함수를 부르는 답에만, 이 중 하나를 돌려 가며 쓴다 (모든 답에 붙이면 어색하다).
FILLERS = ["네, 확인해 보겠습니다.", "네, 바로 확인해 볼게요.", "잠시만요, 확인해 드리겠습니다."]

# ── 유형별 매크로: 받아쓰기가 끝나면 LLM 을 기다리지 않고 키워드로 유형(case)을 바로 정해, 미리 만들어 둔 '먼저 하는 말'을 내보낸다.
# 그 말이 나가는 2~3초 동안 LLM 답과 음성이 만들어지고, 끝나면 본 답이 이어지며, 끝맺음 말이 붙는다. 먼저 하는 말·끝맺음 말은 warm_up 이 미리 음성으로 만들어 둔다.
# 먼저 하는 말은 결과를 약속하지 않는 '이해했다는 표시' 한 문장(질문·숫자 없음)이라, LLM 이 어떤 답을 하든 어색하지 않다.
MACRO = os.getenv("MACRO", "1") == "1"                    # 0 이면 매크로를 끈다 (전후 비교용)
OPENER = os.getenv("OPENER", "1") == "1"                  # 0 이면 일반 첫마디를 끈다 (아래 pick_opener)
SKIP_RE = re.compile(r"상담원|통화|전화\s*(주|해|요청)|불편|부작용|아프|아파|알레르기|두드러기|복용\s*중|먹고\s*있|약을|치료|증상|일째|변을|변비|설사|두통|어지|구토|통증"
                     r"|잘못|오배송|누락|빠졌|파손|불량|분실|못\s*받|안\s*왔|왜.*왔|화나|짜증|사기|항의|신고|광고")   # 사람 상담원에게 넘기거나 조심해야 할 말은 매크로 없이 LLM 이 판단한다
CASES = {   # 유형: (키워드, 번호가 있어야 쓰는지, 먼저 하는 말 변형, 끝맺음 말을 붙이는지). 키워드가 정확히 한 유형에만 걸릴 때만 쓴다
    "반품현황": (re.compile(r"(반품|회수|수거|환불).{0,15}(진행|현황|상태|어디|언제|아직|안\s*[돼되]|확인\s*(이\s*)?안)"), True,
              ["반품 진행 상황 문의 주셨네요, 바로 확인해 볼게요.", "네, 반품 접수 내역부터 먼저 확인해 볼게요."], True),
    "접수": (re.compile(r"(취소|반품|교환|환불).{0,6}(신청|접수|해\s*주|처리|부탁|하고\s*싶|할게|할래|원해|요청|용청)"), True,
           ["요청하신 내용 확인했어요, 주문 내역부터 먼저 살펴볼게요.", "네, 접수 도와드릴게요, 주문 내역부터 확인해 볼게요."], False),
    "주문조회": (re.compile(r"(배송|택배|송장|도착|출고|입금).{0,12}(언제|어디|상태|조회|확인|됐|되나|왔|가나|오나|안\s*[돼되]|문의)|배송\s*문의|입금\s*확인|주문\s*(내역|상태|확인|조회)"), True,
              ["주문 내역 문의 주셨네요, 바로 확인해 볼게요.", "네, 주문 상태부터 차근차근 확인해 볼게요."], True),
    "상품": (re.compile(r"가격|얼마|섭취|먹는\s*(법|시간)|복용\s*방법|복용법|성분|함량|효과|효능|추천|몇\s*일\s*분|며칠\s*분|유통기한|몇\s*포"), False,
           ["상품 문의 주셨네요, 정확한 정보 바로 찾아볼게요.", "네, 제품 정보 확인해서 안내해 드릴게요."], True),
    "정책": (re.compile(r"쿠폰|배송비|결제\s*방법|카드|무통장|적립|포인트|(환불|교환|반품)\s*(기간|비용|가능|방법)|운영\s*시간|영업"), False,
           ["문의 주신 내용 확인해서 안내해 드릴게요.", "네, 해당 내용 바로 안내해 드릴게요."], True),
}
CLOSINGS = ["더 궁금하신 점 있으시면 편하게 말씀해 주세요.", "다른 도움이 필요하시면 언제든 말씀해 주세요."]
NO_CLOSING_RE = re.compile(r"\?|주세요|주시겠|말씀해|알려|여쭤|연결|전달|죄송")    # 본 답이 되묻거나 넘기는 말이면 끝맺음 말을 붙이지 않는다


def load_manual():
    """data/brand 의 말투 규칙·상담 시나리오·브랜드 정보를 매뉴얼 글 하나로 합친다 (서버를 켤 때 한 번)."""
    brand = ROOT / "data" / "brand"
    try:
        style = json.loads((brand / "style.json").read_text(encoding="utf-8"))
        info = json.loads((brand / "information.json").read_text(encoding="utf-8"))
        scenarios = (brand / "scenarios.md").read_text(encoding="utf-8")
    except (OSError, ValueError) as e:
        print(f"[주의] 브랜드 자료(data/brand)를 못 읽었습니다. 매뉴얼 없이 시작합니다: {e!r}", flush=True)
        return ""
    facts = "\n\n".join(f"[{x['tag'].strip('[]')}]\n{x['content'].strip()}" for x in info)
    return f"\n\n# [말투 규칙]\n{style['style']}\n\n{scenarios}\n\n# [브랜드 정보]\n{facts}"


PROMPT = Path(__file__).with_name("prompt.md").read_text(encoding="utf-8") + load_manual()   # 규칙은 prompt.md, 브랜드 자료는 data/brand
ai = AsyncOpenAI(timeout=10, max_retries=1, **({"base_url": BAZE_URL, "api_key": BAZE_KEY} if BAZE_KEY else {}))   # LLM 전용. 키는 BAZE_API_KEY (게이트웨이) 또는 OPENAI_API_KEY
gemini = None if BAZE_KEY else genai.Client(http_options=types.HttpOptions(      # 게이트웨이를 쓰면 Gemini SDK 는 안 쓴다. 직접 부를 땐 키는 GEMINI_API_KEY, preview 모델은 v1beta 로 부른다
    base_url=os.getenv("GEMINI_BASE_URL"), api_version="v1beta", timeout=10_000))
http = httpx.AsyncClient(timeout=10)
FIXED = {}                          # 고정 문구(사과·채움말 등) → 미리 만들어 둔 음성. warm_up() 이 채운다

# 서버를 켠 뒤 누적 사용량 (크레딧 관리용). 단가는 대략적인 추정이니 공식 가격 페이지로 확인할 것
USAGE = {"calls": 0, "llm_in": 0, "llm_out": 0, "stt_sec": 0.0, "tts_sec": 0.0}
PRICE = {"llm_in": 0.75e-6, "llm_out": 4.5e-6, "stt_sec": 0.12 / 3600, "tts_sec": 25 * 10e-6}   # 달러 (TTS 는 음성 1초=25토큰 가정)


def usage_line():
    cost = sum(USAGE[k] * PRICE[k] for k in PRICE)
    return (f"[사용량] 누적 {USAGE['calls']}통 | LLM {USAGE['llm_in']}/{USAGE['llm_out']} 토큰 | "
            f"STT {USAGE['stt_sec']:.0f}초 | TTS {USAGE['tts_sec']:.0f}초 | 추정 ${cost:.3f}")


def count_llm(resp):
    if resp.usage:
        USAGE["llm_in"] += resp.usage.prompt_tokens
        USAGE["llm_out"] += resp.usage.completion_tokens


class Lap:
    """단계별로 걸린 시간을 재서, 통화가 끝나면 한 줄로 출력한다."""
    def __init__(self, case_id):
        self.case_id, self.t, self.marks = case_id, time.perf_counter(), []

    def __call__(self, label):
        now = time.perf_counter()
        self.marks.append((label, now - self.t))
        self.t = now

    def report(self):
        steps = " | ".join(f"{label} {sec:.2f}s" for label, sec in self.marks)
        total = sum(sec for _, sec in self.marks)
        print(f"[{self.case_id}] 말 끝 감지 뒤: {steps} | 합계 {total:.2f}s", flush=True)


# ── 1. 받아쓰기: Soniox 실시간 (통화 한 통) ──────────────────────────
class Stt:
    """고객 목소리를 말하는 동안 Soniox 로 흘려보내고, 말이 끝나면 finalize 로 확정된 글자를 받는다."""

    def __init__(self):
        self.queue, self.text, self.error = asyncio.Queue(), "", None     # queue: bytes=목소리, str=제어 메시지
        self.ok, self.done = False, asyncio.Event()                      # ok: <fin> 을 받음 / done: 결과가 났거나 실패함
        self.partial = ""                                                # 아직 확정 전인 글자 (말 끝 판단 보조용)
        self.t0 = time.time()
        self.task = asyncio.create_task(self._run())

    def feed(self, pcm):
        self.queue.put_nowait(pcm)

    def looks_unfinished(self):
        """지금까지 받아쓴 글이 '~는데', '~하고' 처럼 말이 이어질 모양으로 끝났는가 (문장부호는 무시)."""
        tail = (self.text + self.partial).strip().rstrip(".,?!…~ ")
        return tail.endswith(CONTINUING)

    def rule(self, live=False):
        """(기다릴 시간 ms, 규칙 이름). 문장부호가 찍힌 말은 Jev 보다 이 규칙을 따른다 (검토 1·2-1절, voice_agent 실측).
        punct·plain 만 '빨리 확정하는 규칙'이라 Jev end·말 속도 조절의 대상이다."""
        raw = (self.text + self.partial).strip()
        tail = raw.rstrip(".,?!…~ ")
        last = [p for p in re.split(r"(?<=[.?!])\s+", raw) if p.strip()]
        if last and CORRECTING_ONLY.match(last[-1].strip()):     # 마지막 문장이 정정하는 말뿐 ("아니,", "잠깐만요.", "잘못 말했어요."). "아니 배송이 왜 안 와요?"는 해당 없음
            return CORRECT_SILENCE_MS, "correcting"
        if raw.endswith((",", "，", "…", "-")):                   # 쉼표·말줄임으로 끝남: 이어 말하는 중
            return COMMA_SILENCE_MS, "comma"
        m = DIGITS_TAIL.search(tail)
        d = re.sub(r"\D", "", m.group(1)) if m else ""
        if d and (len(d) < 10 or (d.startswith("20") and len(d) < 15)):   # 번호를 말하다 쉬는 중 (전화 11자리, 주문 15자리)
            return LONG_SILENCE_MS, "partial_number"
        if tail.endswith(CONTINUING):
            return LONG_SILENCE_MS, "connective"
        words = [w for w in re.split(r"[\s.,?!…~]+", tail) if w]
        if words and all(w in PREAMBLE_WORDS for w in words):     # 인사·머뭇거림뿐: 본론이 이어지는 경우가 많다
            return PREAMBLE_SILENCE_MS, "preamble"
        if SENT_END.search(raw):
            if not raw.endswith("?") and len(re.sub(r"\D", "", tail)) < 7 and len(re.sub(r"\s", "", tail)) <= 14 and not VERB_END.search(tail):
                return (LIVE_NOUN_SILENCE_MS if live else NOUN_SILENCE_MS), "noun"   # 서술어 없는 명사로 끝난 짧은 말 ('대장영양제.'): 이어 말할 수 있다
            return (LIVE_PUNCT_SILENCE_MS if live else PUNCT_SILENCE_MS), "punct"
        return END_SILENCE_MS, "plain"

    def wait_ms(self, live=False):
        """말이 멈춘 뒤 '끝났다'고 보기까지 기다릴 시간(ms)."""
        return self.rule(live)[0]

    async def finish(self):
        self.queue.put_nowait(json.dumps({"type": "finalize"}))
        await asyncio.wait_for(self.done.wait(), STT_TIMEOUT_S)
        if not self.ok:
            raise RuntimeError(f"Soniox 받아쓰기 실패: {self.error or '연결이 끊김'}")
        return self.text.strip()

    async def close(self):                                   # 스트림 길이만큼 과금되니 통화가 끝나면 닫는다
        USAGE["stt_sec"] += time.time() - self.t0
        self.task.cancel()
        with suppress(asyncio.CancelledError):
            await self.task

    async def _run(self):
        try:
            async with websockets.connect(SONIOX_URL, additional_headers={"Authorization": f"Bearer {SONIOX_KEY}"}) as ws:
                await ws.send(json.dumps(SONIOX_CONFIG))
                sender = asyncio.create_task(self._send_loop(ws))
                try:
                    await self._receive_loop(ws)
                except asyncio.CancelledError:                # 통화가 끝나 닫는 경우: 정상 종료 코드로 연결을 닫는다
                    await ws.close()
                    raise
                finally:
                    sender.cancel()
        except Exception as e:
            self.error = self.error or repr(e)
        finally:
            self.done.set()

    async def _send_loop(self, ws):
        while True:
            await ws.send(await self.queue.get())

    async def _receive_loop(self, ws):
        async for message in ws:
            reply = json.loads(message)
            if reply.get("error_code"):
                self.error = f"{reply['error_code']} {reply.get('error_message', '')}"
                self.done.set()
                return
            self.partial = "".join(t["text"] for t in reply.get("tokens", []) if not t.get("is_final"))
            for token in reply.get("tokens", []):
                if not token.get("is_final"):
                    continue                                 # 임시 글자는 바뀔 수 있으니 확정된 것만 모은다
                if token["text"] == "<fin>":
                    self.ok = True
                    self.done.set()
                    return
                if token["text"] != "<end>":
                    self.text += token["text"]


# ── 2. 듣기: 말이 끝날 때까지 목소리를 Soniox 로 넘기며 기다린다 ──────
def rms(pcm):
    a = np.frombuffer(pcm, dtype=np.int16).astype(np.float32)
    return float(np.sqrt(np.mean(a * a))) if len(a) else 0.0


def turn_rate(text, voiced_ms):
    """발음 속도(글자/초) = 받아쓴 한글·영문 글자 수 ÷ 소리가 난 시간(쉬는 시간 제외). 숫자는 읽는 길이가 사람마다 달라 뺀다.
    근거가 부족하면(8자 미만, 소리 1.5초 미만, 번호 위주의 말) None."""
    chars = len(re.findall(r"[가-힣A-Za-z]", text))
    digits = sum(c.isdigit() for c in text)
    if chars < 8 or voiced_ms < 1500 or digits * 2 > chars:
        return None
    return chars / (voiced_ms / 1000)


def pace_wait(base_ms, rate, own_pause_ms=0, live=True):
    """기본 대기(base_ms)를 말이 느린 사람에게만 늘린다. 줄이지는 않는다.
    - 속도: 기준 속도 ÷ 이 사람 속도를 1.0~PACE_MAX_FACTOR 배로 곱한다.
    - 본인 쉼: 이 사람이 말 중간에 쉰 시간의 1.25배는 기다리되, 늘어나는 폭은 모드별 상한(시연 +600ms, 측정 +300ms)까지."""
    factor = min(max(PACE_REF_CPS / rate, 1.0), PACE_MAX_FACTOR) if rate else 1.0
    wait = base_ms * factor
    if own_pause_ms > 0:
        wait = max(wait, min(own_pause_ms * PACE_PAUSE_MARGIN, base_ms + (PACE_EXTRA_LIVE_MS if live else PACE_EXTRA_MEASURE_MS)))
    return max(wait, base_ms)


def pace_enabled(start):
    """이 통화에서 말 속도 대기를 쓰는가. 기본(live)은 시험기·시연(debug)만이라, debug 를 켜지 않는 측정기에는 영향이 없다."""
    return PACE_ADAPT == "all" or (PACE_ADAPT == "live" and bool(start.get("debug")))


@dataclass
class Profile:
    """통화 한 건 동안 쌓는 말하는 사람의 버릇 (말이 바뀌어도 유지)."""
    rate: float | None = None                                        # 지난 말들의 발음 속도 (지수 평균)
    pauses: deque = field(default_factory=lambda: deque(maxlen=5))   # 끝난 줄 알았는데 이어 말한 쉼 (ms), 최근 5개

    def note_rate(self, rate):
        """이번 말의 발음 속도를 섞는다 (지수 평균 0.6/0.4). 1~12 글자/초 밖의 값은 잘못 잰 것이라 버린다."""
        if rate and 1 <= rate <= 12:
            self.rate = rate if self.rate is None else 0.6 * self.rate + 0.4 * rate

    def note_pause(self, ms):
        if 200 <= ms <= 4000:
            self.pauses.append(float(ms))

    def pause_hint(self):
        return max(self.pauses, default=0.0)

    def ready(self):
        """이전 말에서 잰 기록이 있는가. 첫 말은 기록만 하고 대기를 늘리지 않는다."""
        return self.rate is not None or bool(self.pauses)


def profile_for(start):
    """이 통화의 Profile. 통화를 알아보는 call_id 가 없으면 None (채점기는 안 보내므로 기록도 적용도 하지 않는다)."""
    key = str(start.get("call_id") or "")
    if not key:
        return None
    if key not in PACE_PROFILES and len(PACE_PROFILES) >= 200:       # 메모리가 무한히 늘지 않게 가장 오래된 것부터 지운다
        PACE_PROFILES.pop(next(iter(PACE_PROFILES)))
    return PACE_PROFILES.setdefault(key, Profile())


def play_speed(profile, live=True):
    """상담원 음성 재생 속도 (1.0 = 원래 속도). 느리게 말하는 사람에게만 느리게, 시험기·시연에서 이전 말의 속도 기록이 있을 때만.
    0.05 단위로 맞춰 같은 문장의 변환 결과를 다시 쓴다."""
    if not (live and PACE_PLAY and PACE_ADAPT != "off") or profile is None or not profile.rate:
        return 1.0
    sp = min(1.0, max(PACE_PLAY_MIN_SPEED, profile.rate / PACE_REF_CPS))
    return round(round(sp / 0.05) * 0.05, 2)


async def listen(ws, stt, pace_on=False, profile=None, judge=None, router=None, live=True):
    """말이 끝날 때까지 목소리를 받아 받아쓰기로 넘긴다. judge(Jev 말 끝 판정)·router(Jev 업무 분류)가 있으면 조용한 동안 그쪽에 묻는다.
    pace_on 이고 profile(통화 기록)에 이전 말의 기록이 있으면(ready) 말이 느린 사람의 기본 대기를 늘린다. live: 시연(True)/측정(False)에 따라 본인 쉼 상한이 다르다.
    돌려주는 값: 소리가 난 시간(voiced_ms), 말 끝으로 확정한 때의 무음(silent_ms), 말 속도 조절 내용(pace), Jev 판정 요약(eot)."""
    voiced_ms = silent_ms = total_ms = 0
    pauses, info = [], None                                  # pauses: 말 중간에 쉰 시간들 (250ms 이상)
    try:
        async for m in ws:
            if not isinstance(m, bytes):
                continue
            stt.feed(m)                                      # 받아쓰기는 고객이 말하는 동안 이미 진행된다
            ms = len(m) * 1000 // (RATE * 2)
            total_ms += ms
            if rms(m) >= SPEECH_RMS:
                if silent_ms >= PACE_MIN_PAUSE_MS and voiced_ms >= MIN_SPEECH_MS:   # 쉬었다가 다시 말하기 시작함
                    pauses.append(silent_ms)
                if judge and silent_ms:
                    judge.resume()                           # 다시 말했으니 이전 쉼에서 받은 판정은 버린다
                voiced_ms, silent_ms = voiced_ms + ms, 0
            elif voiced_ms >= MIN_SPEECH_MS:                 # 말이 시작된 뒤의 조용한 시간만 센다
                silent_ms += ms
            text, verdict = stt.text + stt.partial, None
            if silent_ms:
                if judge:
                    judge.tick(text, silent_ms)              # Jev 에게 "지금 답해도 되나(end)/더 기다려야 하나(hold)" 묻는다 (답은 비동기로 온다)
                    verdict = judge.verdict(text, silent_ms)
                    if verdict == "hold" and hold_ignored(text):
                        verdict = None                       # 끝난 말(인사·완결 번호·문장부호로 끝남)을 hold 로 본 것은 따르지 않는다: 2.5초를 괜히 기다리게 된다
                if router and silent_ms >= jev.CASE_EARLY_MS:
                    router.speculate(text)                   # 말 끝 판정을 기다리는 동안 업무를 미리 분류해 둔다
            info = None
            wait_ms, rule = stt.rule(live)
            if rule == "punct":
                verdict = None                               # 문장부호로 끝난 말은 Jev 보다 규칙 (Jev 가 끝난 질문을 hold 로 봐 첫 소리가 늦어짐: voice_agent 실측)
            if silent_ms >= min(wait_ms, jev.EOT_END_MS if verdict == "end" else wait_ms):
                if rule in ("punct", "plain"):               # 이어질 말투·번호·말 고치기 같은 규칙이 없는 보통 말
                    if verdict == "end":
                        wait_ms = jev.EOT_END_MS             # Jev 가 끝났다고 보면 EOT_END_MS 에 끊는다 (규칙이 '더 기다려라' 하면 규칙을 따른다: 끊는 쪽 실수는 속도 0점이라)
                    elif pace_on and profile is not None:    # 빨리 확정하는 기본 규칙에만 말 속도 조절을 적용한다
                        rate = turn_rate(text, voiced_ms) or profile.rate   # 이번 말로 못 재면 통화 누적값. (받아쓰기 글은 소리보다 늦게 와서 말하는 도중엔 속도가 실제보다 느리게 나오지만, 상한 배율이 막아 준다)
                        own = max(max(pauses, default=0), profile.pause_hint())    # 이번 말 중간의 쉼, 지난번에 끼어들며 배운 쉼 중 큰 쪽
                        ready = profile.ready()
                        if ready:                            # 첫 말은 기록만 한다 (채점은 질문 하나뿐이라 이전 기록이 없어 구조적으로 발동하지 않는다)
                            wait_ms = pace_wait(wait_ms, rate, own, live)
                        info = {"cps": round(rate, 1) if rate else None, "own_pause_ms": round(own), "base_ms": END_SILENCE_MS, "thr_ms": round(wait_ms), "ready": ready}
            if verdict == "hold":
                wait_ms = max(wait_ms, jev.EOT_HOLD_MS)      # Jev 가 더 이어질 말이라고 보면 2.5초까지 기다린다
            if silent_ms >= wait_ms or total_ms >= MAX_LISTEN_MS:
                break
    finally:
        if judge:
            judge.close()
    return {"voiced_ms": voiced_ms, "silent_ms": silent_ms, "pace": info, "eot": judge.summary() if judge else None}


async def drain(ws, profile=None, silent_ms=0):
    """말이 끝난 뒤에도 측정기는 무음을 계속 보낸다. 읽어서 버린다.
    profile 이 있으면(시험기·시연) 그 사이에 고객이 이어 말하는지도 본다: 말 끝으로 확정한 뒤 PACE_LEARN_MAX_MS 안에 다시 말하기 시작하면
    '서버가 너무 일찍 끊었다'는 뜻이라, 쉬었던 시간(silent_ms + 그 뒤 흐른 시간)을 이 사람의 쉼으로 기억해 다음 말의 대기에 반영한다."""
    elapsed = run = 0
    learned = profile is None
    with suppress(websockets.ConnectionClosed):
        async for m in ws:
            if learned or not isinstance(m, bytes):
                continue
            ms = len(m) * 1000 // (RATE * 2)
            elapsed += ms
            run = run + 1 if rms(m) >= SPEECH_RMS else 0
            if run >= 5:                                     # 소리가 100ms 이상 이어져야 말이 다시 시작된 것 (잡음 한 조각은 거른다)
                learned = True
                gap = silent_ms + elapsed - run * ms
                if gap <= PACE_LEARN_MAX_MS:
                    profile.note_pause(gap)
            elif elapsed + silent_ms > PACE_LEARN_MAX_MS:
                learned = True                               # 이 시간이 지난 뒤의 말은 답하는 말을 끊은 것이라 배우지 않는다


# ── 3. 생각: LLM + 모의 함수 서버 ───────────────────────────────────
async def load_tools(start):
    """함수 목록(GET /tools)은 OpenAI 도구 형식 그대로라서 LLM에 바로 넘긴다."""
    try:
        r = await http.get(f"{start['tools_url'].rstrip('/')}/tools")
        r.raise_for_status()
        return r.json()
    except (httpx.HTTPError, ValueError, KeyError, AttributeError) as e:
        print(f"[{start.get('case_id')}] 함수 목록을 못 가져옴: {e!r}", flush=True)
        return []


def check_args(name, args, tools, ctx=None):
    """문지기: 함수를 부르기 전에 검사한다. 문제가 있으면 이유(문자열)를, 괜찮으면 None을 돌려준다."""
    spec = next((t["function"] for t in tools if t["function"]["name"] == name), None)
    if spec is None:
        return f"정의에 없는 함수입니다: {name}"
    if not isinstance(args, dict):
        return "arguments는 JSON 객체여야 합니다"
    missing = [k for k in spec.get("parameters", {}).get("required", []) if not str(args.get(k) or "").strip()]
    if name == "get_order_info" and not (str(args.get("order_id") or "").strip() or str(args.get("buyer_cellphone") or "").strip()):
        missing = ["order_id 또는 buyer_cellphone"]               # 주문 조회는 필수 값이 없지만 번호 하나는 있어야 한다
    if missing:
        return f"필요한 값이 없습니다: {', '.join(missing)}. 고객에게 먼저 물어보세요."
    return check_numbers(name, args, ctx or {})


EXAMPLE_PHONES = {"01012345678", "01000000000", "01012341234", "01000001234", "01011112222", "01022223333", "01033334444", "01044445555", "01055556666"}
CANCELLED = ("취소완료", "입금전취소")


def _digits(v):
    return re.sub(r"\D", "", str(v)) if v not in (None, "") else ""


def valid_phone(v):
    d = _digits(v)
    return len(d) == 11 if d.startswith("010") else (9 <= len(d) <= 11 and d.startswith("0"))


def valid_order_id(v):
    """하루온 주문번호: YYYYMMDD + 7자리 (예: 20261101-0001234)."""
    d = _digits(v)
    return len(d) == 15 and d.startswith("20") and 1 <= int(d[4:6]) <= 12 and 1 <= int(d[6:8]) <= 31


def call_context(start, results=()):
    """번호 검사에 쓰는 근거: 발신자·이전 대화에 나온 번호, 조회된 주문번호, 취소된 주문번호."""
    ctx = {"phones": set(), "orders": set(), "cancelled": set()}
    caller = start.get("caller")
    for v in (caller.values() if isinstance(caller, dict) else [caller]):
        if valid_phone(v):
            ctx["phones"].add(_digits(v))
    for h in start.get("history") or []:
        if h.get("role") == "customer":
            ctx["phones"].update(_digits(m.group(0)) for m in PHONE_RE.finditer(h.get("text") or ""))
    items = [(h.get("name"), h.get("result")) for h in start.get("history") or [] if h.get("role") == "tool_result"] + list(results)
    lines = {}
    for name, res in items:
        data = res.get("data") if isinstance(res, dict) else None
        if name != "get_order_info" or not isinstance(data, dict):
            continue
        for o in data.get("orders") or []:
            no = _digits(o.get("주문번호"))
            if no:
                ctx["orders"].add(no)
                lines.setdefault(no, []).append(o.get("배송상태") in CANCELLED)
    ctx["cancelled"] = {no for no, flags in lines.items() if flags and all(flags)}
    return ctx


def check_numbers(name, args, ctx):
    """번호가 깨져 들린 채로 조회·접수하면 틀린 호출이 된다 (voice_agent toolflow.block_reason). 문제가 없으면 None."""
    for k in ("buyer_cellphone", "customer_phone_number", "phone_number"):
        d = _digits(args.get(k))
        if d and d in EXAMPLE_PHONES and d not in ctx.get("phones", ()):
            return "전화번호가 정확히 들리지 않았습니다(예시 번호). 호출하지 않았습니다. 휴대폰 번호 11자리를 한 자리씩 천천히 다시 여쭙니다."
    if name == "get_order_info":
        ph, oid = args.get("buyer_cellphone"), args.get("order_id")
        if not ((ph and valid_phone(ph)) or (oid and (valid_order_id(oid) or _digits(oid) in ctx.get("orders", ())))):
            return "번호가 정확히 들리지 않았습니다(전화번호 11자리 또는 주문번호 YYYYMMDD-7자리가 아님). 조회하지 않았습니다. 번호를 한 자리씩 천천히 다시 여쭙니다."
    if name in ("get_return_info", "request_callback"):
        k = "customer_phone_number" if name == "get_return_info" else "phone_number"
        if not valid_phone(args.get(k)):
            return "전화번호가 정확히 들리지 않았습니다. 호출하지 않았습니다. 휴대폰 번호 11자리를 한 자리씩 천천히 다시 여쭙니다."
    if name in ("cancel_order", "return_order", "exchange_order"):
        v = args.get("order_id" if name == "cancel_order" else "order_number")
        if not (valid_order_id(v) or _digits(v) in ctx.get("orders", ())):
            return "주문번호가 정확하지 않습니다. 접수하지 않았습니다. 주문 조회로 먼저 확인하거나 주문번호를 다시 여쭙니다."
        if name != "cancel_order" and _digits(v) in ctx.get("cancelled", ()):
            return "취소된 주문은 반품·교환 대상이 아닙니다. 접수하지 않았습니다. 이미 취소된 주문이라 반품·교환할 수 없다고 안내합니다."
    return None


def parse_call(call, tools, masker, ctx=None):
    """LLM 의 함수 호출을 해석(자리표시는 원래 번호로 되돌림)하고 문지기 검사를 한다. (인수, 문제) 를 돌려준다. 문제가 없으면 문제 자리는 None."""
    try:
        args = json.loads(masker.unmask(call.function.arguments or "{}"))
    except ValueError:
        return None, "arguments가 올바른 JSON이 아닙니다"
    return args, check_args(call.function.name, args, tools, ctx)


async def run_tool(start, name, args, problem):
    if problem:
        return {"error": problem}
    try:                                                  # X-Session-Id 가 없으면 호출하지 않은 것으로 채점된다
        r = await http.post(f"{start['tools_url'].rstrip('/')}/tools/{name}", json=args,
                            headers={"X-Session-Id": start["session_id"]})
        return r.json()
    except (httpx.HTTPError, ValueError) as e:
        return {"error": f"함수 호출 실패: {e!r}"}


MASKING = bool(BAZE_KEY)            # 게이트웨이는 LLM·TTS 입력의 전화번호를 [PHONE] 으로 가리거나 다른 번호로 바꿔 버린다 → 번호를 자리표시로 바꿔 보내고 되돌린다
PHONE_RE = re.compile(r"(?<!\d)01[016789][\s.\-]*\d{3,4}[\s.\-]*\d{4}(?!\d)")
ORDER_RE = re.compile(r"(?<!\d)20\d{6}-\d{7}(?:-\d{2})?(?!\d)")
KO_DIGITS = "공일이삼사오육칠팔구"


class Masker:
    """번호를 <전화1>, <주문1> 같은 자리표시로 바꿔 LLM 에 보내고(원래 번호는 table 에 기억), LLM 이 돌려준 자리표시를 원래 번호로 되돌린다. 통화 한 통마다 하나."""

    def __init__(self):
        self.table = {}                                       # {"<전화1>": "01049576966", "<주문1>": "20261001-0005987"}

    def _swap(self, kind, m):
        value = re.sub(r"\D", "", m.group(0)) if kind == "전화" else m.group(0)
        for key, v in self.table.items():                     # 같은 번호는 같은 자리표시
            if v == value:
                return key
        key = f"<{kind}{sum(k.startswith('<' + kind) for k in self.table) + 1}>"
        self.table[key] = value
        return key

    def mask(self, text):
        if not MASKING or not isinstance(text, str):
            return text
        return PHONE_RE.sub(lambda m: self._swap("전화", m), ORDER_RE.sub(lambda m: self._swap("주문", m), text))

    def unmask(self, text):
        for key, value in self.table.items():
            text = text.replace(key, value)
        return text


def mask_messages(masker, messages):
    """LLM 에 보낼 메시지(글, 함수 인수)에서 번호를 자리표시로 바꾼 복사본을 돌려준다."""
    out = []
    for m in messages:
        m = {**m, "content": masker.mask(m.get("content"))}
        if m.get("tool_calls"):
            m["tool_calls"] = [{**c, "function": {**c["function"], "arguments": masker.mask(c["function"]["arguments"])}} for c in m["tool_calls"]]
        out.append(m)
    return out


def for_tts(text):
    """게이트웨이가 TTS 입력의 번호를 다른 번호로 바꿔 읽어 버려서, 번호는 한 글자씩 띄운 한글 숫자로 바꿔 보낸다.
    예) 010-4957-6966 → 공 일 공, 사 구 오 칠, 육 구 육 육  (붙여 쓴 한글 숫자는 TTS 가 자주 실패한다)"""
    if not MASKING:
        return text
    spell = lambda group: " ".join(KO_DIGITS[int(c)] for c in group)
    phone = lambda m: ", ".join(spell(g) for g in (lambda d: (d[:3], d[3:-4], d[-4:]))(re.sub(r"\D", "", m.group(0))))
    return PHONE_RE.sub(phone, ORDER_RE.sub(lambda m: ", ".join(spell(g) for g in m.group(0).split("-")), text))


def clean(text):
    """상담 기록에 섞인 링크·이모티콘은 전화에서 읽을 수 없으니 지운다."""
    text = re.sub(r"https?://\S+", "", str(text or ""))
    return re.sub(r"[\U00010000-\U0010ffff☀-➿️]", "", text).strip()


def history_messages(history):
    """session.start 의 history(손님·상담원 말, 함수 호출·결과)를 LLM 메시지로 바꾼다. 함수는 OpenAI 의 tool_calls / tool 형식으로 옮긴다."""
    out, waiting, prev = [], {}, None                         # waiting: 아직 결과가 안 온 호출 {id: 함수 이름}
    for i, h in enumerate([*(history or []), {}]):            # 맨 끝의 빈 항목은 '끝'을 알려 남은 호출을 닫는 용도
        role, name = h.get("role"), h.get("name")
        if role not in ("tool_call", "tool_result"):
            for cid in waiting:                               # 결과 없이 끝난 호출은 API 가 거절하니 빈 결과로 닫는다
                out.append({"role": "tool", "tool_call_id": cid, "content": "{}"})
            waiting = {}
        if role in ("customer", "agent"):
            if text := clean(h.get("text")):
                out.append({"role": "user" if role == "customer" else "assistant", "content": text})
        elif role == "tool_call":
            call = {"id": f"h{i}", "type": "function",
                    "function": {"name": name, "arguments": json.dumps(h.get("arguments") or {}, ensure_ascii=False)}}
            if prev == "tool_call":                           # 연달아 부른 호출은 한 메시지로 묶는다
                out[-1]["tool_calls"].append(call)
            else:
                out.append({"role": "assistant", "content": None, "tool_calls": [call]})
            waiting[call["id"]] = name
        elif role == "tool_result":
            if cid := next((c for c, n in waiting.items() if n == name), None):
                del waiting[cid]
                out.append({"role": "tool", "tool_call_id": cid, "content": json.dumps(h.get("result"), ensure_ascii=False)})
        prev = role
    return out


def system_prompt(start, said="", decision=None, closing=True):
    fixed ="\n\n[고정 문장]\n아래 상황에서는 해당 문장을 한 글자도 바꾸지 말고 그 문장만 말한다.\n" + "\n".join(
        f"- {when}: {line}" for when, line in PHRASES.items())
    info = f"\n\n[통화 정보]\n현재 시각: {start.get('now')}"        # 통화마다 달라지는 내용은 맨 끝에 둔다 (앞부분이 같아야 LLM 이 캐시를 쓴다)
    if start.get("caller"):
        info += f"\n발신자 정보(걸려 온 번호로 찾은 회원. 다른 번호로 주문했을 수 있으니 조회가 안 되면 다른 번호를 여쭙는다): {json.dumps(start['caller'], ensure_ascii=False)}"
    if said:                                                         # 매크로가 이미 나간 통화: 같은 말을 되풀이하지 않게 알려 준다
        info += (f"\n서버가 방금 손님에게 먼저 한 말: \"{said}\" 이 말은 되풀이하지 말고 결과·본론부터 답한다. "
                 "'확인해 볼게요' 같은 말과 끝인사('더 궁금하신 점…')는 쓰지 않는다" + (" (끝인사는 서버가 붙인다)." if closing else "."))
    if decision and decision.guide:                                  # Jev 가 업무를 가려냈으면 그 업무의 지침(scenarios.md 해당 구역)을 한 번 더 붙인다
        info += f"\n[이번 통화의 업무: {decision.name}] 손님의 말이 이 업무로 분류됐다. 아래 지침을 우선 따른다 (다른 업무가 분명하면 그 업무 지침을 따른다).\n{decision.guide}"
    return PROMPT + fixed + info


def llm_options(tools=None):
    options = {"tools": tools} if tools else {}
    if LLM_EFFORT:
        options["reasoning_effort"] = LLM_EFFORT
    return options


async def think(start, text, tools, filler=None, trace=None, said="", decision=None, closing=True):
    """LLM 에게 묻는다. 함수를 부르면 전산실에 물어보고 그 결과로 다시 묻는다. filler 는 함수를 부를 때 먼저 한마디 하는 함수 (말한 문장을 돌려줌).
    trace 가 리스트면 실제로 부른 함수와 결과를 history 형식으로 담는다 (시험기 live_chat.py 용)."""
    masker = Masker()
    messages = [{"role": "system", "content": system_prompt(start, said, decision, closing)},
                *history_messages(start.get("history")),
                {"role": "user", "content": text}]
    done = []                                                 # 이번 답에서 받은 함수 결과 (번호 검사 근거)
    for round_no in range(MAX_TOOL_ROUNDS + 1):
        resp = await ai.chat.completions.create(model=LLM_MODEL, messages=mask_messages(masker, messages), **llm_options(tools))
        count_llm(resp)
        msg = resp.choices[0].message
        if not msg.tool_calls:
            return masker.unmask((msg.content or "").strip()) or SORRY
        ctx = call_context(start, done)
        checked = [parse_call(c, tools, masker, ctx) for c in msg.tool_calls]
        said = await filler() if filler and round_no == 0 and any(problem is None for _, problem in checked) else None
        messages.append({"role": "assistant", "content": said or msg.content, "tool_calls": [
            {"id": c.id, "type": "function", "function": {"name": c.function.name, "arguments": c.function.arguments}}
            for c in msg.tool_calls]})
        results = await asyncio.gather(*(run_tool(start, c.function.name, *p) for c, p in zip(msg.tool_calls, checked)))
        for call, result in zip(msg.tool_calls, results):
            done.append((call.function.name, result))
            messages.append({"role": "tool", "tool_call_id": call.id, "content": json.dumps(result, ensure_ascii=False)})
        if trace is not None:
            for call, (args, problem), result in zip(msg.tool_calls, checked, results):
                if problem is None:
                    trace += [{"role": "tool_call", "name": call.function.name, "arguments": args},
                              {"role": "tool_result", "name": call.function.name, "result": result}]
    return SORRY


# ── 4. 말하기: Gemini TTS ───────────────────────────────────────────
def sentences(text):
    """TTS 로 읽힐 단위로 나눈다. TTS_PER_SENTENCE=0 이면 답 전체를 한 번에 읽혀 요청 수를 줄인다."""
    if not TTS_PER_SENTENCE:
        return [text.strip()] if text.strip() else []
    return [s for s in re.split(r"(?<=[.!?])\s+", text.strip()) if s]


def tts_audio(r):
    """Gemini 응답에서 음성 데이터를 찾는다. 없으면 None."""
    for cand in r.candidates or []:
        for part in (cand.content.parts if cand.content else None) or []:
            if part.inline_data and part.inline_data.data:
                return part.inline_data.data
    return None


def tts_problem(r):
    """음성이 없을 때 이유(종료 사유, 차단 사유, 글자로만 답했는지)를 한 줄로 정리한다."""
    cand = r.candidates[0] if r.candidates else None
    parts = (cand.content.parts if cand and cand.content else None) or []
    said = " ".join(p.text for p in parts if p.text)[:60]
    block = r.prompt_feedback.block_reason if r.prompt_feedback else None
    return f"종료사유={cand.finish_reason if cand else None}, 차단사유={block}, 글자 응답={said!r}, 모델={r.model_version}"


def tts_config():
    return types.GenerateContentConfig(
        response_modalities=["AUDIO"],
        speech_config=types.SpeechConfig(voice_config=types.VoiceConfig(
            prebuilt_voice_config=types.PrebuiltVoiceConfig(voice_name=TTS_VOICE))))


async def gateway_tts(text):
    """BAZE 게이트웨이의 /audio/speech 를 부른다. 24kHz 16-bit PCM 이 통째로 돌아온다 (조각으로 나눠 주지 않는다)."""
    for _ in range(2):                                        # 가끔 "음성 없음"(400)으로 실패하니 한 번 더 시도한다
        r = await http.post(f"{BAZE_URL}/audio/speech/", headers={"Authorization": f"Bearer {BAZE_KEY}"}, timeout=20,
                            json={"model": TTS_MODEL, "input": text, "voice": TTS_VOICE})
        if r.status_code != 400 or "no audio" not in r.text:
            break
    r.raise_for_status()
    return r.content or None, ("" if r.content else f"빈 음성 (모델={TTS_MODEL})")


async def gemini_tts(text):
    """TTS 를 한 번 부른다 (게이트웨이 키가 있으면 게이트웨이, 없으면 Gemini 직접). (24kHz PCM 음성, 실패 이유)를 돌려준다. 음성이 없으면 음성 자리는 None."""
    if BAZE_KEY:
        return await gateway_tts(text)
    r = await gemini.aio.models.generate_content(model=TTS_MODEL, contents=text, config=tts_config())
    audio = tts_audio(r)
    return audio, ("" if audio else tts_problem(r))


def to_16k(raw):
    """Gemini 의 24kHz PCM 을 대회 규격 16kHz 로 바꾼다."""
    raw = raw[: len(raw) // 2 * 2]
    g = math.gcd(RATE, TTS_RATE)
    pcm = resample_poly(np.frombuffer(raw, dtype=np.int16), RATE // g, TTS_RATE // g)
    return np.clip(pcm, -32768, 32767).astype(np.int16).tobytes()


async def synth(sentence):
    """문장 하나를 Gemini TTS 로 읽혀 16kHz PCM 으로 만든다."""
    problem = ""
    for style in STYLES:                                      # 실패하면 읽어 달라는 지시를 바꿔 가며 최대 3번 시도한다 (게이트웨이는 지시 없이 1번)
        try:
            raw, problem = await gemini_tts(style + sentence)
        except genai_errors.ClientError as e:
            if "generate text" not in str(e):                 # 글로 답하려다 실패한 경우만 다시 시도하고, 키·한도 같은 문제는 그대로 알린다
                raise
            raw, problem = None, str(e)[:150]
        if raw:
            break
    else:
        raise RuntimeError(f"Gemini TTS 가 음성을 돌려주지 않았습니다 ({problem})")
    pcm = to_16k(raw)
    USAGE["tts_sec"] += len(pcm) / (RATE * 2)
    return pcm


def lead_start(pcm):
    """앞 무음이 끝나는 위치(바이트). 소리가 없으면 None."""
    for i in range(0, len(pcm) - FRAME + 1, FRAME):
        if rms(pcm[i:i + FRAME]) >= LEAD_RMS:
            return max(0, i - LEAD_KEEP_MS * RATE * 2 // 1000)
    return None


def trim_lead(pcm):
    """음성 앞쪽의 무음을 잘라낸다. 측정기는 소리가 처음 도착한 시각을 '첫 소리'로 본다."""
    start = lead_start(pcm)
    return pcm if start is None else pcm[start:]


def time_stretch(pcm, speed, rate=RATE):
    """음높이는 그대로 두고 말 속도만 바꾼다 (WSOLA). speed 0.9 → 길이 1/0.9.
    30ms 창을 15ms 간격으로 이어 붙이되, 앞 조각과 파형이 가장 잘 이어지는 자리를 ±10ms 안에서 찾는다. 10초 문장에 약 20ms 라 첫 소리 시간에 영향이 없다."""
    if not pcm or abs(speed - 1.0) < 1e-3:
        return pcm
    pcm = pcm[: len(pcm) // 2 * 2]                           # 홀수 바이트는 마지막 1바이트를 버린다 (int16 단위)
    n_in = len(pcm) // 2
    frame = int(rate * 0.03)
    hop_out = frame // 2
    hop_in = hop_out * speed
    tol = int(rate * 0.01)
    if n_in < frame * 3 + tol * 2:                           # 약 0.1초 미만은 그대로
        return pcm
    # 끝에 0을 덧대 마지막 조각까지 다 쓰고, 결과를 목표 길이(입력 ÷ 속도)로 자른다 (덧대지 않으면 끝 어미 약 40ms 가 잘린다)
    x = np.concatenate([np.frombuffer(pcm, dtype=np.int16).astype(np.float32), np.zeros(frame + tol + hop_out, dtype=np.float32)])
    target_len = int(round(n_in / speed))
    n = int((len(x) - frame - tol - hop_out) / hop_in)
    if n < 2:
        return pcm
    win = np.hanning(frame).astype(np.float32)
    y = np.zeros(n * hop_out + frame, dtype=np.float32)
    w = np.zeros_like(y)
    prev = 0
    for k in range(n):
        if k == 0:
            pos = 0
        else:
            nat = prev + hop_out                             # 앞 조각이 자연스럽게 이어지는 자리
            ref = x[nat:nat + frame]
            target = int(round(k * hop_in))
            lo, hi = max(0, target - tol), min(len(x) - frame, target + tol)
            if hi <= lo or len(ref) < frame:
                pos = min(max(0, target), len(x) - frame)
            else:
                pos = lo + int(np.argmax(np.correlate(x[lo:hi + frame], ref, mode="valid")))
        o = k * hop_out
        y[o:o + frame] += x[pos:pos + frame] * win
        w[o:o + frame] += win
        prev = pos
    y = (y / np.where(w > 0.1, w, 1.0))[:target_len]
    return np.clip(y, -32768, 32767).astype(np.int16).tobytes()


PACED = {}                                                   # (음성 해시, 길이, 속도) → 느리게 만든 음성. 같은 고정 문구를 다시 변환하지 않는다 (최대 200개)


def paced(pcm, speed):
    """보내기 직전에 재생 속도를 적용한다. 음성은 늘 1.0배로 만들어 캐시하고, 여기서 한 번만 바꾼다 (캐시에 속도를 넣으면 다시 바꿀 때 두 번 변환돼 음질이 떨어진다). 실패하면 원래 음성."""
    if speed >= 1.0 or not pcm:
        return pcm
    key = (hash(pcm), len(pcm), speed)
    if key not in PACED:
        try:
            out = time_stretch(pcm, speed)
        except Exception as e:
            print(f"재생 속도 변환 실패, 원래 음성으로: {e!r}"[:200], flush=True)
            return pcm
        if len(PACED) >= 200:
            PACED.pop(next(iter(PACED)))
        PACED[key] = out
    return PACED[key]


def silence_pcm(ms):
    return bytes(int(RATE * ms / 1000) * 2)


def phrase_key(text):
    """고정 문구를 찾을 때 띄어쓰기와 문장부호 차이는 무시한다."""
    return re.sub(r"[\s.,!?~…]", "", text)


async def stream_tts(text):
    """Gemini 가 음성을 조각으로 보내는 대로 16kHz PCM 조각을 내보낸다 (24kHz→16kHz 는 샘플 3개 단위로 끊어서 변환)."""
    stream = await gemini.aio.models.generate_content_stream(model=TTS_MODEL, contents=TTS_STYLE + text, config=tts_config())
    carry, lead = b"", True
    async for chunk in stream:
        raw = tts_audio(chunk)
        if not raw:
            continue
        carry += raw
        cut = len(carry) // 6 * 6
        if not cut:
            continue
        piece, carry = to_16k(carry[:cut]), carry[cut:]
        if lead:                                              # 맨 앞 무음은 건너뛴다
            start = lead_start(piece)
            if start is None:
                continue
            piece, lead = piece[start:], False
        USAGE["tts_sec"] += len(piece) / (RATE * 2)
        yield piece


async def speak(sentence):
    """문장 하나를 통째로 읽혀서(앞 무음은 자르고) 돌려준다. 고정 문구는 미리 만들어 둔 것을 쓴다."""
    return FIXED.get(phrase_key(sentence)) or trim_lead(await synth(sentence))


async def tts_pieces(text):
    """한 번 시도: 스트리밍으로 받아 16kHz PCM 조각을 내보낸다. 스트리밍이 안 되면 통째로 만든 음성을 내보낸다."""
    if TTS_STREAM:
        sent = False
        try:
            async for piece in stream_tts(text):
                sent = True
                yield piece
        except Exception as e:
            if sent or "RESOURCE_EXHAUSTED" in str(e):        # 이미 말하기 시작했거나 한도 초과면 통째 방식으로 되돌리지 않고 그대로 알린다
                raise
            print(f"스트리밍 실패, 통째 방식으로 다시 시도: {str(e)[:80]}", flush=True)
        else:
            if sent:
                return
    yield trim_lead(await synth(text))


def quota_wait(e):
    """Gemini 분당 호출 한도(429)에 걸렸고 '몇 초 뒤에 다시'라고 알려 준 시간이 짧으면 그 시간(초)을, 아니면 None 을 돌려준다."""
    m = re.search(r"retry in ([\d.]+)s", str(e)) if "RESOURCE_EXHAUSTED" in str(e) else None
    return float(m.group(1)) + 0.5 if m and float(m.group(1)) <= QUOTA_RETRY_MAX_S else None


async def speak_chunks(text):
    """문장 하나를 읽혀서 16kHz PCM 조각을 만들어지는 대로 내보낸다. 고정 문구는 미리 만든 것을 쓰고, 한도 초과(429)는 기다릴 만하면 한 번 더 시도한다."""
    ready = FIXED.get(phrase_key(text))
    if ready:
        yield ready
        return
    sent = False
    for attempt in range(2):
        try:
            async for piece in tts_pieces(text):
                sent = True
                yield piece
            return
        except Exception as e:
            wait = quota_wait(e)
            if sent or attempt or wait is None:               # 말하기 시작했거나, 이미 한 번 다시 했거나, 기다릴 수 없으면 그대로 알린다
                raise
            print(f"[한도 초과] {wait:.0f}초 뒤 다시 시도합니다 (Gemini 분당 호출 한도)", flush=True)
            await asyncio.sleep(wait)


async def audio_pieces(units):
    """문장들의 음성 조각을 문장 순서대로 내보낸다. 스트리밍이 꺼져 있으면(게이트웨이) 모든 문장의 음성을 동시에 만들기 시작해서,
    첫 문장이 끝나는 즉시 말을 시작하고 나머지는 그동안 만들어 둔다."""
    if TTS_STREAM:
        for unit in units:
            async for piece in speak_chunks(unit):
                yield piece
        return

    async def whole(unit):
        return [piece async for piece in speak_chunks(unit)]

    tasks = [asyncio.create_task(whole(unit)) for unit in units]
    try:
        for task in tasks:
            for piece in await task:
                yield piece
    finally:
        for task in tasks:
            task.cancel()


async def sentence_audio(units):
    """문장 하나의 음성을 통째로 모아 문장 순서대로 내보낸다 (재생 속도를 바꿀 때 쓴다: 문장을 다 받은 뒤에야 변환할 수 있다).
    스트리밍이 꺼져 있으면(게이트웨이) 모든 문장을 동시에 만들기 시작하고, 켜져 있으면(직접 호출) 한도를 아끼려고 차례로 만든다."""
    async def whole(unit):
        return b"".join([piece async for piece in speak_chunks(unit)])

    if TTS_STREAM:
        for unit in units:
            yield await whole(unit)
        return
    tasks = [asyncio.create_task(whole(unit)) for unit in units]
    try:
        for task in tasks:
            yield await task
    finally:
        for task in tasks:
            task.cancel()


async def say(ws, text, lap=None, label="첫 소리 전송"):
    """답을 읽혀서, 음성이 만들어지는 대로 20ms 조각으로 바로 내보낸다. 이번 답의 재생 속도(PLAY_SPEED)가 1.0 미만이면
    문장 단위로 음성을 모아 음높이를 유지한 채 느리게 바꾸고, 문장 사이 쉼도 같은 비율로 늘려서 내보낸다."""
    announced = False                                         # 글(agent.transcript)은 첫 음성이 만들어진 뒤에 보낸다: 음성 만들기가 실패하면 말하지도 않은 글이 대화 기록에 남지 않게

    async def announce():
        nonlocal announced
        if not announced:
            announced = True
            await ws.send(json.dumps({"type": "agent.transcript", "text": text}, ensure_ascii=False))

    pending, first = b"", True

    async def feed(piece):
        nonlocal pending, first
        pending += piece
        while len(pending) >= FRAME:
            await announce()
            await ws.send(pending[:FRAME])
            pending = pending[FRAME:]
            if first and lap:
                lap(label)
                first = False

    units, speed = sentences(for_tts(text)), PLAY_SPEED.get()
    if speed < 1.0:
        count = 0
        async for pcm in sentence_audio(units):
            if count:
                await feed(silence_pcm(PACE_PLAY_GAP_MS / speed))
            await feed(paced(pcm, speed))
            count += 1
    else:
        async for piece in audio_pieces(units):
            await feed(piece)
    if pending:
        await announce()
        await ws.send(pending.ljust(FRAME, b"\0"))


# ── 5. 한 통화 처리 ─────────────────────────────────────────────────
def has_number(text, start):
    """조회에 쓸 번호가 있는가: 걸려 온 번호, 앞선 조회 결과, 손님이 말한 숫자 7자리 이상(전화·주문번호)."""
    history = start.get("history") or []
    said = text + " ".join(h.get("text") or "" for h in history if h.get("role") == "customer")
    return bool(start.get("caller")) or any(h.get("role") == "tool_result" for h in history) or sum(c.isdigit() for c in said) >= 7 or "공일공" in said


# ── 앞뒤 인사: 인사·감사만 한 말은 LLM 을 거치지 않고 미리 만들어 둔 음성으로 바로 답한다 (가장 빠르고, 말이 늘 같다) ──
_FILLER = r"(?:아하|아|어|음|네|넵|예|그래요|알겠습니다|알겠어요|좋아요|됐어요|괜찮아요|그렇군요|오케이)"
_ADVERB = r"(?:정말|너무|진짜|많이|도와주셔서|덕분에|친절하게|친절히|잘|그럼요|그럼)"
GREETING_RE = re.compile(rf"^{_FILLER}*(?:안녕하세요|안녕하십니까|안녕하세여|하이|여보세요|저기요)(?:하루온(?:입니까|인가요|이죠|이에요|맞나요|입니다)?|고객센터(?:인가요|죠|맞나요)?)?요?$")
_THANKS = (r"(?:감사(?:합니다|해요|드립니다|드려요|했습니다|했어요)|고맙(?:습니다|네요|어요)|고마워요|수고(?:하세요|하십시오|하셨습니다|많으셨어요)|고생(?:하세요|하셨습니다|많으셨어요)"
           r"|안녕히(?:계세요|계십시오|가세요)|들어가세요|좋은하루(?:되세요|보내세요|되십시오))")
CLOSING_RE = re.compile(rf"^{_FILLER}*{_ADVERB}*{_THANKS}(?:{_FILLER}|{_ADVERB}|{_THANKS})*$")


def hold_ignored(text):
    """Jev 가 hold(말이 더 이어질 것)라고 해도 따르지 않을 만큼 '이미 끝난' 글인가. hold 를 따르면 무음을 2.5초까지 기다려 첫 소리가 크게 늦어진다.
    (연습 80문항 실측: hold 9% 중 인사·완결 번호·'~했다네요' 같은 끝난 말이 섞여 있었고, 인사는 무음 길이와 상관없이 늘 hold 0.95 이상이었다)
    끝난 글: 인사·감사만 / 번호를 다 말함(10자리 이상) / 문장부호나 ~요·~다·~죠·~까 로 끝남. 단 '~는데요'처럼 이어질 모양이거나 번호를 덜 말했으면 hold 를 따른다."""
    t = (text or "").strip()
    core = t.rstrip(".,?!…~ ")
    if not core:
        return False
    flat = re.sub(r"[\s.,!?~…]+", "", t)
    if GREETING_RE.match(flat) or CLOSING_RE.match(flat):
        return True
    m = DIGITS_TAIL.search(core)
    if m:
        return sum(c.isdigit() for c in m.group(1)) >= 10
    if core.endswith(CONTINUING) or core.endswith(jev.OPEN_ENDS):
        return False
    return t[-1] in ".?!" or core.endswith(("요", "다", "죠", "까"))


def greeting_reply(text, start):
    """손님 말이 인사만(처음), 또는 감사·끝인사만(상담 끝)이면 정해 둔 답 문장을, 아니면 None. 다른 말이 조금이라도 섞이면 None (LLM 이 처리).
    - 인사: 이 통화의 첫 말일 때만. - 감사: 상담원이 한 말이 있고, 그 말이 손님에게 답을 기다리는 질문이 아닐 때만.
    - 답의 음성이 아직 준비되지 않았으면 None (지금 만들면 오히려 느리다)."""
    flat = re.sub(r"[\s.,!?~…]+", "", text or "")                 # 띄어쓰기·문장부호 차이는 무시 ("아 넵 감사합니다." = "아넵감사합니다")
    history = [h for h in start.get("history") or [] if h.get("role") in ("customer", "agent")]
    if not flat or not history and not GREETING_RE.match(flat):
        return None
    if not history:
        reply = GREETING_REPLY
    else:
        asked = next((h.get("text") or "" for h in reversed(history) if h.get("role") == "agent"), None)
        if asked is None or "?" in asked[-60:] or not CLOSING_RE.match(flat):
            return None                                           # 상담원이 한 말이 없거나, 방금 질문했거나(감사가 대답이 아님), 다른 말이 섞임
        reply = CLOSING_REPLY
    return reply if all(phrase_key(unit) in FIXED for unit in sentences(for_tts(reply))) else None


def pick_case(text, start, decision=None):
    """이번 손님 말의 유형을 키워드로 정해 (먼저 하는 말, 끝맺음 말을 붙일지) 를 돌려준다. 확실하지 않으면 None (잘못된 매크로는 없는 것보다 나쁘다).
    같은 통화에서 이미 한 말은 피하고, 음성이 미리 만들어져 있지 않은 문구는 쓰지 않는다 (지금 만들면 오히려 늦다).
    decision(Jev 업무 분류)이 있으면 거부권만 준다: 상담원 연결·오배송·배송지 변경이거나 키워드와 다른 유형이면 매크로를 내보내지 않는다. (Jev 가 없거나 '기타'면 키워드만 본다)"""
    if not (MACRO and text) or SKIP_RE.search(text):
        return None
    hits = [name for name, case in CASES.items() if case[0].search(text)]
    if len(hits) != 1:
        return None
    if decision and decision.case_id and (decision.case_id in jev.NO_MACRO or (decision.macro and decision.macro != hits[0])):
        return None
    _, need_number, intros, closing = CASES[hits[0]]
    if need_number and not has_number(text, start):
        return None
    before = " ".join(h.get("text") or "" for h in start.get("history") or [] if h.get("role") == "agent")
    ready = [p for p in intros if p not in before and phrase_key(p) in FIXED]
    return (random.choice(ready), closing) if ready else None


# 첫마디: 말 끝 확정 직후 바로 내는 녹음 한마디 (결과를 약속하지 않는 말). 상황별 문구 중 음성이 준비된 것이 없으면 채움말(FILLERS)을 쓴다.
# 새 문구는 키를 넣고 서버를 처음 켤 때 phrases/ 에 만들어진다 (warm_up)
OPENERS = {
    "empathy": ["불편을 드려 죄송합니다.", "많이 불편하셨겠어요."],
    "ack_check": ["네, 확인해 드리겠습니다.", "네, 확인 도와드릴게요."],
    "neutral_info": ["네, 안내해 드리겠습니다.", "네, 말씀드리겠습니다."],
    "neutral": ["네, 고객님.", "네, 알겠습니다."],
}
UPSET_RE = re.compile(r"왜|화나|짜증|아직도|언제까지|몇\s*번")


def pick_opener(text, start, decision=None):
    """유형 매크로(pick_case)가 안 걸린 말에도 말 끝 직후 미리 만든 짧은 첫마디를 먼저 내보낸다.
    왜: 본 답은 LLM(약 0.9초) + 음성 합성(게이트웨이 2.3초 이상)이 끝나야 나와서 첫 소리가 4초대인데, 첫마디는 합성이 필요 없어 말 끝 판정 + 받아쓰기(약 1.0초) 직후 나간다 (첫 소리 만점은 1.5초).
    돌려주는 값은 pick_case 와 같은 (먼저 하는 말, 끝맺음 여부=False) 또는 None. 아래는 첫마디가 어색한 말이라 쓰지 않는다:
    짧은 말(대답·추임새) / 말 고치기 / 몸이 불편하다·상담원·불만처럼 조심할 말(SKIP_RE) / Jev 가 상담원 연결·오배송·배송지 변경·기타(잡담)로 본 말."""
    flat = re.sub(r"[\s.,!?~…]+", "", text or "")
    if not (OPENER and MACRO) or len(flat) < 6 or SKIP_RE.search(text) or CORRECTING.search(text.strip()) or CORRECTING_ONLY.match(text.strip()):
        return None
    if decision and decision.case_id and (decision.case_id in jev.NO_MACRO or decision.case_id == "other"):
        return None
    kind = ("empathy" if UPSET_RE.search(text) else
            "ack_check" if re.search(r"\d{4}|배송|주문|입금|환불|반품|취소|교환|조회|확인|택배|송장", text) else
            "neutral_info" if re.search(r"효과|먹|섭취|복용|성분|가격|얼마|쿠폰|결제|카드|회원|방법|적립", text) else "neutral")
    before = " ".join(h.get("text") or "" for h in start.get("history") or [] if h.get("role") == "agent")
    usable = lambda lines: [p for p in lines if p not in before and phrase_key(p) in FIXED]   # 같은 통화에서 한 말은 피한다
    ready = usable(OPENERS[kind]) or usable(FILLERS)
    return (random.choice(ready), False) if ready else None


def after_opener(reply, opener):
    """첫마디 뒤 본 답: 앞의 '네,'와 첫마디와 거의 같은 첫 문장은 뺀다 ('네, 확인해 드리겠습니다. 확인 도와드릴게요.' 방지)."""
    parts = [p for p in re.split(r"(?<=[.!?])\s+", reply.strip()) if p]
    if len(parts) > 1 and difflib.SequenceMatcher(None, phrase_key(parts[0]), phrase_key(opener)).ratio() >= 0.6:
        parts = parts[1:]
    out = " ".join(parts)
    if opener.startswith("네"):
        out = re.sub(r"^\s*(네|예)\s*[,.!]\s*", "", out) or out
    return out or reply


def with_closing(reply, closing):
    """본 답이 끝난 일을 알리는 말이면 끝맺음 말을 붙인다 (되묻거나 넘기는 답, 사과에는 붙이지 않는다)."""
    if not closing or reply in (SORRY, NOT_HEARD) or NO_CLOSING_RE.search(reply):
        return reply
    ready = [c for c in CLOSINGS if phrase_key(c) in FIXED]
    return f"{reply.rstrip()}{'' if reply.rstrip()[-1] in '.!' else '.'} {random.choice(ready)}" if ready else reply


async def respond(ws, start, stt, tools, lap, heard=None, router=None, profile=None):
    cid = start.get("case_id")
    text = await stt.finish()
    lap("받아쓰기 확정")
    if profile is not None:                                   # 말 속도에 맞춘 처리 (시험기·시연): 이전 말들의 기록으로 이번 답의 재생 속도를 정하고, 그다음 이번 말의 속도를 기록에 섞는다
        speed = play_speed(profile, live=bool(start.get("debug")))
        PLAY_SPEED.set(speed)                                 # 이 답을 말하는 작업들(먼저 하는 말·채움말·본 답)이 읽어 간다
        if speed < 1.0:
            print(f"[{cid}] 느린 말투 → 상담원 재생 {speed}배, 문장 사이 쉼 {round(PACE_PLAY_GAP_MS / speed)}ms", flush=True)
        if heard:
            profile.note_rate(turn_rate(text, heard["voiced_ms"]))
    if reply := greeting_reply(text, start):                  # 인사만 / 감사·끝인사만: LLM·업무 분류·매크로 없이 미리 만든 음성으로 바로 답한다
        if start.get("debug"):                                # 시험기에게도 다른 답과 같은 신호를 보낸다 (함수 호출은 없으므로 history_add 는 비어 있다)
            await ws.send(json.dumps({"type": "debug.heard", "customer": text}, ensure_ascii=False))
            await ws.send(json.dumps({"type": "debug.turn", "customer": text, "history_add": []}, ensure_ascii=False))
        if VERBOSE:
            print(f"[{cid}] 고객: {text}\n[{cid}] 상담원: {reply} (인사 응대)", flush=True)
        else:
            print(f"[{cid}] 인사 응대", flush=True)
        lap("인사 응대")
        await say(ws, reply, lap)
        lap("말하기 끝")
        return
    decision = await router.resolve(text) if router and text else None   # Jev 업무 분류: 말하는 동안 미리 물어 둔 결과를 쓰고, 없으면 최대 CASE_WAIT_S 만 기다린다
    if decision:
        lap("업무 분류")
        print(f"[{cid}] Jev 업무: {decision.short()}", flush=True)
    if start.get("debug"):                                    # 시험기에게 받아쓴 글을 바로 알려 준다 (손님이 말을 끊어도 어떤 말이었는지 기록할 수 있게)
        await ws.send(json.dumps({"type": "debug.heard", "customer": text}, ensure_ascii=False))
    macro = pick_case(text, start, decision)                  # (먼저 하는 말, 끝맺음 여부) 또는 None
    opener = None if macro or not text else pick_opener(text, start, decision)   # 유형 매크로가 없으면 일반 첫마디 (검토 2-1절: 함수 없는 답의 첫 소리)
    first = (macro or opener or (None,))[0]
    intro = asyncio.create_task(say(ws, first, lap, "첫마디 전송")) if first else None   # 고정 음성이라 바로 나간다. 그동안 아래에서 LLM 이 생각한다

    async def filler():
        """함수 조회는 1~2초 걸리니, 그동안 미리 만들어 둔 짧은 한마디를 먼저 말한다 (첫 소리가 빨라짐). 아직 준비가 안 됐으면 None."""
        ready = [p for p in FILLERS if phrase_key(p) in FIXED]
        if not ready:
            return None
        phrase = random.choice(ready)
        await say(ws, phrase, lap, "채움말 전송")
        return phrase

    trace = [] if start.get("debug") else None                # 시험기(live_chat.py)가 session.start 에 debug 를 켠 때만 쓴다. 측정기는 켜지 않는다
    try:
        reply = await think(start, text, await tools, None if first else filler, trace, first or "", decision, closing=bool(macro and macro[1])) if text else NOT_HEARD   # 첫마디가 나갔으면 채움말은 또 하지 않는다
        if opener:
            reply = after_opener(reply, opener[0])
    except BaseException:
        if intro:
            intro.cancel()
        raise
    lap("생각")
    if trace is not None:                                     # 시험기가 다음 말의 history 를 만들 수 있게 받아쓴 글과 부른 함수를 알려 준다
        await ws.send(json.dumps({"type": "debug.turn", "customer": text, "history_add": trace}, ensure_ascii=False))
    if VERBOSE:
        print(f"[{cid}] 고객: {text}\n[{cid}] 상담원: {reply}", flush=True)
    if intro:
        await intro                                           # 먼저 하는 말을 다 보낸 뒤 본 답을 잇는다 (순서 보장. 고정 음성이라 보통 이미 끝나 있다)
    await say(ws, with_closing(reply, macro and macro[1]), lap)
    lap("말하기 끝")


async def talk(ws, start, stt):
    tools = asyncio.create_task(load_tools(start))            # 고객이 말하는 동안 미리 받아 둔다
    judge = jev.TurnJudge(http, start.get("history")) if jev.active(jev.EOT_MODE, start) else None      # Jev 말 끝 판정 (키가 없거나 꺼져 있으면 None → 예전 규칙)
    router = jev.CaseRouter(http, start.get("history")) if jev.active(jev.CASE_ROUTER, start) else None    # Jev 업무 분류
    pace_on = pace_enabled(start)
    profile = profile_for(start) if pace_on else None         # 통화 기록 (call_id 를 보내는 시험기·시연만. 채점기는 안 보내서 None → 말 속도 기능이 구조적으로 꺼짐)
    try:
        heard = await listen(ws, stt, pace_on, profile, judge, router, live=bool(start.get("debug")))
    except BaseException:
        if router:
            router.close()
        raise
    cid = start.get("case_id")
    if e := heard["eot"]:                                     # 받아쓴 글은 남기지 않는다
        print(f"[{cid}] Jev 말 끝: 요청 {e['calls']}회 (end {e['end']}·hold {e['hold']}·오류 {e['errors']}·글 모양으로 거름 {e['vetoed']}), 평균 {e['avg_ms']}ms", flush=True)
    if p := heard["pace"]:                                    # 받아쓴 글은 남기지 않는다. cps 를 보고 PACE_REF_CPS 를 맞춘다 (pace.md 7절)
        cps = str(p["cps"]) + "자/초" if p["cps"] else "(못 잼)"
        print(f"[{cid}] 말 속도 {cps}, 말 중간 최대 쉼 {p['own_pause_ms']}ms → 말 끝 대기 {p['base_ms']}→{p['thr_ms']}ms" + ("" if p["ready"] else " (이 통화의 첫 말: 기록만)"), flush=True)
    lap, sink = Lap(cid), asyncio.create_task(drain(ws, profile, heard["silent_ms"]))      # 답하는 동안 고객이 이어 말하면 그 쉼을 배운다
    work = asyncio.create_task(asyncio.wait_for(respond(ws, start, stt, tools, lap, heard, router, profile), DEADLINE_S))
    closed, gone = asyncio.create_task(ws.wait_closed()), False
    try:
        await asyncio.wait({work, closed}, return_when=asyncio.FIRST_COMPLETED)
        if not work.done():                                   # 답하는 도중에 손님 쪽이 연결을 닫았다 (말 끊기): 하던 LLM·음성 합성을 바로 멈춘다 (크레딧도 아낌)
            gone = True
            work.cancel()
            print(f"[{cid}] 손님이 말을 끊고 연결을 닫음 — 답을 멈춤", flush=True)
            return
        work.result()
    except Exception as e:                                    # 무슨 일이 있어도 끝 신호는 보낸다
        print(f"[{cid}] 오류: {e!r}"[:400], flush=True)
        if "RESOURCE_EXHAUSTED" in str(e):                          # 위 오류 로그는 400자에서 잘려 '하루/분당' 구분이 안 보이니, 어떤 한도인지 한 줄로 알려 준다
            daily, wait = "PerDay" in str(e), re.search(r"retry in ([0-9hms.]+)", str(e))
            when = f" 풀리는 시간: 약 {re.sub(r'[.][0-9]+', '', wait.group(1)).rstrip('.')} 뒤." if wait else ""
            print(f"[{cid}] ※ Gemini {'하루' if daily else '분당'} 요청 한도를 다 썼어요 → 새 음성을 못 만들어 사과 문구만 나갑니다.{when} "
                  + ("BAZE_API_KEY 를 켜서 게이트웨이로 바꾸거나 한도가 풀릴 때까지 기다려야 해요." if daily else "1분쯤 쉬었다가 다시 해 보세요."), flush=True)
        lap("실패")
        with suppress(Exception):
            await asyncio.wait_for(say(ws, SORRY), 4)
    finally:
        sink.cancel()
        closed.cancel()
        if router:
            router.close()
        if not gone:
            lap.report()
            USAGE["calls"] += 1
            print(usage_line(), flush=True)
            await ws.send(json.dumps({"type": "turn.end"}))


async def handle(ws):
    with suppress(websockets.ConnectionClosed):
        start = json.loads(await ws.recv())                   # session.start
        stt = Stt()                                           # 고객이 말하는 동안 받아쓰기 연결을 미리 열어 둔다
        try:
            await talk(ws, start, stt)
        finally:
            await stt.close()


async def warm_up():
    """첫 통화만 느려지지 않게, 서버를 켤 때 자주 쓰는 문구를 음성으로 준비해 두고(사과 음성은 API 가 실패해도 나간다) LLM 연결도 데워 둔다.
    한 번 만든 음성은 phrases/ 폴더에 저장해 두고 다음부터는 불러온다 (TTS 분당 한도를 아끼고 바로 준비된다)."""
    async def fill(unit):
        name = hashlib.sha1(f"{TTS_MODEL}|{TTS_VOICE}|{unit}".encode()).hexdigest()[:12] + ".pcm"     # 모델·목소리·문장이 바뀌면 새로 만든다
        path = PHRASE_DIR / name
        with suppress(Exception):
            if path.exists():
                FIXED[phrase_key(unit)] = path.read_bytes()
                return
            FIXED[phrase_key(unit)] = pcm = trim_lead(await synth(unit))
            path.parent.mkdir(exist_ok=True)
            path.write_bytes(pcm)

    async def llm():
        with suppress(Exception):
            count_llm(await ai.chat.completions.create(model=LLM_MODEL, messages=[{"role": "user", "content": "안녕하세요"}]))

    lines = [NOT_HEARD, SORRY, *PHRASES.values(), *FILLERS, *[p for case in CASES.values() for p in case[2]], *CLOSINGS, *[p for v in OPENERS.values() for p in v]]
    await asyncio.gather(llm(), *(fill(unit) for line in lines for unit in sentences(line)))
    print(f"[준비 완료] 미리 만든 고정 문구 음성 {len(FIXED)}개. 이제 테스트를 시작해도 돼요.", flush=True)


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=9000)
    port = ap.parse_args().port
    async with websockets.serve(handle, "0.0.0.0", port, max_size=None):
        print(f"AI 상담원 서버 시작: ws://0.0.0.0:{port}  (연결={'BAZE 게이트웨이' if BAZE_KEY else '직접 호출'}, LLM={LLM_MODEL}, STT={SONIOX_CONFIG['model']}, TTS={TTS_MODEL}, 스트리밍={'켬' if TTS_STREAM else '끔'}, 문장별 동시 합성={'켬' if TTS_PER_SENTENCE else '끔'}) - '[준비 완료]'가 나올 때까지 기다리세요", flush=True)
        overrides = {k: os.environ[k] for k in ("OPENAI_BASE_URL", "GEMINI_BASE_URL", "SONIOX_URL") if os.getenv(k)}
        if overrides:
            print(f"[주의] 기본 주소 대신 다른 주소를 쓰는 중입니다 (시험용 가짜 서버일 수 있음): {overrides}", flush=True)
        warm = asyncio.create_task(warm_up())
        await asyncio.Future()


if __name__ == "__main__":
    asyncio.run(main())
