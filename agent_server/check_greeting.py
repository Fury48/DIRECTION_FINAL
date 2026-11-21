# [앞뒤 인사 자가 점검] "안녕하세요"만 한 말과 "감사합니다"만 한 말을 정확히 가려내서 미리 만든 음성으로 바로 답하고, 다른 말이 섞이면 LLM 에 맡기는지 확인합니다. (API 는 부르지 않아요)
# 왜 있나: 인사로 잘못 알아들으면 진짜 용건을 놓치고, 인사를 못 알아보면 느리고 어색해서 고칠 때마다 바로 돌려 보려고요.
# 실행: python agent_server/check_greeting.py
import asyncio, json, os
import numpy as np

os.environ.setdefault("BAZE_API_KEY", "x")                  # 시험용: 실제 호출은 하지 않음
import server

TONE = (np.sin(2 * np.pi * 220 * np.arange(8000) / 16000) * 6000).astype(np.int16).tobytes()      # 0.5초짜리 가짜 음성


def ready(*lines):
    server.FIXED.clear()
    for line in lines:
        for unit in server.sentences(server.for_tts(line)):
            server.FIXED[server.phrase_key(unit)] = TONE


ready(server.GREETING_REPLY, server.CLOSING_REPLY)
assert server.GREETING_REPLY == "네 안녕하세요, 무엇을 도와드릴까요?" and server.CLOSING_REPLY == "네 감사합니다. 좋은 하루 보내시길 바랍니다."
assert server.GREETING_REPLY in server.PHRASES.values() and server.CLOSING_REPLY in server.PHRASES.values()     # 서버를 켤 때 음성을 미리 만들고, LLM 이 고정 문장으로도 쓴다

FIRST = {"history": []}                                                                    # 이 통화의 첫 말
LATER = {"history": [{"role": "customer", "text": "배송 언제 와요?"}, {"role": "agent", "text": "내일 도착 예정입니다. 더 궁금하신 점 있으시면 편하게 말씀해 주세요."}]}
ASKED = {"history": [{"role": "customer", "text": "취소할래요"}, {"role": "agent", "text": "주문번호나 주문하신 분 전화번호를 말씀해 주시겠어요?"}]}
g = server.greeting_reply

# 1. 처음 인사만 → 인사 답
for t in ("안녕하세요", "안녕하세요.", "안녕하세요?", "네 안녕하세요", "아, 안녕하세요", "안녕하십니까", "여보세요", "저기요", "안녕하세요 하루온이죠?", "안녕하세요 고객센터인가요", "안녕하세요요"):
    assert g(t, FIRST) == server.GREETING_REPLY, t
# 인사에 용건이 섞이면 LLM 이 처리
for t in ("안녕하세요 배송 문의드려요", "안녕하세요. 주문 취소하고 싶은데요", "네", "배송 언제 와요?", "", "안녕하세요 저기 제가 주문한 게"):
    assert g(t, FIRST) is None, t
assert g("안녕하세요", LATER) is None                                                       # 통화 중간의 인사는 LLM 이 (처음에만 이 답)
assert g("안녕하세요", {}) == server.GREETING_REPLY                                         # history 가 없어도 첫 말

# 2. 상담 끝의 감사·끝인사만 → 끝인사 답
for t in ("감사합니다", "감사합니다.", "아 넵 감사합니다", "아넵감사합니다", "네 알겠습니다 감사합니다", "정말 감사합니다", "도와주셔서 감사합니다", "고맙습니다",
          "수고하세요", "수고하셨습니다", "안녕히 계세요", "그럼 안녕히 계세요", "네 감사합니다 수고하세요", "감사합니다 감사합니다", "좋은 하루 되세요"):
    assert g(t, LATER) == server.CLOSING_REPLY, t
# 다른 말이 섞이거나 감사가 아니면 LLM 이 처리
for t in ("감사합니다. 그런데 반품은 어떻게 하나요?", "감사합니다 하나만 더 여쭤볼게요", "네 알겠습니다", "네", "아 그렇군요", "감사한데 그게 아니라", "감사합니다 근데 환불은요", "안녕히 계세요 아 잠깐만요"):
    assert g(t, LATER) is None, t
assert g("감사합니다", FIRST) is None                                                       # 대화가 없었는데 감사부터 하면 LLM 이
assert g("감사합니다", ASKED) is None                                                       # 상담원이 방금 질문했으면 감사는 대답이 아니다
assert g("감사합니다", {"history": [{"role": "customer", "text": "취소할래요"}]}) is None      # 상담원이 한 말이 없으면 LLM 이
assert g("감사합니다", {"history": [{"role": "customer", "text": "배송?"}, {"role": "agent", "text": "상담원 연결을 도와드리겠습니다…(손님이 말을 끊음)"}]}) == server.CLOSING_REPLY   # 끊긴 말 뒤의 감사도 인사로

# 3. 음성이 준비되지 않았으면 쓰지 않는다 (지금 만들면 오히려 늦다)
ready(server.GREETING_REPLY)
assert g("안녕하세요", FIRST) == server.GREETING_REPLY and g("감사합니다", LATER) is None
ready()
assert g("안녕하세요", FIRST) is None
ready(server.GREETING_REPLY, server.CLOSING_REPLY)


# 4. respond(): 인사면 LLM 을 부르지 않고 미리 만든 음성이 바로 나간다
class SinkWs:
    def __init__(self):
        self.audio, self.events = [], []

    async def send(self, m):
        (self.audio.append(m) if isinstance(m, bytes) else self.events.append(json.loads(m)))


class FakeStt:
    def __init__(self, text):
        self.text = text

    async def finish(self):
        return self.text


def boom(*a, **k):
    raise RuntimeError("LLM 이 불렸어요")


async def no_think(*a, **k):
    boom()

loop = asyncio.new_event_loop()
real_think = server.think
server.think = no_think


def respond(text, start, debug=False):
    ws = SinkWs()
    s = {**start, "case_id": "G1", "debug": debug}
    tools = loop.create_future()
    tools.set_result([])                                      # 함수 목록은 이미 받아 둔 것으로 친다
    loop.run_until_complete(server.respond(ws, s, FakeStt(text), tools, server.Lap("G1")))
    return ws


ws = respond("안녕하세요", FIRST, debug=True)
assert [e["type"] for e in ws.events] == ["debug.heard", "debug.turn", "agent.transcript"]                                     # 시험기가 받는 신호는 다른 답과 같다
assert ws.events[2]["text"] == server.GREETING_REPLY and ws.events[1]["history_add"] == []
assert len(b"".join(ws.audio)) >= len(TONE) - server.FRAME and all(len(f) == server.FRAME for f in ws.audio)                   # 미리 만든 음성(0.5초)이 20ms 조각으로 나간다
ws = respond("아 넵 감사합니다", LATER)
assert [e["type"] for e in ws.events] == ["agent.transcript"] and ws.events[0]["text"] == server.CLOSING_REPLY
assert len(b"".join(ws.audio)) >= len(TONE) * 2 - server.FRAME                                                              # 끝인사는 두 문장이라 미리 만든 음성 둘이 이어서 나간다
try:
    respond("배송 언제 와요?", FIRST)
    raise SystemExit("인사가 아닌 말은 LLM 으로 가야 합니다")
except RuntimeError as e:
    assert "LLM" in str(e)
try:
    respond("감사합니다. 그런데 환불은요?", LATER)
    raise SystemExit("다른 말이 섞이면 LLM 으로 가야 합니다")
except RuntimeError as e:
    assert "LLM" in str(e)
server.think = real_think
loop.close()
print("전부 통과")
