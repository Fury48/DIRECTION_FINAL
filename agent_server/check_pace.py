# [말 속도 대기 자가 점검] 말하는 속도에 맞춰 '말 끝' 대기를 늘리는 기능(pace.md)이 규칙대로 동작하는지 확인합니다. (API 는 부르지 않아요)
# 왜 있나: 이 기능은 늘리기만 하고 줄이면 안 되며, 측정(합성 음성)에는 영향이 없어야 해서 고칠 때마다 바로 돌려 보려고요.
# 실행: python agent_server/check_pace.py
import asyncio, os
import numpy as np

os.environ.setdefault("BAZE_API_KEY", "x")                  # 시험용: 실제 호출은 하지 않음
import server

assert server.PACE_REF_CPS == 5.3 and server.PACE_MAX_FACTOR == 1.2, "기본값이 바뀌었어요 (환경변수 PACE_* 를 지우고 다시 돌려 주세요)"

# 1. 발음 속도: 근거가 부족하면 None
assert server.turn_rate("가나다라마바사아", 1000) is None                    # 소리가 1.5초 미만
assert server.turn_rate("가나다라", 3000) is None                           # 글자 8자 미만
assert server.turn_rate("공일공 사구오칠 0104957696", 3000) is None          # 번호 위주(숫자가 글자의 절반 초과)
assert abs(server.turn_rate("가나다라마바사아자차카타파하거너", 5000) - 3.2) < 1e-9   # 16자 / 5초
assert abs(server.turn_rate("가나다라마바사아 자차카타, 파하거너. 12", 5000) - 3.2) < 1e-9   # 숫자·공백·문장부호는 글자 수에서 뺀다

# 2. 대기 조절: 빠른 사람은 그대로, 느린 사람은 최대 1.2배, 줄이지 않는다, 본인 쉼 반영
assert server.pace_wait(600, 6.1) == 600 and server.pace_wait(600, 5.3) == 600
assert abs(server.pace_wait(600, 4.8) - 600 * 5.3 / 4.8) < 1e-9             # 약 0.66초
assert abs(server.pace_wait(600, 4.4) - 720) < 1e-9 and abs(server.pace_wait(600, 2.0) - 720) < 1e-9   # 상한 1.2배
assert abs(server.pace_wait(600, 6.1, own_pause_ms=500) - 625) < 1e-9       # 본인 쉼 0.5초 → 0.625초
assert abs(server.pace_wait(600, 6.1, own_pause_ms=900) - 720) < 1e-9       # 쉼이 아주 길어도 상한까지만
assert server.pace_wait(800, None) == 800                                   # 못 재면 원래 대기
assert all(server.pace_wait(800, r, p) >= 800 for r in (None, 1.0, 3.0, 5.3, 9.0) for p in (0, 100, 500, 2000))   # 어떤 경우에도 줄지 않는다

# 3. 통화 누적값: 이상값은 버리고 지수 평균
server.PACE_PROFILES.clear()
server.note_rate("a", 5.0); server.note_rate("a", 4.0)
assert abs(server.PACE_PROFILES["a"] - (0.6 * 5.0 + 0.4 * 4.0)) < 1e-9
server.note_rate("a", 0.5); server.note_rate("a", 30); server.note_rate("a", None); server.note_rate("", 5.0)
assert abs(server.PACE_PROFILES["a"] - 4.6) < 1e-9 and "" not in server.PACE_PROFILES

# 4. listen(): 합성 소리로 '조용해진 뒤 몇 ms 만에 말이 끝났다고 보는지' 잰다 (20ms 조각 하나 = 20ms)
LOUD, QUIET = np.full(320, 3000, np.int16).tobytes(), bytes(640)


class FakeStt(server.Stt):
    def __init__(self, text):                                # 받아쓰기 연결 없이 글만 흉내 낸다
        self.text, self.partial = text, ""

    def feed(self, pcm):
        pass


class FakeWs:
    def __init__(self, plan):                                # plan: [(조각, 개수), …]
        self.frames = [f for frame, n in plan for f in [frame] * n]
        self.used = 0

    def __aiter__(self):
        return self

    async def __anext__(self):
        if self.used >= len(self.frames):
            raise StopAsyncIteration
        self.used += 1
        return self.frames[self.used - 1]


async def waited(text, plan, pace_on, caller=""):
    """말소리가 끝난 뒤 조용한 조각을 몇 ms 받고 나서 듣기를 멈췄는가"""
    ws = FakeWs(plan + [(QUIET, 200)])
    heard = await server.listen(ws, FakeStt(text), pace_on, caller)
    return (ws.used - sum(n for f, n in plan)) * 20, heard


SLOW = "가나다라마바사아자차카타파하거너"                          # 16자
FAST = "가나다라마바사아자차카타파하거너가나다라마바사아자차카타파하거너"   # 32자
loop = asyncio.new_event_loop()
run = loop.run_until_complete

ms, h = run(waited(SLOW, [(LOUD, 250)], False));  assert ms == 800 and h["pace"] is None        # 끄면 기본 800ms
ms, h = run(waited(SLOW, [(LOUD, 250)], True));   assert ms == 960 and h["pace"]["cps"] == 3.2 and h["voiced_ms"] == 5000   # 느린 사람: 800 × 1.2
ms, h = run(waited(FAST, [(LOUD, 150)], True));   assert ms == 800                               # 빠른 사람(32자/3초)은 그대로
ms, h = run(waited("네.", [(LOUD, 100)], True));  assert ms == 800 and h["pace"]["cps"] is None  # 근거 부족이면 손대지 않음
ms, h = run(waited(FAST, [(LOUD, 75), (QUIET, 35), (LOUD, 75)], True))                           # 말 중간에 700ms 쉬었다 → 875ms
assert ms == 880 and h["pace"]["own_pause_ms"] == 700, (ms, h)                                  # (20ms 단위로 올림)
ms, h = run(waited(FAST, [(LOUD, 75), (QUIET, 10), (LOUD, 75)], True))                          # 200ms 끊김은 쉼이 아님
assert ms == 800 and h["pace"]["own_pause_ms"] == 0, (ms, h)
ms, h = run(waited("배송 확인하고 싶은데", [(LOUD, 250)], True)); assert ms == 1800 and h["pace"] is None   # 이어질 말투(1.8초 규칙)는 건드리지 않음
ms, h = run(waited("잘못 얘기했는데", [(LOUD, 250)], True));   assert ms == 2500 and h["pace"] is None   # 말 고치기(2.5초 규칙)도 그대로
server.PACE_PROFILES["누구"] = 4.0                                                                  # 이번 말로 못 재면 통화 누적값을 쓴다
ms, h = run(waited("네.", [(LOUD, 100)], True, "누구")); assert ms == 800 + 160 and h["pace"]["cps"] == 4.0, (ms, h)
loop.close()

# 5. 모드: 기본(live)은 측정기(debug 없음)에서 꺼진다
def on(mode, **start):
    server.PACE_ADAPT = mode
    return server.pace_enabled(start)
assert (on("live"), on("live", debug=True), on("all"), on("off", debug=True)) == (False, True, True, False)

print("전부 통과")
