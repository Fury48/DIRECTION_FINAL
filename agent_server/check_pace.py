# [말 속도 자가 점검] 말하는 속도에 맞춘 ① 말 끝 대기 늘리기 ② 끼어들기에서 쉼 배우기 ③ 상담원 음성 재생 속도 조절이 규칙대로 동작하는지 확인합니다. (API 는 부르지 않아요)
# 왜 있나: 이 기능은 늘리기만 하고 줄이면 안 되며, 채점(합성 음성·질문 하나)에는 영향이 없어야 해서 고칠 때마다 바로 돌려 보려고요. (설계: pace.md, tts_speed_and_pace_handoff.md)
# 실행: python agent_server/check_pace.py
import asyncio, os, time
import numpy as np

os.environ.setdefault("BAZE_API_KEY", "x")                  # 시험용: 실제 호출은 하지 않음
os.environ["END_SILENCE_MS"] = "800"                         # 아래 기대값은 기본 대기 800ms 기준으로 계산했다 (배율·상한 규칙을 보는 시험이라 기본값과 무관)
import server

assert server.PACE_REF_CPS == 5.3 and server.PACE_MAX_FACTOR == 1.2 and server.PACE_PLAY_MIN_SPEED == 0.85, "기본값이 바뀌었어요 (환경변수 PACE_* 를 지우고 다시 돌려 주세요)"

# 1. 발음 속도: 근거가 부족하면 None
assert server.turn_rate("가나다라마바사아", 1000) is None                    # 소리가 1.5초 미만
assert server.turn_rate("가나다라", 3000) is None                           # 글자 8자 미만
assert server.turn_rate("공일공 사구오칠 0104957696", 3000) is None          # 번호 위주(숫자가 글자의 절반 초과)
assert abs(server.turn_rate("가나다라마바사아자차카타파하거너", 5000) - 3.2) < 1e-9   # 16자 / 5초
assert abs(server.turn_rate("가나다라마바사아 자차카타, 파하거너. 12", 5000) - 3.2) < 1e-9   # 숫자·공백·문장부호는 글자 수에서 뺀다

# 2. 대기 조절: 빠른 사람은 그대로, 느린 사람은 최대 1.2배, 줄이지 않는다, 본인 쉼은 속도 상한과 따로 (시연 +600ms / 측정 +300ms)
w = server.pace_wait
assert w(600, 6.1) == 600 and w(600, 5.3) == 600
assert abs(w(600, 4.8) - 600 * 5.3 / 4.8) < 1e-9                            # 약 0.66초
assert abs(w(600, 4.4) - 720) < 1e-9 and abs(w(600, 2.0) - 720) < 1e-9      # 속도 상한 1.2배
assert abs(w(600, 6.1, own_pause_ms=500) - 625) < 1e-9                      # 본인 쉼 0.5초 → 0.625초
assert abs(w(800, 6.1, own_pause_ms=780) - 975) < 1e-9                      # 지난번 0.78초 쉬고 이어 말함 → 0.975초 (속도 상한 0.96초에 묶이지 않음)
assert abs(w(800, 6.1, own_pause_ms=2000) - 1400) < 1e-9                    # 배운 쉼이 아주 길어도 기본 + 0.6초까지
assert abs(w(800, 6.1, own_pause_ms=2000, live=False) - 1100) < 1e-9        # 측정 모드는 기본 + 0.3초까지
assert w(800, None) == 800
assert all(w(800, r, p, lv) >= 800 for r in (None, 1.0, 3.0, 5.3, 9.0) for p in (0, 100, 500, 2000, 9000) for lv in (True, False))   # 어떤 경우에도 줄지 않는다

# 3. 통화 기록: 이상값은 버리고 지수 평균, 쉼은 200~4000ms 만 최근 5개, 이전 기록이 있어야 ready
p = server.Profile()
assert not p.ready() and p.pause_hint() == 0
p.note_rate(5.0); p.note_rate(4.0)
assert abs(p.rate - (0.6 * 5.0 + 0.4 * 4.0)) < 1e-9 and p.ready()
p.note_rate(0.5); p.note_rate(30); p.note_rate(None)
assert abs(p.rate - 4.6) < 1e-9
q = server.Profile()
for ms in (150, 5000, 300, 400, 500, 600, 700, 800):
    q.note_pause(ms)
assert list(q.pauses) == [400.0, 500.0, 600.0, 700.0, 800.0] and q.pause_hint() == 800 and q.ready()    # 150·5000 은 버리고 최근 5개
server.PACE_PROFILES.clear()
assert server.profile_for({}) is None and server.profile_for({"call_id": ""}) is None and server.profile_for({"caller": {"이름": "x"}}) is None   # 통화 번호가 없으면(채점기) 기록도 적용도 없다
a = server.profile_for({"call_id": "C1"})
assert a is server.profile_for({"call_id": "C1"}) and a is not server.profile_for({"call_id": "C2"})
for i in range(300):
    server.profile_for({"call_id": f"X{i}"})
assert len(server.PACE_PROFILES) <= 200                                      # 메모리가 무한히 늘지 않는다

# 4. 재생 속도: 느리게만(0.85~1.0), 이전 말의 기록이 있을 때만, 시연에서만
def speed(rate, live=True):
    pr = server.Profile(); pr.rate = rate
    return server.play_speed(pr, live)
assert server.play_speed(None) == 1.0 and server.play_speed(server.Profile()) == 1.0     # 기록이 없으면(첫 말·채점) 1.0
assert speed(6.1) == 1.0 and speed(5.3) == 1.0                                          # 빠른 사람은 그대로 (빠르게 하지 않는다)
assert speed(4.8) == 0.9 and speed(4.4) == 0.85 and speed(2.0) == 0.85                  # 0.05 단위, 0.85 아래로는 안 내려감
assert speed(4.0, live=False) == 1.0                                                     # 측정 모드는 늘 1.0
server.PACE_PLAY = False; assert speed(4.0) == 1.0; server.PACE_PLAY = True               # 끄면 1.0

# 5. 음높이 유지 속도 변환(WSOLA)
sr, f0 = server.RATE, 200
tone = (np.sin(2 * np.pi * f0 * np.arange(int(sr * 1.5)) / sr) * 8000).astype(np.int16).tobytes()      # 1.5초 200Hz
for sp in (0.85, 0.9):
    out = server.time_stretch(tone, sp)
    assert len(out) // 2 == round(len(tone) // 2 / sp), (sp, len(out))                                  # 길이가 정확히 입력 ÷ 속도
    a = np.frombuffer(out, dtype=np.int16).astype(np.float32)
    peak = np.argmax(np.abs(np.fft.rfft(a * np.hanning(len(a))))) * sr / len(a)
    assert abs(peak - f0) < f0 * 0.03, (sp, peak)                                                       # 음높이가 그대로
assert server.time_stretch(tone, 1.0) == tone and server.time_stretch(b"", 0.9) == b"" and server.time_stretch(tone[:1000], 0.9) == tone[:1000]   # 1.0배·빈 음성·0.1초 미만은 그대로
assert len(server.time_stretch(tone + b"\x01", 0.9)) % 2 == 0                                            # 홀수 바이트도 오류 없이
speech = (np.random.RandomState(0).randn(sr * 10) * 3000).astype(np.int16).tobytes()                    # 10초 문장
t0 = time.perf_counter(); server.time_stretch(speech, 0.85); assert time.perf_counter() - t0 < 0.8      # 첫 소리를 늦출 만큼 느리지 않다 (보통 0.02~0.1초)
server.PACED.clear()
calls = []
real_stretch = server.time_stretch
server.time_stretch = lambda pcm, sp, rate=sr: (calls.append(sp), real_stretch(pcm, sp))[1]
assert server.paced(tone, 1.0) is tone and not calls                                                     # 1.0배는 변환하지 않는다
x1, x2 = server.paced(tone, 0.9), server.paced(tone, 0.9)
assert x1 == x2 and len(calls) == 1                                                                      # 같은 문장은 한 번만 변환
server.time_stretch = lambda *a, **k: 1 / 0
assert server.paced(tone, 0.85) is tone                                                                  # 변환이 실패하면 원래 음성
server.time_stretch = real_stretch

# 6. listen(): 합성 소리로 '조용해진 뒤 몇 ms 만에 말이 끝났다고 보는지' 잰다 (20ms 조각 하나 = 20ms)
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


loop = asyncio.new_event_loop()
run = loop.run_until_complete


def waited(text, plan, pace_on, profile=None, live=True):
    """말소리가 끝난 뒤 조용한 조각을 몇 ms 받고 나서 듣기를 멈췄는가"""
    ws = FakeWs(plan + [(QUIET, 200)])
    heard = run(server.listen(ws, FakeStt(text), pace_on, profile, live=live))
    return (ws.used - sum(n for f, n in plan)) * 20, heard


def prof(rate=None, *pauses):
    pr = server.Profile()
    pr.rate = rate
    for ms in pauses:
        pr.note_pause(ms)
    return pr


SLOW = "가나다라마바사아자차카타파하거너"                          # 16자
FAST = SLOW + SLOW                                               # 32자
ms, h = waited(SLOW, [(LOUD, 250)], False, prof(4.0));            assert ms == 800 and h["pace"] is None and h["silent_ms"] == 800   # 끄면 기본 800ms
ms, h = waited(SLOW, [(LOUD, 250)], True, None);                  assert ms == 800 and h["pace"] is None                              # 통화 기록이 없으면(채점기) 적용하지 않음
ms, h = waited(SLOW, [(LOUD, 250)], True, prof());                assert ms == 800 and h["pace"]["cps"] == 3.2 and h["pace"]["ready"] is False   # 이 통화의 첫 말은 느려도 기록만
ms, h = waited(SLOW, [(LOUD, 250)], True, prof(5.0));             assert ms == 960 and h["pace"]["ready"], ms                         # 이전 기록이 있으면 느린 사람은 800 × 1.2
ms, h = waited(FAST, [(LOUD, 150)], True, prof(5.0));             assert ms == 800                                                    # 빠른 사람(32자/3초)은 그대로
ms, h = waited("배송비요", [(LOUD, 100)], True, prof(4.0));          assert ms == 960 and h["pace"]["cps"] == 4.0                         # 이번 말로 못 재면 통화 누적 속도
ms, h = waited("배송비요", [(LOUD, 100)], True, prof());             assert ms == 800                                                    # 근거가 전혀 없으면 손대지 않음
ms, h = waited(FAST, [(LOUD, 150)], True, prof(None, 780));       assert ms == 980 and h["pace"]["own_pause_ms"] == 780, (ms, h)       # 지난번에 0.78초 쉬다 끼어들렸다 → 0.975초 (20ms 단위로 올림)
ms, h = waited(FAST, [(LOUD, 150)], True, prof(None, 2000));      assert ms == 1400, ms                                               # 쉼이 아주 길어도 기본 + 0.6초
ms, h = waited(FAST, [(LOUD, 150)], True, prof(None, 2000), live=False);   assert ms == 1100, ms                                    # 측정 모드는 + 0.3초
ms, h = waited(FAST, [(LOUD, 75), (QUIET, 35), (LOUD, 75)], True, prof(5.0));  assert ms == 880 and h["pace"]["own_pause_ms"] == 700   # 이번 말 중간에 0.7초 쉬었다 → 875ms
ms, h = waited(FAST, [(LOUD, 75), (QUIET, 10), (LOUD, 75)], True, prof(5.0));  assert ms == 800 and h["pace"]["own_pause_ms"] == 0     # 0.2초 끊김은 쉼이 아님
ms, h = waited("배송 확인하고 싶은데", [(LOUD, 250)], True, prof(4.0));   assert ms == server.LONG_SILENCE_MS and h["pace"] is None   # 이어질 말투(1.8초 규칙)는 건드리지 않음
ms, h = waited("아니,", [(LOUD, 250)], True, prof(4.0));                 assert ms == server.CORRECT_SILENCE_MS and h["pace"] is None   # 말 고치기(2.5초 규칙)도 그대로
ms, h = waited("대장영양제.", [(LOUD, 250)], True, prof(4.0), live=False);   assert ms == server.NOUN_SILENCE_MS and h["pace"] is None      # 명사 끝(2.0초)·쉼표·인사 규칙도 그대로
ms, h = waited("배송 언제 와요?", [(LOUD, 250)], True, prof(4.0), live=False);   assert ms == 1200 and h["pace"]["ready"], ms   # 문장부호 끝(1.0초)은 빠른 규칙이라 말 속도 조절 대상: 1000 × 1.2


# 7. drain(): 말 끝으로 확정한 뒤 고객이 곧바로 이어 말하면 그 쉼을 배운다
def learn(plan, silent_ms=800, profile=True):
    pr = server.Profile() if profile else None
    run(server.drain(FakeWs(plan), pr, silent_ms))
    return list(pr.pauses) if pr else None


assert learn([(QUIET, 10), (LOUD, 10)]) == [1000.0]                                  # 800ms 쉰 뒤 + 0.2초 더 지나 말하기 시작 → 1.0초를 쉬던 사람
assert learn([(LOUD, 10)]) == [800.0]                                                # 곧바로 이어 말함
assert learn([(QUIET, 10), (LOUD, 1), (QUIET, 100)]) == []                           # 잡음 한 조각은 거른다
assert learn([(QUIET, 80), (LOUD, 20)]) == []                                        # 2초가 지난 뒤의 말은 답하는 말을 끊은 것이라 배우지 않는다
assert learn([(QUIET, 10), (LOUD, 10), (QUIET, 10), (LOUD, 10)]) == [1000.0]         # 처음 한 번만 배운다
assert learn([(QUIET, 10), (LOUD, 10)], profile=False) is None                       # 기록이 없으면(채점기) 아무것도 하지 않음 (예외도 없음)

# 8. say(): 재생 속도가 1.0 미만이면 문장마다 느리게 만들고 문장 사이 쉼을 늘린다
TONE = tone[: sr * 2 * 6 // 10]                                                      # 0.6초짜리 문장 음성


async def fake_chunks(text):
    yield TONE


class SinkWs:
    def __init__(self):
        self.audio, self.texts = [], []

    async def send(self, m):
        (self.audio.append(m) if isinstance(m, bytes) else self.texts.append(m))


real_chunks, real_tts_stream = server.speak_chunks, server.TTS_STREAM
server.speak_chunks, server.TTS_STREAM = fake_chunks, False


def spoken(speed_value):
    ws = SinkWs()
    server.PLAY_SPEED.set(speed_value)
    run(server.say(ws, "첫째 문장입니다. 둘째 문장입니다."))
    server.PLAY_SPEED.set(1.0)
    return ws


normal, slow = spoken(1.0), spoken(0.85)
n_normal, n_slow = len(b"".join(normal.audio)) / 32000, len(b"".join(slow.audio)) / 32000
assert abs(n_normal - 1.2) < 0.03, n_normal                                          # 1.0배: 0.6초 × 2 (쉼 없음, 예전과 같음)
assert abs(n_slow - (2 * 0.6 / 0.85 + 0.15 / 0.85)) < 0.05, n_slow                    # 0.85배: 각 문장 0.706초 + 문장 사이 쉼 0.176초 = 약 1.59초
assert all(len(f) == server.FRAME for f in slow.audio) and len(slow.texts) == 1       # 20ms 조각으로 보내고, 글(agent.transcript)은 한 번만
server.speak_chunks, server.TTS_STREAM = real_chunks, real_tts_stream
loop.close()

print("전부 통과")
