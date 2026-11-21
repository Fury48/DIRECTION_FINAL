# [Jev 자가 점검] Jev 연동(말 끝 판정·업무 분류)이 규칙대로 동작하는지 가짜 응답으로 확인합니다. (API·키·크레딧 필요 없음)
# 왜 있나: Jev 를 잘못 믿으면 고객 말을 끊어 속도 0점이 되고, Jev 가 죽어도 상담은 계속돼야 해서 고칠 때마다 바로 돌려 보려고요.
# 실행: python agent_server/check_jev.py        (진짜 Jev 로 한 번 불러 보려면: python agent_server/check_jev.py --live  ← TYPESAFE_API_KEY 필요)
import asyncio, os, sys, time
import numpy as np

from pathlib import Path
from dotenv import dotenv_values

# 진짜 키는 따로 보관해 둔다 (--live 에서 쓴다). 터미널에 설정한 값이 먼저, 없으면 .env 의 값.
JEV_KEYS = ("TYPESAFE_API_KEY", "TYPESAFE_URL", "TYPESAFE_MODEL", "BAZE_API_KEY", "BAZE_BASE_URL")
REAL = {k: os.environ[k] for k in JEV_KEYS if os.environ.get(k)}
for env_file in (Path(__file__).with_name(".env"), Path(__file__).resolve().parents[1] / ".env"):
    for k, v in dotenv_values(env_file, encoding="utf-8-sig").items():
        if k in JEV_KEYS and v and k not in REAL:
            REAL[k] = v.strip()


def clear_keys():
    """오프라인 점검은 '키가 없는 상태'에서 시작해야 한다. server 를 불러오면 .env 가 다시 읽혀 키가 도로 들어오니, 불러온 뒤에도 한 번 더 비운다."""
    for k in (*JEV_KEYS, "EOT_MODE", "CASE_ROUTER", "CASE_PROMPT"):
        os.environ.pop(k, None)


clear_keys()
os.environ["BAZE_API_KEY"] = "x"                              # server 를 불러오는 데 필요한 시험용 값: 실제 호출은 하지 않음
import jev, server
clear_keys()

# ── 1. 켜짐 조건과 호출 주소 ─────────────────────────────────────────
os.environ.pop("BAZE_API_KEY", None)
assert jev.target() == (None, None, None) and not any(jev.active(m, {"debug": True}) for m in ("auto", "jev", "live", "off"))   # 키가 없으면 어떤 모드든 꺼짐
os.environ["BAZE_API_KEY"] = "bkey"
assert jev.target() == ("https://api.baze.co/v1/gateway/decisions/", "bkey", "jev-1.14.0")                                  # 제출용: 게이트웨이
assert (jev.active("auto"), jev.active("jev"), jev.active("live"), jev.active("live", {"debug": True}), jev.active("off"), jev.active("silence")) \
    == (False, True, False, True, False, False)                                                                              # auto 는 TYPESAFE 키가 있을 때만
os.environ["TYPESAFE_API_KEY"] = " tkey "
assert jev.target() == ("https://api.typesafe.ai/v1/systemone", "tkey", "jev-1.14.0") and jev.active("auto")                # 키 한 줄이면 바로 켜짐 (직접 호출이 우선)
assert not jev.active("silence") and not jev.active("off")


# ── 가짜 HTTP: 보낸 요청을 기록하고, 정해 둔 규칙대로 답한다 ─────────────
class Resp:
    def __init__(self, body, code=200):
        self.body, self.status_code = body, code

    def raise_for_status(self):
        if self.status_code >= 400:
            err = RuntimeError(f"HTTP {self.status_code}")
            err.response = self
            raise err

    def json(self):
        return self.body


class FakeHttp:
    def __init__(self, reply):
        self.reply, self.sent = reply, []

    async def post(self, url, headers=None, json=None, timeout=None):
        self.sent.append((url, headers, json, timeout))
        await asyncio.sleep(0)
        out = self.reply(json)
        return out if isinstance(out, Resp) else Resp(out)


def choice(name, pick, prob=0.9, others=()):
    return {"answers": {name: {"type": "choice", "choice": pick, "confidence": prob,
                               "probabilities": {pick: prob, **{o: round((1 - prob) / max(len(others), 1), 3) for o in others}}}}}


def reset():
    jev._calls.clear()
    jev._state.update(fails=0, down_until=0.0)


loop = asyncio.new_event_loop()
run = loop.run_until_complete

# ── 2. ask(): 요청 모양, 실패 처리 ───────────────────────────────────
reset()
h = FakeHttp(lambda j: choice("turn", "end"))
ans = run(jev.ask(h, {"caller_transcript": "가"}, {"turn": jev.TURN_QUESTION}, 1.0))
url, headers, body, timeout = h.sent[0]
assert url.endswith("/v1/systemone") and headers == {"Authorization": "Bearer tkey"} and timeout == 1.0
assert set(body) == {"model", "state", "questions"} and body["model"] == "jev-1.14.0"                                         # 게이트웨이는 이 세 개 말고 다른 필드를 주면 400
assert body["questions"]["turn"]["type"] == "choice" and set(body["questions"]["turn"]["criteria"]) == {"end", "hold"} and ans["turn"]["choice"] == "end"

reset()
for code, rest in ((402, 300), (429, 30)):                                                                                    # 크레딧·키 문제는 5분, 한도는 30초 쉰다
    reset()
    bad = FakeHttp(lambda j, c=code: Resp({}, c))
    try:
        run(jev.ask(bad, {}, {}, 1.0))
        raise SystemExit("예외가 나야 합니다")
    except RuntimeError:
        pass
    assert abs(jev._state["down_until"] - time.monotonic() - rest) < 2, (code, jev._state)
    try:
        run(jev.ask(bad, {}, {}, 1.0))
        raise SystemExit("쉬는 동안은 부르지 않고 예외")
    except RuntimeError as e:
        assert "쉬는 중" in str(e) and len(bad.sent) == 1                                                                    # 쉬는 동안은 요청도 보내지 않는다
reset()
jev.MAX_PER_MIN = 3
ok = FakeHttp(lambda j: choice("turn", "end"))
for _ in range(3):
    run(jev.ask(ok, {}, {}, 1.0))
try:
    run(jev.ask(ok, {}, {}, 1.0))
    raise SystemExit("분당 상한을 넘으면 예외")
except RuntimeError as e:
    assert "상한" in str(e) and len(ok.sent) == 3
jev.MAX_PER_MIN = 60
reset()

# ── 3. 글 가리기 ─────────────────────────────────────────────────────
assert jev.mask("저는 010-4957-6966 이고요 주문번호 20260929-0033428 입니다") == "저는 <번호> 이고요 주문번호 <번호> 입니다"
assert jev.mask("조인플렉스 2개요 010") == "조인플렉스 2개요 010"                                                            # 짧은 숫자는 그대로
assert "4957" not in jev.dialogue_tail([{"role": "customer", "text": "010 4957 6966이에요"}, {"role": "tool_result", "text": "비밀"}]) \
    and "비밀" not in jev.dialogue_tail([{"role": "tool_result", "text": "비밀"}])


# ── 4. TurnJudge ─────────────────────────────────────────────────────

def tick(j, text, ms):                                         # 실제 서버에서는 listen() 안(실행 중인 루프)에서 부르므로, 시험도 루프 안에서 부르고 요청이 끝나길 잠깐 기다린다
    async def go():
        j.tick(text, ms)
        await asyncio.sleep(0.01)
    run(go())


def spec(r, text):
    async def go():
        r.speculate(text)
    run(go())


def judge_with(reply, history=None):
    reset()
    h = FakeHttp(reply)
    return jev.TurnJudge(h, history), h


async def settle():
    await asyncio.sleep(0.01)                                  # 띄워 둔 요청들이 끝나게 한다


T = "제가 주문한 상품이 언제 도착하나요?"
j, h = judge_with(lambda b: choice("turn", "end", 0.9, ("hold",)), [{"role": "agent", "text": "어떤 주문이세요?"}])
tick(j, T, 100); assert j.calls == 0                                                      # 무음 250ms 전에는 묻지 않는다
tick(j, T, 250); run(settle())
assert h.sent[0][2]["state"] == {"caller_transcript": T, "previous_agent_line": "어떤 주문이세요?", "silence_ms": 250}
assert j.verdict(T, 300) == "end"
assert j.verdict(T + " 감사합니다", 300) is None                                           # 글이 바뀌었으면 그 답은 쓰지 않는다 (end 는 같은 글만)
assert j.verdict(T, 250 + 700) is None                                                    # 물어본 뒤 무음이 650ms 넘게 길어지면 낡은 답
tick(j, T, 300); assert j.calls == 1                                                       # 같은 글은 200ms 가 지나야 다시 묻는다
tick(j, T, 450); run(settle()); assert j.calls == 2
j.resume(); assert j.verdict(T, 500) is None                                              # 다시 말하기 시작하면 이전 쉼의 답은 버린다

j, h = judge_with(lambda b: choice("turn", "end", 0.6, ("hold",)))                        # 확신이 낮으면(0.75 미만) 안 씀
tick(j, T, 250); run(settle()); assert j.verdict(T, 300) is None

for greeting in ("안녕하세요", "저기요", "주문번호가 DEMO1001인데요", "잠시만요", "배송 문의하고요 그리고"):   # end 라고 해도 글 모양이 이어질 말이면 믿지 않는다
    j, h = judge_with(lambda b: choice("turn", "end", 0.99, ("hold",)))
    tick(j, greeting, 250); run(settle())
    assert j.verdict(greeting, 300) is None and j.vetoed == 1, greeting

j, h = judge_with(lambda b: choice("turn", "hold", 0.9, ("end",)))
tick(j, "주문번호는 이십 이십 영", 250); run(settle())
assert j.verdict("주문번호는 이십 이십 영", 300) == "hold"
assert j.verdict("주문번호는 이십 이십 영구", 300) == "hold"                                  # 받아쓰기가 1~2자 따라온 글에도 hold 유지
assert j.verdict("주문번호는 이십 이십 영구이공 칠", 300) is None                              # 많이 달라졌으면 유지하지 않음

j, h = judge_with(lambda b: Resp({}, 500))                                                 # HTTP 오류 → 판정 없음(None), 예외 없이 계속
tick(j, T, 250); run(settle()); assert j.verdict(T, 300) is None and j.summary()["errors"] == 1
j, h = judge_with(lambda b: {"answers": {"turn": {"choice": "maybe", "probabilities": {"maybe": 0.99}}}})   # 모르는 선택지 → None
tick(j, T, 250); run(settle()); assert j.verdict(T, 300) is None
j, h = judge_with(lambda b: {"nonsense": 1})                                               # 잘못된 JSON 모양 → None
tick(j, T, 250); run(settle()); assert j.verdict(T, 300) is None and j.summary()["errors"] == 1

j, h = judge_with(lambda b: choice("turn", "hold"))                                       # 말 한 번에 최대 16번, 동시에 2개까지
async def burst():
    for i in range(40):
        j.tick(T + "가" * i, 250 + i)                                                       # 응답이 오기 전에 연달아 부르면 2개까지만 띄운다
    assert len(j.inflight) == 2 and j.calls == 2
    for i in range(40, 80):
        await asyncio.sleep(0.01)
        j.tick(T + "가" * i, 250 + i)
    await asyncio.sleep(0.01)
run(burst())
assert j.calls == jev.EOT_MAX_CALLS, j.calls


# ── 5. listen(): Jev 판정이 말 끝 대기를 바꾸는가 ────────────────────
LOUD, QUIET = np.full(320, 3000, np.int16).tobytes(), bytes(640)


class FakeStt(server.Stt):
    def __init__(self, text):
        self.text, self.partial = text, ""

    def feed(self, pcm):
        pass


class FakeWs:
    def __init__(self, plan):
        self.frames = [f for frame, n in plan for f in [frame] * n]
        self.used = 0

    def __aiter__(self):
        return self

    async def __anext__(self):
        if self.used >= len(self.frames):
            raise StopAsyncIteration
        self.used += 1
        await asyncio.sleep(0)                                  # Jev 요청(작업)이 진행되게 매 조각마다 양보한다
        return self.frames[self.used - 1]


def waited(text, reply, **kw):
    """말소리(2초) 뒤 조용해진 시간이 몇 ms 일 때 듣기를 멈췄는가 (20ms 단위)"""
    reset()
    plan = [(LOUD, 100)]
    ws, h = FakeWs(plan + [(QUIET, 300)]), FakeHttp(reply)
    judge = jev.TurnJudge(h, None)
    heard = run(server.listen(ws, FakeStt(text), judge=judge, **kw))
    return (ws.used - 100) * 20, heard, h


end = lambda b: choice("turn", "end", 0.95, ("hold",))
hold = lambda b: choice("turn", "hold", 0.95, ("end",))
up = lambda ms: -(-ms // 20) * 20                                                           # 20ms 조각 단위로 올림
U = T.rstrip("?")                                                                            # 문장부호 없는 글: 이때만 Jev 판정을 쓴다
saved_end, jev.EOT_END_MS = jev.EOT_END_MS, 650                                              # Jev end 가 대기를 줄이는지 보려고 기본(=END_SILENCE_MS)보다 짧게 둔다
ms, heard, h = waited(U, end);                       assert ms == up(jev.EOT_END_MS) and heard["eot"]["end"] > 0, ms     # Jev end → EOT_END_MS
ms, heard, h = waited(U, hold);                      assert ms == jev.EOT_HOLD_MS, ms                                      # Jev hold → 2.5초
ms, heard, h = waited(U, lambda b: Resp({}, 500));   assert ms == server.END_SILENCE_MS and heard["eot"]["errors"] > 0     # Jev 오류 → 예전 규칙
ms, heard, h = waited(U, lambda b: choice("turn", "end", 0.5, ("hold",)));  assert ms == server.END_SILENCE_MS             # 확신 낮음 → 예전 규칙
ms, heard, h = waited(T, hold, live=False);          assert ms == server.PUNCT_SILENCE_MS, ms                               # 문장부호로 끝난 말은 Jev hold 를 따르지 않음 (검토 1절)
ms, heard, h = waited(T, end, live=False);           assert ms == server.PUNCT_SILENCE_MS, ms                               # 문장부호로 끝난 말은 Jev end 로 줄이지도 않음
ms, heard, h = waited("배송 확인하고 싶은데", end);  assert ms == server.LONG_SILENCE_MS, ms                                   # Jev 가 end 여도 '~는데'로 끝나면 규칙(1.8초)
ms, heard, h = waited("잘못 얘기했는데", end);       assert ms == server.CORRECT_SILENCE_MS, ms                                # 말 고치기 시작 말은 2.5초
ms, heard, h = waited("아니,", end);                 assert ms == server.CORRECT_SILENCE_MS, ms                                # 정정하는 말뿐이면 2.5초
ms, heard, h = waited("010", end);                   assert ms >= server.LONG_SILENCE_MS, ms                                   # 번호를 말하다 쉬는 중에는 Jev end 를 따르지 않음
ms, heard, h = waited(U, end, pace_on=True);         assert ms == up(jev.EOT_END_MS)                                          # Jev 가 있으면 말 속도 조절은 쓰지 않음
jev.EOT_END_MS = saved_end
reset()
ws = FakeWs([(LOUD, 100), (QUIET, 20), (LOUD, 50), (QUIET, 300)])                                                           # 쉬는 중에 던진 질문의 답은, 다시 말하면 버린다
judge = jev.TurnJudge(FakeHttp(end), None)
run(server.listen(ws, FakeStt(T), judge=judge))
assert judge.epoch == 1                                                                                                    # 다시 말한 횟수만큼 resume


# ── 6. 업무 분류 ─────────────────────────────────────────────────────
def case_reply(pick, prob=0.9):
    return lambda b: choice("case", pick, prob, [c for c in jev.CASE_SPEC if c != pick][:2])


reset()
r = jev.CaseRouter(FakeHttp(case_reply("refund_status")), [{"role": "customer", "text": "010 4957 6966"}])
spec(r, "반품 진행 상황이 궁금해서요"); assert not r.attempts                         # 처음 본 글은 바로 묻지 않고, 안정될 때까지 본다
time.sleep(0.12); spec(r, "반품 진행 상황이 궁금해서요"); assert len(r.attempts) == 1
spec(r, "반품 진행 상황이 궁금해서요"); assert len(r.attempts) == 1                    # 같은 글은 다시 묻지 않는다
d = run(r.resolve("반품 진행 상황이 궁금해서요."))                                          # 문장부호만 달라도 같은 글: 미리 낸 결과를 쓴다
assert d.early and d.case_id == "refund_status" and d.macro == "반품현황" and d.name == "반품·환불 진행 상황"
assert "진행 상황" in d.guide and d.guide.startswith("반품·환불 진행 상황")                   # scenarios.md 의 7번 구역
sent = r.http.sent[0][2]
assert set(sent["questions"]) == {"case"} and set(sent["questions"]["case"]["criteria"]) == set(jev.CASE_SPEC) and "4957" not in str(sent["state"])
r.close()

reset()
r = jev.CaseRouter(FakeHttp(case_reply("delivery")), None)                                  # 미리 낸 게 없으면 확정 뒤에 새로 묻는다
d = run(r.resolve("배송 언제 오나요?")); assert not d.early and d.case_id == "delivery" and d.macro == "주문조회"
d = run(jev.CaseRouter(FakeHttp(case_reply("delivery", 0.4)), None).resolve("배송?")); assert d.case_id is None and d.top == "delivery" and d.macro is None and d.guide == ""   # 확신이 낮으면 안 씀
d = run(jev.CaseRouter(FakeHttp(lambda b: Resp({}, 500)), None).resolve("배송?")); assert d.case_id is None and d.error == "RuntimeError"
reset()


class Slow(FakeHttp):
    async def post(self, *a, **k):
        await asyncio.sleep(1)
        return await super().post(*a, **k)


r = jev.CaseRouter(Slow(case_reply("delivery")), None)
d = run(r.resolve("배송?", wait_s=0.05)); assert d.error == "timeout" and d.case_id is None
r.close(); run(asyncio.sleep(0))                                                                                            # 늦는 요청은 정리한다
reset()
jev.CASE_PROMPT = "off"; assert jev.Decision(top="cancel", prob=0.9).guide == ""; jev.CASE_PROMPT = "add"                    # CASE_PROMPT=off 면 지침을 붙이지 않음
assert all(jev.GUIDES.get(spec[1]) for spec in jev.CASE_SPEC.values() if spec[1]), "scenarios.md 의 번호별 구역을 못 찾음"        # 업무마다 지침 구역이 있다
for cid, spec in jev.CASE_SPEC.items():
    assert spec[2] is None or spec[2] in server.CASES, cid                                                                  # 매크로 유형 이름이 server.CASES 와 맞는다

# ── 7. 매크로 거부권 ─────────────────────────────────────────────────
for line in [p for c in server.CASES.values() for p in c[2]] + server.CLOSINGS:
    server.FIXED[server.phrase_key(line)] = b""
start = {"history": [], "caller": {"이름": "x"}}
D = lambda cid: jev.Decision(top=cid, prob=0.9)
assert server.pick_case("환불신청합니다", start) is not None                              # Jev 없이는 키워드만 (예전과 같음)
assert server.pick_case("환불신청합니다", start, D("refund")) is not None                  # Jev 도 접수 계열이면 그대로
assert server.pick_case("환불신청합니다", start, D("handoff")) is None                     # 상담원 연결이면 매크로 없음
assert server.pick_case("환불신청합니다", start, D("misdelivery")) is None
assert server.pick_case("환불신청합니다", start, D("product")) is None                     # 키워드와 다른 유형이면 거부
assert server.pick_case("환불신청합니다", start, D("other")) is not None                   # '기타'는 거부권이 없다 (키워드가 정한다)
assert server.pick_case("환불신청합니다", start, jev.Decision(error="timeout")) is not None   # 실패한 분류는 없는 것과 같다
assert server.pick_case("환불신청합니다", start, jev.Decision(top="refund", prob=0.3)) is not None   # 확신이 낮은 분류도 없는 것과 같다
assert "[이번 통화의 업무: 반품·환불 요청]" in server.system_prompt(start, "", D("refund")) and "[이번 통화의 업무" not in server.system_prompt(start, "")

loop.close()
print("전부 통과")

if "--live" in sys.argv:                                       # 진짜 Jev 로 한 번 불러 본다 (키 필요, 아주 적은 비용)
    async def live():
        import httpx
        clear_keys()                                           # 점검에 쓴 가짜 키(tkey, bkey, x)를 지우고, 맨 위에서 보관해 둔 진짜 키로 바꾼다
        os.environ.update(REAL)
        reset()                                                # 점검 중에 쌓인 호출 기록·쉬는 상태도 비운다
        url, key, model = jev.target()
        if not key:
            raise SystemExit("TYPESAFE_API_KEY (또는 BAZE_API_KEY) 가 없어요. agent_server/.env 에 적어 주세요.")
        print(f"진짜 Jev 호출 — 주소: {url} | 모델: {model} | 키: {'TYPESAFE_API_KEY' if os.environ.get('TYPESAFE_API_KEY') else 'BAZE_API_KEY'} (값은 출력하지 않음)")
        ok = True
        async with httpx.AsyncClient() as http:
            for text, sil in (("주문번호가 이십 이십 영구이공인데요", 450), (T, 450)):
                t = time.perf_counter()
                try:
                    a = (await jev.ask(http, {"caller_transcript": text, "previous_agent_line": "", "silence_ms": sil}, {"turn": jev.TURN_QUESTION}, 5))["turn"]
                    print(f"  말 끝: {text!r} → {a['choice']} {a['probabilities']} ({(time.perf_counter() - t) * 1000:.0f}ms)")
                except Exception as e:                         # 실패해도 이유를 보여 준다 (HTTP 상태와 응답 앞부분)
                    r, ok = getattr(e, "response", None), False
                    print(f"  말 끝 실패: {type(e).__name__}: {e}" + (f" | HTTP {r.status_code} | {r.text[:200]}" if r is not None else ""))
            for text in ("배송 언제 오나요", "약 먹고 두드러기가 났는데 환불해 주세요", "안녕하세요"):
                d = await jev.classify(http, text, None)
                ok = ok and not d.error
                print(f"  업무: {text!r} → {d.short()}")
        print("진짜 Jev 호출 점검:", "통과" if ok else "실패 (위 오류를 확인하세요)")
    asyncio.run(live())
