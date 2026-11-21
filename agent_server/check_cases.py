# [유형 분류 자가 점검] 손님 말을 키워드로 유형(case)으로 나눠 '먼저 하는 말'을 고르는 pick_case 가 맞게 동작하는지 확인합니다.
# 왜 있나: 잘못 고른 매크로는 안 하는 것보다 나빠서, 키워드를 고칠 때마다 연습 문항 80개로 바로 돌려 보려고요. (API 는 부르지 않아요)
# 실행: python agent_server/check_cases.py        (80개 전체 결과를 보려면: python agent_server/check_cases.py -v)
import json, os, sys
from pathlib import Path

os.environ.setdefault("BAZE_API_KEY", "x")                  # 시험용: 실제 호출은 하지 않음
import server

for line in [p for c in server.CASES.values() for p in c[2]] + server.CLOSINGS:
    server.FIXED[server.phrase_key(line)] = b""             # 음성이 준비된 것으로 친다

kind = {p: name for name, c in server.CASES.items() for p in c[2]}
def case_of(text, **start):
    got = server.pick_case(text, {"history": [], **start})
    return kind[got[0]] if got else None

assert case_of("주문상품은 언제 배송되나요??") is None                       # 번호가 없으면 매크로 대신 LLM 이 번호를 여쭌다
assert case_of("주문상품은 언제 배송되나요??", caller={"이름": "x"}) == "주문조회"
assert case_of("01040270247 입금확인이 안되요") == "주문조회"
assert case_of("카드결재는안되나요? 카드로구매하고싶어요") == "정책"
assert case_of("하루온 간 한박스에 몇일분인가요?") == "상품"
assert case_of("환불   방법  문의") == "정책"
assert case_of("환불신청합니다", caller={"이름": "x"}) == "접수"
assert case_of("제품이 잘못 배송 되었습니다.", caller={"이름": "x"}) is None    # 오배송은 상담원 연결이라 매크로 없음
assert case_of("상담원 연결 환불 안됨", caller={"이름": "x"}) is None
assert case_of("안녕하세요") is None and case_of("") is None
assert server.pick_case("배송문의 01040270247", {"history": [{"role": "agent", "text": p} for p in server.CASES["주문조회"][2]]}) is None   # 같은 통화에서 한 말은 안 쓴다
assert server.with_closing("현재 배송 중입니다.", True).endswith(tuple(server.CLOSINGS))
assert server.with_closing("현재 배송 중입니다.", False) == "현재 배송 중입니다."
assert server.with_closing("주문번호를 말씀해 주시겠어요?", True) == "주문번호를 말씀해 주시겠어요?"        # 되묻는 답에는 끝맺음 없음

rows = [json.loads(l) for l in (Path(__file__).parents[1] / "data" / "practice_cases.jsonl").read_text(encoding="utf-8").splitlines()]
count = {}
for r in rows:
    c = case_of(r["customer_utterance"], history=r["history"], caller=r.get("customer_profile"))
    count[c] = count.get(c, 0) + 1
    if "-v" in sys.argv and c:
        print(f"{c:6} | {r['customer_utterance'][:60]!r}")
print("연습 문항 80개 유형 분포:", count)

# ── 일반 첫마디(pick_opener) ──
import jev
for line in server.FILLERS:
    server.FIXED[server.phrase_key(line)] = b""
op = server.pick_opener
got = op("배송비는 얼마예요?", {"history": []})
assert got and got[0] in server.FILLERS and got[1] is False
assert op("네", {"history": []}) is None and op("아니요", {"history": []}) is None                  # 짧은 대답
assert op("잘못 얘기했는데요 다시 말씀드릴게요", {"history": []}) is None                         # 말 고치는 중
assert op("약 먹고 두드러기가 났어요", {"history": []}) is None                                   # 조심할 말
assert op("배송비는 얼마예요?", {"history": []}, jev.Decision(top="handoff", prob=0.9)) is None
assert op("배송비는 얼마예요?", {"history": []}, jev.Decision(top="other", prob=0.9)) is None
server.OPENER = False; assert op("배송비는 얼마예요?", {"history": []}) is None; server.OPENER = True

# ── hold_ignored ──
for t in ("안녕하세요.", "아 넵 감사합니다", "01079044351", "010-4957-6966입니다.", "제 주문 배송이 언제 오나요?", "반품회수해갔다네요"):
    assert server.hold_ignored(t), t
for t in ("", "주문번호가 DEMO1001인데요.", "하루온 간~", "주문할려고 하는데 여", "010-4957", "잠시만요", "배송 확인하고 싶은데"):
    assert not server.hold_ignored(t), t
print("전부 통과")

