# [전산실 자가 점검] lookup() 이 연습 문항의 실제 호출에 장부대로 답하는지 한 번에 확인합니다 (서버를 켜지 않아도 돼요).
# 왜 있나: lookup() 을 고칠 때마다 "예전에 되던 게 깨졌나?"를 바로 보려고요. 실행: python mock_tools/check_lookup.py
import json
from pathlib import Path

from mock_server import Store

ROOT = Path(__file__).resolve().parents[1]
store = Store(ROOT / "data" / "mock_db.json", ROOT / "data" / "brand" / "tools.json")
cases = [json.loads(l) for l in open(ROOT / "data" / "practice_cases.jsonl", encoding="utf-8")]

# 1) 연습 문항의 조회·검색 호출을 그대로 보내면, 장부에 있는 같은 인수의 결과가 돌아와야 한다
#    (장부에 같은 인수가 두 번 들어 있는 경우가 있어서, 그중 하나와 같으면 통과로 본다)
same = total = 0
for c in cases:
    for call in c["reference"]["tool_calls"]:
        if call["name"] in ("get_order_info", "get_return_info", "search_products"):
            total += 1
            expected = [r["result"] for r in store.db[call["name"]] if r["arguments"] == call["arguments"]]
            same += store.lookup(call["name"], call["arguments"])[0] in expected
print(f"조회·검색 {same}/{total}건이 실제 결과와 같음")
assert same == total, "일부 호출이 장부와 다른 결과를 돌려줌"

# 2) 표기만 다른 번호(하이픈·공백)도 같은 기록을 찾아야 한다
rec = next(r for r in store.db["get_return_info"] if r["result"]["status"] == "ok")
phone = rec["arguments"]["customer_phone_number"]
spaced = " ".join([phone[:3], phone[3:7], phone[7:]]) if phone.isdigit() else phone
assert store.lookup("get_return_info", {"customer_phone_number": spaced})[0] == rec["result"]

# 3) 없는 번호는 not_found, 빠진 값은 invalid, 접수는 값이 다 있으면 ok
assert store.lookup("get_order_info", {"order_id": "99999999-0000000"})[0]["status"] == "not_found"
assert store.lookup("get_order_info", {})[0]["status"] == "invalid"
assert store.lookup("cancel_order", {"order_id": "20260101-0000001"})[0]["status"] == "invalid"      # reason 이 빠짐
assert store.lookup("cancel_order", {"order_id": "20260101-0000001", "reason": "단순 변심"})[0]["status"] == "ok"
assert store.lookup("request_callback", {"customer_name": "홍길동", "phone_number": "010-1234-5678"})[0]["status"] == "ok"

# 4) 검색어가 달라도 가장 가까운 기록을 돌려준다
q = store.db["search_products"][0]["arguments"]["query"]
assert store.lookup("search_products", {"query": q + " 알려주세요"})[0] == store.db["search_products"][0]["result"]
print("전부 통과")
