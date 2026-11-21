# [번호 가리기 자가 점검] 게이트웨이가 번호를 [PHONE] 으로 가려 버리는 문제를 우리가 피하는 장치(자리표시, 한글 숫자)가 맞게 동작하는지 확인합니다.
# 왜 있나: 번호가 틀리면 함수 점수가 0 이라서, server.py 의 Masker·for_tts 를 고칠 때마다 바로 돌려 보려고요. (API 는 부르지 않아요)  실행: python agent_server/check_mask.py
import os

os.environ.setdefault("BAZE_API_KEY", "x")                  # 시험용: 게이트웨이 모드로 불러온다 (실제 호출은 하지 않음)
import server

m = server.Masker()
text = "기진서 010 4957 6966, 01040270247성건경, 주문 20261001-0005987 상세 20261001-0005987-01 송장 500212345678"
masked = m.mask(text)
assert masked == "기진서 <전화1>, <전화2>성건경, 주문 <주문1> 상세 <주문2> 송장 500212345678", masked
assert m.mask("다시 010-4957-6966") == "다시 <전화1>"                  # 같은 번호는 같은 자리표시
assert m.unmask('{"buyer_cellphone":"<전화1>","order_id":"<주문1>"}') == '{"buyer_cellphone":"01049576966","order_id":"20261001-0005987"}'

msgs = [{"role": "assistant", "content": None, "tool_calls": [{"id": "h1", "type": "function", "function": {"name": "get_order_info", "arguments": '{"buyer_cellphone": "01049576966"}'}}]},
        {"role": "tool", "tool_call_id": "h1", "content": '{"status":"ok","data":{"orders":[{"주문번호":"20261001-0005987"}]}}'}]
out = server.mask_messages(server.Masker(), msgs)
assert "<전화1>" in out[0]["tool_calls"][0]["function"]["arguments"] and "<주문1>" in out[1]["content"]
assert "01049576966" in msgs[0]["tool_calls"][0]["function"]["arguments"]                              # 원본은 그대로

assert server.for_tts("번호는 010-4957-6966입니다.") == "번호는 공 일 공, 사 구 오 칠, 육 구 육 육입니다."
assert server.for_tts("주문 20261001-0005987") == "주문 이 공 이 육 일 공 공 일, 공 공 공 오 구 팔 칠"
assert server.for_tts("가격은 29,900원입니다.") == "가격은 29,900원입니다."                              # 번호가 아닌 숫자는 그대로
print("전부 통과")
