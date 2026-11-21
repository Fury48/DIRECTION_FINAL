# [가짜 주문 전산실] 상담원이 "이 주문 어디쯤이에요?" 하고 물어보러 오는 HTTP 서버입니다 (8100번).
# 왜 있나: 진짜 쇼핑몰 시스템 대신, 장부(data/mock_db.json)를 찾아 정해진 답을 돌려주려고요. 최종 측정 때도 같이 켜 둡니다.
# 응답을 고르는 lookup() 이 핵심: 번호로 찾고(조회), 빠진 값을 검사하고(접수), 비슷한 검색어를 고릅니다(상품 검색).
"""모의 함수 서버 — 쇼핑몰 시스템 대신 함수(주문 조회·반품 접수 등)에 응답하는 서버.

HTTP 주소 처리와 응답을 고르는 lookup() 이 모두 구현되어 있다. (원래 뼈대에서는 lookup() 이 비어 있었다)
호출 기록과 응답 지연은 측정기(run_eval.py)의 함수 중계가 맡으므로 여기서 할 필요가 없다.
이 파일을 고쳐 써도 되고, 다른 언어로 새로 만들어도 된다. 지켜야 할 규칙은 SPEC.md 의 「모의 함수 서버」에 있다.

실행:  python3 mock_server.py --port 8100 [--db ../data/mock_db.json]

  GET  /tools                 함수 목록 (OpenAI tools 형식, data/brand/tools.json)
  POST /tools/{함수명}         body = 인수 JSON. 헤더 X-Session-Id 에 session.start 로 받은 session_id 가 온다.
  GET  /calls?session_id=...  (참고용) 이 서버가 받은 호출 기록. 채점에는 쓰지 않는다.
"""
import argparse, json, os, re, sys, threading, time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse, parse_qs

HERE = os.path.dirname(os.path.abspath(__file__))
ID_KEYS = ("order_id", "buyer_cellphone", "customer_phone_number")     # 조회 함수가 받는 번호 인수
OK_MESSAGE = {"cancel_order": "주문 취소 접수 성공", "return_order": "주문 반품 접수 성공", "return_order_item": "상품 반품 접수 성공",
              "exchange_order": "교환 접수 성공", "request_callback": "고객 정보 접수 성공"}


def digits(v):
    """번호 비교용: 하이픈·공백 같은 표기 차이를 지우고 숫자만 남긴다."""
    return re.sub(r"\D", "", str(v)) if v is not None else ""


def bigrams(s):
    s = re.sub(r"\s", "", s)
    return {s[i:i + 2] for i in range(len(s) - 1)} or {s}


def order_numbers(result):
    """조회 결과 안에 든 주문번호들 (주문번호로 물었는데 장부에는 전화번호로 저장된 경우를 찾기 위해)."""
    data = result.get("data")
    return {digits(o.get("주문번호")) for o in data.get("orders", [])} if isinstance(data, dict) else set()


class Store:
    def __init__(self, db_path, tools_path):
        self.db = json.load(open(db_path, encoding="utf-8")) if os.path.exists(db_path) else {}   # {함수명: [{"arguments", "result"}]}
        self.tools = json.load(open(tools_path, encoding="utf-8"))
        self.calls = []
        self.lock = threading.Lock()

    def lookup(self, name, args):
        """(응답 JSON, 장부와 맞았는지 bool) 을 돌려준다. self.db[name] 에는 실제 상담에서 나온 (인수, 결과) 쌍이 들어 있다.

        - 필수 값이 빠졌으면 {"status": "invalid"}
        - 조회(주문·반품 현황): 전화번호·주문번호가 맞는 기록의 결과. 없으면 {"status": "not_found"}
        - 상품 검색: 검색어가 정확히 같지 않아도 가장 가까운 기록의 결과
        - 접수(취소·반품·교환·전화 요청): 필수 값이 다 있으면 {"status": "ok"}
        """
        records = self.db.get(name, [])
        required = next(t for t in self.tools if t["name"] == name)["parameters"].get("required", [])
        missing = [k for k in required if args.get(k) in (None, "", [], {})]
        if name == "get_order_info" and not (digits(args.get("order_id")) or digits(args.get("buyer_cellphone"))):
            missing = ["order_id 또는 buyer_cellphone"]
        if missing:
            return {"status": "invalid", "message": f"필수 값이 없습니다: {', '.join(missing)}"}, False

        if name in ("get_order_info", "get_return_info"):
            wanted = {d for k in ID_KEYS if (d := digits(args.get(k)))}
            hits = lambda r: len(wanted & ({digits(r["arguments"].get(k)) for k in ID_KEYS} | order_numbers(r["result"])))
            best = max(records, key=hits, default=None)
            if best and hits(best):
                return best["result"], True
            return {"status": "not_found", "message": "일치하는 기록이 없습니다."}, False

        if name == "search_products":
            q = bigrams(args["query"])
            sim = lambda r: 2 * len(q & bigrams(r["arguments"].get("query", ""))) / (len(q) + len(bigrams(r["arguments"].get("query", ""))))
            best = max(records, key=sim, default=None)
            if best and sim(best) > 0:
                return best["result"], True
            return {"status": "not_found", "message": "검색 결과가 없습니다."}, False

        return {"status": "ok", "message": OK_MESSAGE.get(name, "접수 성공")}, any(r["arguments"] == args for r in records)


def make_handler(store):
    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def _send(self, code, obj):
            b = json.dumps(obj, ensure_ascii=False).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(b)))
            self.end_headers()
            self.wfile.write(b)

        def do_GET(self):
            u = urlparse(self.path)
            if u.path == "/tools":
                return self._send(200, [{"type": "function", "function": t} for t in store.tools])
            if u.path == "/calls":
                cid = parse_qs(u.query).get("session_id", [None])[0]
                with store.lock:
                    return self._send(200, [c for c in store.calls if cid is None or c["session_id"] == cid])
            self._send(404, {"error": "not found"})

        def do_POST(self):
            m = re.fullmatch(r"/tools/(\w+)", urlparse(self.path).path)
            if not m or m.group(1) not in {t["name"] for t in store.tools}:
                return self._send(404, {"error": "unknown function"})
            name = m.group(1)
            t0 = time.time()
            try:
                args = json.loads(self.rfile.read(int(self.headers.get("Content-Length") or 0)) or b"{}")
            except json.JSONDecodeError:
                return self._send(400, {"error": "body must be JSON"})
            result, matched = store.lookup(name, args)
            # 참고용 호출 기록 (디버깅용)
            rec = {"session_id": self.headers.get("X-Session-Id"), "name": name, "arguments": args, "matched": matched,
                   "t_request": t0, "t_response": time.time()}
            with store.lock:
                store.calls.append(rec)
            self._send(200, result)
    return H


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8100)
    ap.add_argument("--db", default=os.path.join(HERE, "..", "data", "mock_db.json"))
    ap.add_argument("--tools", default=os.path.join(HERE, "..", "data", "brand", "tools.json"))
    a = ap.parse_args()
    store = Store(a.db, a.tools)
    print(f"mock tools on :{a.port}  functions={len(store.tools)}  records={sum(map(len, store.db.values()))}",
          file=sys.stderr)
    ThreadingHTTPServer(("0.0.0.0", a.port), make_handler(store)).serve_forever()


if __name__ == "__main__":
    main()
