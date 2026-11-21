# [크레딧 확인기] BAZE 게이트웨이에 남은 크레딧과, 오늘 어디에 얼마나 썼는지를 보여 줍니다 (읽기만 해서 크레딧이 들지 않아요).
# 왜 있나: 팀이 같은 키를 나눠 쓰고 최종 측정용 크레딧을 남겨 둬야 해서, 수시로 잔액을 보려고요.  실행: python eval/credits.py
# 키는 agent_server/.env (또는 맨 위 폴더의 .env) 의 BAZE_API_KEY 를 읽습니다.
import os
from pathlib import Path

import httpx
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
for env in (ROOT / "agent_server" / ".env", ROOT / ".env"):
    load_dotenv(env, encoding="utf-8-sig")
BASE = os.getenv("BAZE_BASE_URL", "https://api.baze.co/v1/gateway").rstrip("/")
if not os.getenv("BAZE_API_KEY"):
    raise SystemExit("BAZE_API_KEY 가 없습니다. agent_server/.env 에 적어 주세요.")
H = {"Authorization": "Bearer " + os.environ["BAZE_API_KEY"]}


def get(path, **params):
    r = httpx.get(f"{BASE}{path}", headers=H, params=params, timeout=30)
    r.raise_for_status()
    return r.json()


c = get("/credits/")
t = c["total"]
print(f"남은 크레딧 {t['remaining']:,.0f} / 전체 {t['quota']:,.0f}  (사용 {t['used']:,.0f}, {t['remaining'] / t['quota']:.0%} 남음)")
for key, label in (("monthly_allocated", "월 지급"), ("org_granted", "그룹 지급"), ("purchased", "구매")):
    v = c[key]
    print(f"  {label:6s} 한도 {v['quota']:,.0f} | 사용 {v['used']:,.0f} | 남음 {v['remaining']:,.0f}")

u = get("/usage/", by="model")
total = u["total"]["credits"] or 1
print(f"\n사용량 {u['start_date']} ~ {u['end_date']} (이 키): 호출 {u['total']['call_count']:,}회, 크레딧 {u['total']['credits']:,.0f}")
for r in u["rows"][:12]:
    print(f"  {r['key']:34s} {r['call_count']:>6,}회 {r['credits']:>9,.1f} ({r['credits'] / total:.0%})")
print("\n※ 'mindlogic-masking' 은 개인정보 가리기 수수료예요. 우리 서버가 쓰는 모델은 gpt-5.4-mini, gemini-3.8-flash-tts, stt-rt-v6 입니다.")
