# [실시간 대화 채점기] live_chat.py 로 한 통화(여러 말)를 마치면, 대회 채점기(score.py)와 같은 규칙으로 속도·응대·함수 점수를 냅니다.
# 왜 있나: 멀티턴 대화가 "대회 측정 기준으로 몇 점인지"를 바로 보려고요. 점수표·함수 판정·응대 판정(채점 AI)은 score.py 것을 그대로 가져다 씁니다.
# 말 한 번을 문항 하나로 보고, 앞선 대화는 history 로 넘겨서 채점합니다. (공식 점수가 아니라 연습용 점수예요)
"""사용:  python eval/live_score.py --run runs/live/20261121_131500            # 속도 + 응대(OPENAI_API_KEY 가 있을 때) + 함수(정답을 줬을 때)
        python eval/live_score.py --run runs/live/20261121_131500 --no-judge  # 속도·함수만 (무료)

점수 (SCORING.md 와 같음):
  속도  첫 소리(손님 말 끝 → 상담원 첫 소리) 점수표. 함수를 부른 말은 '조회 뒤 답'(마지막 함수 응답 → 그 뒤 첫 소리)과 평균.
        손님 말 도중에 소리를 냈거나(끼어듦) 소리가 없으면 0점. 손님 말 끝은 마이크 소리로 어림한 값이라 몇십 ms 오차가 있어요.
  응대  채점 AI(gpt-5.5)가 시나리오·브랜드 정보·앞선 대화를 보고 맞음 1.0 / 괜찮음 0.7 / 틀림 0 으로 판정. OPENAI_API_KEY 로 부릅니다.
  함수  '정답 호출'을 알려 준 말만 채점(--expect 파일). 정답을 모르면 채점하지 않고 부른 함수만 보여 줍니다.
총점 = 응대 × 0.5 + 함수 × 0.2 + 속도 × 0.3 은 세 점수가 모두 있을 때(모든 말에 정답을 줬을 때)만 나옵니다.
"""
import argparse, asyncio, json, statistics, sys
from pathlib import Path

import httpx
from dotenv import load_dotenv

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
for env in (ROOT / "agent_server" / ".env", ROOT / ".env"):          # 응대 채점용 OPENAI_API_KEY 를 .env 에서 읽는다
    load_dotenv(env, encoding="utf-8-sig")
sys.path.insert(0, str(HERE))
import score as official                                            # 대회 채점기: 점수표·함수 판정·응대 판정을 그대로 쓴다

INF = float("inf")


def speed_of(r):
    """(첫 소리, 조회 뒤 답, 속도 점수). score.py 와 같은 규칙: 끼어들었거나 소리가 없으면 0점."""
    first = r.get("first_sound")
    first = INF if first is None or r.get("spoke_before_end") else first
    pts = official.speed_points(first)
    if not r.get("tool_calls"):
        return first, None, pts
    after = r.get("after_tool_sound")
    after = INF if after is None else after
    return first, after, (pts + official.speed_points(after, official.AFTER_TOOL_TABLE)) / 2


async def score_run(run_dir, judge=True):
    run_dir = Path(run_dir)
    cases = [json.loads(l) for l in open(run_dir / "cases.jsonl", encoding="utf-8")]
    results = {r["case_id"]: r for r in map(json.loads, open(run_dir / "results.jsonl", encoding="utf-8"))}
    judge_on = judge and official.HAS_KEY()
    if judge_on:
        print("응대 채점 중... (말마다 채점 AI를 불러요)", flush=True)
    if judge and not judge_on:
        print("※ OPENAI_API_KEY 가 없어서 응대 채점은 건너뜁니다 (게이트웨이 키로는 채점기를 부를 수 없어요).")
    ctx = {"scenarios": open(Path(official.BRAND) / "scenarios.md", encoding="utf-8").read(),
           "info": "\n".join(f"{i['tag']}\n{i['content']}" for i in json.load(open(Path(official.BRAND) / "information.json", encoding="utf-8"))),
           "rubric": {}}

    async def one(c, case):
        r = results[case["case_id"]]
        out = {**r, "customer": case["customer_utterance"], "has_expectation": case.get("has_expectation", False)}
        out["first"], out["after"], out["speed_points"] = speed_of(r)
        out["tool_ok"], out["tool_why"] = (official.tool_match(case["reference"]["tool_calls"], r.get("tool_calls", []), case["history"])
                                           if out["has_expectation"] else (None, "정답 미지정"))
        if judge_on:
            said = await official.transcribe(c, str(run_dir / "agent_audio" / (case["case_id"] + ".wav")))
            verdict = await official.judge(c, case, said, r.get("tool_calls", []), ctx)
            out.update(said=said, **verdict)
        return out

    async with httpx.AsyncClient(timeout=180) as c:
        rows = await asyncio.gather(*[one(c, case) for case in cases])
    with open(run_dir / "scored.jsonl", "w", encoding="utf-8") as f:
        for s in rows:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")

    fmt = lambda v: "—" if v is None or v == INF else f"{v:.2f}초"
    print(f"\n{'=' * 60}\n대화 채점 결과 ({len(rows)}번 말, 대회 채점기와 같은 규칙)\n{'=' * 60}")
    for i, s in enumerate(rows, 1):
        cut = " [손님이 상담원 말을 끊음]" if s.get("interrupted") else ""
        print(f"\n[{i}] 손님: {s['customer'][:60]}\n    상담원: {(s.get('agent_text') or '(말 없음)')[:90]}{cut}")
        after = f" | 조회 뒤 답 {fmt(s['after'])}" if s["after"] is not None else ""
        early = " (손님 말 도중에 답함 → 0점)" if s.get("spoke_before_end") else ""
        print(f"    속도 {s['speed_points']:.0f}점: 첫 소리 {fmt(s['first'])}{after}{early}")
        if judge_on:
            print(f"    응대 {s['verdict']}: {s['reason'][:110]}")
        calls = [c["name"] for c in s.get("tool_calls", [])] or "없음"
        print("    함수 " + (f"{'일치' if s['tool_ok'] else '틀림 — ' + s['tool_why']} (부른 함수 {calls})" if s["has_expectation"] else f"정답 미지정 (부른 함수 {calls})"))
        if s.get("turn_end") is not None and s["turn_end"] > 25:
            print(f"    ! turn.end 가 말 끝 뒤 {s['turn_end']:.1f}초에 왔어요 (규정 25초)")

    speed = statistics.mean(s["speed_points"] for s in rows)
    resp = statistics.mean(official.PTS[s["verdict"]] for s in rows) * 100 if judge_on else None
    graded = [s for s in rows if s["has_expectation"]]
    tool = statistics.mean(1.0 if s["tool_ok"] else 0.0 for s in graded) * 100 if graded else None
    firsts = [s["first"] for s in rows]
    afters = [s["after"] for s in rows if s["after"] is not None]
    print(f"\n첫 소리 중앙값 {fmt(statistics.median(firsts))} | 조회 뒤 답 중앙값 {fmt(statistics.median(afters)) if afters else '—'}")
    total = None
    if judge_on and graded and len(graded) == len(rows):
        total = resp * official.W_RESP + tool * official.W_TOOL + speed * official.W_SPEED
        print(f"응대 {resp:.1f} × 0.5 + 함수 {tool:.1f} × 0.2 + 속도 {speed:.1f} × 0.3 = 총점 {total:.1f}")
    else:
        parts = [f"속도 {speed:.1f}"]
        if judge_on:
            counts = ", ".join("%s %d" % (k, sum(s["verdict"] == k for s in rows)) for k in official.PTS)
            parts.append(f"응대 {resp:.1f} ({counts})")
        parts.append(f"함수 {tool:.1f} (정답을 준 {len(graded)}개 말만)" if graded else "함수 점수 없음 (정답을 안 줬어요)")
        print("  /  ".join(parts))
        print("  → 총점은 응대·함수·속도가 모두 있어야 나와요 (모든 말에 --expect 로 정답 호출을 주세요).")
    json.dump({"turns": len(rows), "speed_score": round(speed, 1), "response_score": None if resp is None else round(resp, 1),
               "tool_score": None if tool is None else round(tool, 1), "graded_tool_turns": len(graded), "total": None if total is None else round(total, 1)},
              open(run_dir / "summary.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"\n저장: {run_dir}\\scored.jsonl, summary.json  (공식 점수가 아니라 연습용 점수예요)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", help="live_chat.py 가 만든 폴더 (runs/live/날짜_시각)")
    ap.add_argument("--latest", action="store_true", help="가장 최근 대화를 채점한다 (폴더 이름을 몰라도 됨)")
    ap.add_argument("--no-judge", action="store_true", help="응대 채점(OpenAI) 없이 속도·함수만")
    a = ap.parse_args()
    if a.latest:
        done = sorted(p for p in (ROOT / "runs" / "live").glob("*") if (p / "results.jsonl").exists())   # 말이 한 번도 없었던 빈 폴더는 건너뛴다
        if not done:
            raise SystemExit("채점할 대화가 없어요 (runs/live 에 말이 기록된 폴더가 없음)")
        a.run = done[-1]
        print(f"가장 최근 대화: {a.run}")
    elif not a.run:
        ap.error("--run 폴더 이름이나 --latest 중 하나를 주세요")
    asyncio.run(score_run(a.run, judge=not a.no_judge))
