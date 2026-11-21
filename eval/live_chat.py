# [실시간 대화 시험기] 마이크로 말하면 상담원 서버와 여러 번 주고받으며(멀티턴) 상담원 목소리를 스피커로 바로 들려 주고, 끝나면 대회 기준으로 채점합니다.
# 왜 있나: 측정기는 "말 한 번"만 재서, 티키타카 대화(끼어들기 포함)의 자연스러움과 점수를 직접 확인할 방법이 필요했어요.
# 규격 그대로 말 한 번마다 연결 하나를 열고 session.start 에 지금까지의 대화(history)를 글로 담아 보냅니다. 말마다 결과를 저장해 live_score.py 가 채점해요.
"""사용:  python eval/live_chat.py                          # Enter 를 한 번만 누르면 그 뒤로는 그냥 말하면 이어짐. 끝내려면 Enter
        python eval/live_chat.py --caller 01058003641 --now "2026-10-09 (금) 08:29"   # 걸려 온 번호·대화 당시 시각을 정해서 시작
        python eval/live_chat.py --expect 정답.json        # 말마다 정답 함수 호출을 알려 주면 함수 점수까지 낸다 (아래 형식)
        python eval/live_chat.py --manual                 # 말할 때마다 Enter 를 누르는 방식 (r=새 전화, q=끝)
        python eval/live_chat.py --no-barge-in            # 끼어들기(상담원 말 끊기)를 끈다 (스피커로 들을 때 등)
        python eval/live_chat.py --list-samples           # 전산실에 기록이 있는 번호 예시
        python eval/live_chat.py --wav a.wav b.wav --no-play   # 마이크 대신 wav 를 차례로 말함 (소리·마이크 없이도 시험 가능)
준비:  python -m pip install sounddevice   (마이크·스피커를 쓸 때만. **이어폰·헤드셋 필수**: 스피커 소리가 마이크로 들어가면 상담원 말을 손님 말로 착각해요)
       서버 2개를 먼저 켜 둔다 (전산실 8100, 상담원 9000)
이어지는 대화:  Enter 를 누르면 마이크가 계속 켜져 있어요. 말을 시작하면 자동으로 한 번의 말로 보내고, 서버가 말 끝을 판단해서 답합니다.
              말이 안 걸리거나 잡음에 걸리면 --vad-rms 를 조절하세요.
끼어들기:      상담원이 말하는 중에(재생 중에) 손님이 말을 시작하면 상담원 말을 그 자리에서 멈추고, 그 말을 새 말로 보내 이어 가요.
              끊긴 말은 '실제로 재생된 만큼'만 상담원이 한 말로 기록하고 끝에 "…(손님이 말을 끊음)"을 붙여서 다음 응대·채점에 반영합니다.
              끼어들기로 보는 볼륨·길이는 --barge-rms (기본 1000)로 조절하세요. 헤드셋 말소리는 보통 2000 이상, 잡음은 수백이에요.
채점:  대화가 끝나면(Enter 또는 Ctrl+C) 자동으로 live_score.py 가 대회 채점기와 같은 규칙으로 속도·응대·함수 점수를 냅니다. --no-judge(응대 채점 생략) --no-score(채점 안 함)
기록:  runs/live/<날짜_시각>/ 에 cases.jsonl, results.jsonl, agent_audio/*.wav 가 남아요 (나중에 python eval/live_score.py --latest 로 다시 채점 가능)
정답 형식(--expect, 말 번호별. 정답을 준 말만 함수 점수를 매겨요):
  {"1": {"tool_calls": []}, "2": {"tool_calls": [{"name": "get_order_info", "arguments": {"buyer_cellphone": "01012345678"}}], "answer": "(선택) 참고 답"}}
"""
import argparse, asyncio, collections, json, math, sys, threading, time, uuid, wave
from datetime import datetime
from pathlib import Path

import numpy as np
import websockets
from scipy.signal import resample_poly

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_eval import Relay                                    # 측정기와 같은 함수 중계 (함수별 응답 지연도 같다)
import live_score                                             # 대회 채점기와 같은 규칙으로 채점

ROOT = Path(__file__).resolve().parents[1]
RATE, FRAME = 16000, 640                                      # 16kHz 16-bit mono, 20ms = 640바이트
VOICE_RMS, SOUND_RMS = 500, 300                               # 서버가 말소리로 보는 볼륨 / 측정기가 첫 소리로 세는 볼륨 (VOICE_RMS 는 --vad-rms 로 바꿈)
BARGE_RMS, BARGE_FRAMES = 1000, 12                            # 끼어들기: 이 볼륨 이상이 연속 12조각(240ms)이면 손님이 말을 시작한 것 (--barge-rms)
GAP_S = 0.3                                                   # 상담원 말 사이 무음이 이보다 길면 '중간 대기'로 센다
PRE_FRAMES, START_FRAMES = 25, 3                              # 말이 시작되면 그 앞 0.5초도 함께 보낸다 / 연속 3조각(60ms)이 크면 말이 시작된 것
EARLY_S = 0.3                                                 # 상담원 첫 소리가 손님 마지막 소리 뒤 이 시간 안에 오면 '손님 말 도중에 답함'으로 본다 (서버는 0.8초 조용해야 답함)
CUT_MARK = "…(손님이 말을 끊음)"                                # 끊긴 상담원 말 뒤에 붙이는 표시. prompt.md 의 "손님이 말을 끊거나…" 규칙이 이 표시를 본다


def rms(pcm):
    a = np.frombuffer(pcm, dtype=np.int16).astype(np.float32)
    return float(np.sqrt((a * a).mean())) if len(a) else 0.0


def load_wav(path):
    with wave.open(str(path)) as w:
        assert (w.getframerate(), w.getnchannels(), w.getsampwidth()) == (RATE, 1, 2), f"{path}: 16kHz 모노 16-bit wav 가 아닙니다"
        return w.readframes(w.getnframes())


class Player:
    """받은 상담원 음성을 스피커로 바로 재생한다. 소리가 모자라면 무음이 나가서, 실제 통화처럼 끊김이 그대로 들린다.
    played = 지금까지 실제로 재생된 바이트 수 (끼어들기로 끊길 때 '어디까지 말했는지' 알려고 센다).
    virtual=True 이면 스피커 없이 '실시간 속도로 재생되는 척' 세어 준다 (끼어들기 시험용)."""

    def __init__(self, enabled, device, virtual=False):
        self.buf, self.lock, self.stream = bytearray(), threading.Lock(), None
        self.virtual, self._played, self._t = bool(virtual and not enabled), 0, time.perf_counter()
        if enabled:
            import sounddevice as sd
            self.stream = sd.RawOutputStream(samplerate=RATE, channels=1, dtype="int16", blocksize=320, device=device, callback=self._fill)
            self.stream.start()

    def _fill(self, outdata, frames, t, status):
        need = frames * 2
        with self.lock:
            chunk, self.buf = bytes(self.buf[:need]), self.buf[need:]
            self._played += len(chunk)
        outdata[:] = chunk.ljust(need, b"\0")

    @property
    def played(self):
        self._tick()
        return self._played

    def _tick(self):
        """가상 재생: 시간이 흐른 만큼 버퍼의 소리를 재생한 것으로 친다."""
        now = time.perf_counter()
        if self.virtual:
            with self.lock:
                take = min(len(self.buf), int((now - self._t) * RATE * 2) // 2 * 2)
                self._played += take
                self.buf = self.buf[take:]
        self._t = now

    def put(self, pcm):
        if self.stream or self.virtual:
            self._tick()
            with self.lock:
                self.buf += pcm

    def stop(self):
        """재생을 그 자리에서 멈추고 남은 소리를 버린다 (손님이 말을 끊었을 때)."""
        self._tick()
        with self.lock:
            self.buf.clear()

    async def drain(self):
        """재생이 다 끝날 때까지 기다린다 (다음 말과 겹치지 않게)."""
        while (self.stream or self.virtual) and self.buf:
            await asyncio.sleep(0.05)
            self._tick()
        if self.stream or self.virtual:
            await asyncio.sleep(0.3)


class Mic:
    """마이크 소리를 20ms(640바이트) 조각으로 큐에 넣는다. 마이크가 16kHz 를 못 받으면 기본 속도로 받아서 바꾼다."""

    def __init__(self, loop, device):
        import sounddevice as sd
        self.q, self.rate = asyncio.Queue(), RATE
        try:
            sd.check_input_settings(device=device, samplerate=RATE, channels=1, dtype="int16")
        except Exception:
            self.rate = int(sd.query_devices(device, "input")["default_samplerate"])

        def callback(indata, frames, t, status):
            pcm = bytes(indata)
            if self.rate != RATE:
                g = math.gcd(RATE, self.rate)
                a = resample_poly(np.frombuffer(pcm, dtype=np.int16).astype(np.float32), RATE // g, self.rate // g)[:320]
                pcm = np.clip(a, -32768, 32767).astype(np.int16).tobytes().ljust(FRAME, b"\0")
            loop.call_soon_threadsafe(self.q.put_nowait, pcm)

        self.stream = sd.RawInputStream(samplerate=self.rate, channels=1, dtype="int16", blocksize=self.rate // 50,
                                        device=device, callback=callback)
        self.stream.start()

    def close(self):
        self.stream.stop()
        self.stream.close()


async def wait_for_speech(mic, stop):
    """마이크를 계속 들으며 손님이 말을 시작할 때까지 기다린다. 시작하면 그 직전 0.5초까지 포함한 조각들을 돌려준다 (말 앞부분이 잘리지 않게)."""
    pre, loud = collections.deque(maxlen=PRE_FRAMES), 0
    while not stop.is_set():
        try:
            frame = await asyncio.wait_for(mic.q.get(), 0.2)
        except asyncio.TimeoutError:
            continue
        pre.append(frame)
        loud = loud + 1 if rms(frame) >= VOICE_RMS else 0
        if loud >= START_FRAMES:
            return list(pre)
    return None


async def watch_barge(mic):
    """상담원 말이 재생되는 동안 마이크를 듣다가 손님이 말을 시작하면 (그 앞 0.5초 포함) 조각들을 돌려준다."""
    recent, loud = collections.deque(maxlen=PRE_FRAMES), 0
    while True:
        frame = await mic.q.get()
        recent.append(frame)
        loud = loud + 1 if rms(frame) >= BARGE_RMS else 0
        if loud >= BARGE_FRAMES:
            return list(recent)


async def drain_or_barge(player, mic):
    """상담원 말이 다 재생될 때까지 기다린다. 그 사이에 손님이 말을 시작하면 재생을 멈추고 그 말의 앞부분을 돌려준다 (아니면 None)."""
    watcher, drainer = asyncio.create_task(watch_barge(mic)), asyncio.create_task(player.drain())
    done, pending = await asyncio.wait({watcher, drainer}, return_when=asyncio.FIRST_COMPLETED)
    for task in pending:
        task.cancel()
    if watcher in done:
        player.stop()
        return watcher.result()
    return None


async def live_frames(pre, mic):
    """미리 모아 둔 앞부분을 먼저 보내고, 그다음부터는 마이크 소리를 실시간으로 계속 보낸다."""
    for frame in pre:
        yield frame
    while True:
        yield await mic.q.get()


async def wav_frames(pcm):
    """wav 를 실제 속도로 보내고, 끝나면 측정기처럼 무음을 계속 보낸다."""
    t0, k = time.perf_counter(), 0
    while True:
        i = k * FRAME
        yield pcm[i:i + FRAME].ljust(FRAME, b"\0") if i < len(pcm) else b"\0" * FRAME
        k += 1
        await asyncio.sleep(max(0.0, t0 + k * 0.02 - time.perf_counter()))


def playback_gaps(chunks):
    """도착한 음성 조각을 '도착하는 즉시, 순서대로' 재생한다고 가정하고 (상담원 말 사이 무음 구간들, 재생이 끝나는 시각)을 계산한다."""
    clock, gaps = None, []
    for arrival, seconds in chunks:
        if clock is None:
            clock = arrival
        elif arrival > clock + GAP_S:
            gaps.append(arrival - clock)
            clock = arrival
        clock += seconds
    return gaps, clock


def now_text():
    d = datetime.now()
    return d.strftime(f"%Y-%m-%d ({'월화수목금토일'[d.weekday()]}) %H:%M")


def spoken_text(segments, total, played):
    """실제로 재생된 만큼만 상담원이 한 말로 본다. segments = [[글, 그 글의 음성이 시작되는 바이트], ...] (서버가 글을 먼저 보내고 그 음성을 보냄).
    다 재생된 글은 그대로, 중간에 끊긴 글은 재생된 비율만큼 잘라(단어 중간은 피함) 끝에 끊김 표시를 붙인다. 아무것도 못 말했으면 빈 글."""
    ends = [s[1] for s in segments[1:]] + [total]
    out = []
    for (text, start), end in zip(segments, ends):
        if played >= end:
            out.append(text)
            continue
        if played > start and end > start:
            cut = int(len(text) * (played - start) / (end - start))
            part = text[:cut].rstrip()
            if " " in part and text[cut:cut + 1].strip():          # 단어 중간에서 끊기면 마지막 공백까지만 남긴다
                part = part.rsplit(" ", 1)[0]
            if part:
                out.append(part)
        break
    return " ".join(out) + CUT_MARK if out else ""


def mark_interrupted(turn, pcm, played):
    """손님이 상담원 말을 끊었다: 상담원이 한 말을 '실제로 재생된 만큼'으로 고치고, 채점기가 듣는 음성도 그만큼만 남긴다."""
    played -= played % 2
    turn["agent"] = spoken_text(turn["segments"], len(pcm), played) or "(말하기 전에 손님이 말을 끊음)"
    turn["interrupted"] = True
    turn["result"].update(agent_text=turn["agent"], interrupted=True)
    return pcm[:played]


class Session:
    """한 통화: 지금까지의 대화(history)와, 말마다 채점기가 읽는 파일(cases.jsonl, results.jsonl, agent_audio/)을 관리한다."""

    def __init__(self, args, relay):
        self.args, self.relay, self.history, self.n = args, relay, [], 0
        self.caller = {"name": None, "phone": args.caller, "email": None} if args.caller else None
        self.expect = json.load(open(args.expect, encoding="utf-8-sig")) if args.expect else {}     # utf-8-sig: 메모장·파워셸이 붙이는 BOM 이 있어도 읽는다
        self.run_dir = ROOT / "runs" / "live" / datetime.now().strftime("%Y%m%d_%H%M%S")
        (self.run_dir / "agent_audio").mkdir(parents=True)

    def record(self, turn, pcm, history_before):
        """말 한 번을 '문항 하나'로 저장한다 (공식 측정기·채점기와 같은 모양). 저장하자마자 history 에 이어 붙인다."""
        cid, exp = f"LIVE{turn['turn']}", self.expect.get(str(turn["turn"]))
        case = {"case_id": cid, "room_id": "LIVE", "now": turn["now"], "history": history_before, "customer_utterance": turn["customer"],
                "customer_profile": self.caller, "has_expectation": exp is not None,
                "reference": {"tool_calls": (exp or {}).get("tool_calls", []), "tool_results": [], "answer": (exp or {}).get("answer", "")}}
        with open(self.run_dir / "cases.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps(case, ensure_ascii=False) + "\n")
        with open(self.run_dir / "results.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps({"case_id": cid, **turn["result"]}, ensure_ascii=False) + "\n")
        with wave.open(str(self.run_dir / "agent_audio" / f"{cid}.wav"), "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(RATE)
            w.writeframes(pcm)
        self.history.append({"role": "customer", "text": turn["customer"]})
        self.history.extend(turn["history_add"])
        self.history.append({"role": "agent", "text": turn["agent"]})


async def one_turn(sess, frames, player, barge=False):
    """말 한 번 = 연결 한 번. 서버가 turn.end 를 보낼 때까지 마이크(또는 wav) 소리를 흘려 보내고, 받은 답 음성을 바로 재생한다.
    barge=True 이면 상담원이 말하는 중에 손님이 말을 시작하는지 지켜보다가, 시작하면 재생을 멈추고 연결을 닫는다."""
    args, n, now = sess.args, sess.n + 1, sess.args.now or now_text()
    sid, done = f"LIVE{n}-{uuid.uuid4().hex[:6]}", asyncio.Event()
    voiced, chunks, sounds, texts, pcm, segments = [], [], [], [], bytearray(), []   # 손님 소리가 컸던 시각들 / (도착 시각, 길이) / 소리 있는 조각의 도착 시각들
    st = {"first_audio": None, "turn_end": None, "customer": "", "trace": [], "barge_pre": None, "played": None}
    baseline, history_before = player.played, list(sess.history)
    async with websockets.connect(args.server, max_size=None, open_timeout=10) as ws:
        await ws.send(json.dumps({"type": "session.start", "session_id": sid, "case_id": f"LIVE{n}", "now": now, "history": history_before,
                                  "caller": sess.caller, "tools_url": f"http://localhost:{args.relay_port}",
                                  "audio": {"encoding": "pcm_s16le", "sample_rate": RATE, "channels": 1}, "debug": True}, ensure_ascii=False))

        async def sender():
            recent, loud = collections.deque(maxlen=PRE_FRAMES), 0
            async for frame in frames:
                if done.is_set():
                    return
                await ws.send(frame)
                recent.append(frame)
                if rms(frame) >= VOICE_RMS:
                    voiced.append(time.perf_counter())
                if barge and st["first_audio"] is not None:
                    loud = loud + 1 if rms(frame) >= BARGE_RMS else 0
                    if loud >= BARGE_FRAMES:                       # 상담원이 말을 시작한 뒤에 손님이 말을 시작했다 → 끼어들기
                        st["barge_pre"], st["played"] = list(recent), player.played - baseline
                        player.stop()
                        done.set()
                        await ws.close()
                        return

        async def receiver():
            async for m in ws:
                t = time.perf_counter()
                if isinstance(m, bytes):
                    if rms(m) >= SOUND_RMS:
                        sounds.append(t)
                        st["first_audio"] = st["first_audio"] or t
                    chunks.append((t, len(m) / (RATE * 2)))
                    pcm.extend(m)
                    player.put(m)
                    continue
                e = json.loads(m)
                if e["type"] == "agent.transcript":
                    texts.append(e["text"])
                    segments.append([e["text"], len(pcm)])
                elif e["type"] == "debug.heard":
                    st["customer"] = e["customer"]
                elif e["type"] == "debug.turn":
                    st["customer"], st["trace"] = e["customer"], e["history_add"]
                elif e["type"] == "turn.end":
                    st["turn_end"] = t
                    done.set()
                    return

        send_task = asyncio.create_task(sender())
        try:
            await asyncio.wait_for(receiver(), 30)
        except asyncio.TimeoutError:
            print("  ! 30초 안에 turn.end 가 오지 않았어요 (규정은 25초)")
        finally:
            done.set()
            send_task.cancel()

    first_audio = st["first_audio"]
    before = [t for t in voiced if first_audio is None or t <= first_audio]
    speech_end = before[-1] + 0.02 if before else None                         # 손님 말 끝: 상담원 첫 소리가 오기 전 마지막으로 손님 소리가 컸던 시각
    first_sound = None if first_audio is None or speech_end is None else first_audio - speech_end
    early = first_sound is not None and first_sound < EARLY_S
    calls, offset = sess.relay.take(sid), time.time() - time.perf_counter()    # 함수 중계가 남긴 호출 기록 (벽시계 → perf 시각으로 환산)
    after = None
    if calls:
        last = max(x["t_response"] for x in calls) - offset
        later = [t for t in sounds if t > last]
        after = later[0] - last if later else None
    gaps, end = playback_gaps(chunks)
    rel = (lambda w: None if speech_end is None else w - offset - speech_end)
    result = {"session_id": sid, "first_sound": first_sound, "spoke_before_end": early,
              "turn_end": None if st["turn_end"] is None or speech_end is None else st["turn_end"] - speech_end,
              "agent_text": " ".join(texts), "after_tool_sound": after,
              "tool_calls": [{"name": x["name"], "arguments": x["arguments"], "matched": x["matched"], "t_response": rel(x["t_response"])} for x in calls]}
    turn = {"turn": n, "now": now, "customer": st["customer"], "agent": " ".join(texts), "history_add": st["trace"], "result": result,
            "tools": [t["name"] for t in st["trace"] if t["role"] == "tool_call"], "first_wait_s": first_sound, "gaps_s": gaps,
            "total_s": None if end is None or speech_end is None else end - speech_end,
            "segments": segments, "baseline": baseline, "barge_pre": st["barge_pre"], "played_at_barge": st["played"], "interrupted": False}
    return turn, bytes(pcm), history_before


async def play_turn(sess, frames, player, mic=None):
    """말 한 번을 처리하고 기록한다. (성공 여부, 상담원 말을 끊고 들어온 손님 말의 앞부분 또는 None)을 돌려준다.
    mic 가 있으면(이어지는 대화) 상담원이 말하는 동안 끼어들기를 지켜본다."""
    barge = mic is not None and sess.args.barge_in
    try:
        turn, pcm, history_before = await one_turn(sess, frames, player, barge)
    except (OSError, websockets.WebSocketException) as e:
        print(f"  ! 서버에 연결하지 못했어요: {e!r}\n    (서버가 켜져 있는지, --server 주소가 맞는지 확인하세요)")
        return False, None
    barge_pre = turn["barge_pre"]
    if barge_pre is not None:                                                   # 답하는 도중(서버가 아직 말을 보내는 중)에 끊김
        pcm = mark_interrupted(turn, pcm, turn["played_at_barge"])
    elif barge:                                                                 # 서버는 다 보냈지만 스피커에서 아직 재생 중인 말을 끊을 수도 있다
        barge_pre = await drain_or_barge(player, mic)
        if barge_pre is not None:
            played = player.played - turn["baseline"]
            if played < len(pcm) - 3200:                                        # 거의 다 말한 뒤(0.1초 남음)에 들어온 말은 끊은 것으로 치지 않는다
                pcm = mark_interrupted(turn, pcm, played)
    else:
        await player.drain()
    if not turn["customer"]:
        print("  ! 서버가 받아쓴 글을 알려 주지 않았어요. 상담원 서버가 옛 코드라면 껐다가 다시 켜 주세요 (이 말은 기록하지 않아요)")
        return True, barge_pre
    if "잠시 문제가 생겨" in turn["agent"]:
        print("  ! 서버가 오류 대응 문구(사과)를 말했어요. 정상 답변의 음성을 못 만든 것이니 상담원 서버 터미널의 '오류:' 줄을 확인하세요 "
              "(Gemini 한도 초과(429)면 '하루 한도'인지 '분당 한도'인지 함께 나와요).")
    sess.n += 1
    sess.record(turn, pcm, history_before)
    fmt = lambda v: "측정 못 함" if v is None else f"{v:.2f}초"
    print(f"\n[{turn['turn']}] 나: {turn['customer']}\n    상담원: {turn['agent']}")
    print(f"    첫 대기 {fmt(turn['first_wait_s'])} | 중간 무음 {len(turn['gaps_s'])}번 합계 {sum(turn['gaps_s']):.2f}초 | "
          f"말 끝부터 상담원 말 끝까지 {fmt(turn['total_s'])} | 함수 {turn['tools'] or '없음'}")
    return True, barge_pre


async def hands_free(sess, mic, player, stop):
    """Enter 를 한 번 누른 뒤로는 말이 시작될 때마다 자동으로 한 번의 말로 보내 이어 가는 대화.
    상담원이 말하는 중에 손님이 말을 시작하면(끼어들기) 그 자리에서 상담원 말을 멈추고, 그 말을 곧바로 새 말로 이어 간다."""
    print("\n● 듣는 중입니다. 말씀하세요. 상담원 말을 끊고 말해도 돼요. (끝내려면 Enter)")
    pre = None
    while not stop.is_set():
        if pre is None:
            pre = await wait_for_speech(mic, stop)
            if pre is None:
                break
        ok, pre = await play_turn(sess, live_frames(pre, mic), player, mic)
        if not ok:
            break
        if pre is not None:
            print("\n● 말을 끊고 들어오셨어요. 이어서 듣습니다...")
            continue
        while not mic.q.empty():
            mic.q.get_nowait()                                                 # 상담원이 말하는 동안 마이크에 들어온 소리는 버린다
        await asyncio.sleep(0.3)
        if not stop.is_set():
            print("\n● 듣는 중입니다...")


def watch_enter(loop, stop):
    """다시 Enter 를 누르면 대화를 끝낸다 (하던 말은 끝까지 처리)."""
    def run():
        try:
            input()
        except EOFError:
            return
        loop.call_soon_threadsafe(stop.set)
    threading.Thread(target=run, daemon=True).start()


def phone_text(d):
    d = "".join(c for c in str(d or "") if c.isdigit())
    return f"{d[:3]}-{d[3:-4]}-{d[-4:]}" if len(d) >= 10 else d


def print_samples():
    db = json.load(open(ROOT / "data" / "mock_db.json", encoding="utf-8"))
    print("전산실에 기록이 있는 번호 예시 (연습 데이터. 이 번호로 말하면 조회가 돼요)\n[주문 조회]")
    for r in [r for r in db.get("get_order_info", []) if r["result"].get("status") == "ok"][:6]:
        a, orders = r["arguments"], (r["result"].get("data") or {}).get("orders", [{}])
        print(f"  전화 {phone_text(a.get('buyer_cellphone')) or '-':14s} 주문번호 {a.get('order_id') or '-':18s} {orders[0].get('제품명', '')} / {orders[0].get('배송상태', '')}")
    print("[반품 현황 조회]")
    for r in [r for r in db.get("get_return_info", []) if r["result"].get("status") == "ok"][:3]:
        print(f"  전화 {phone_text(r['arguments'].get('customer_phone_number'))}")


async def converse(args, sess, player, loop, stop):
    """대화 방식(wav 순서대로 / 말할 때마다 Enter / Enter 한 번 뒤 계속 이어짐)에 따라 대화를 진행한다."""
    if args.wav:
        for wav in args.wav:
            print(f"\n[{sess.n + 1}번째 말] {Path(wav).name} 을(를) 말합니다")
            if not (await play_turn(sess, wav_frames(load_wav(wav)), player))[0]:
                break
    elif args.manual:
        while True:
            try:
                cmd = (await loop.run_in_executor(None, input, "\nEnter=말하기 시작   r=새 전화(대화 초기화)   q=끝내기 > ")).strip().lower()
            except EOFError:
                break
            if cmd == "q":
                break
            if cmd == "r":
                sess.history.clear()
                print("대화를 초기화했어요 (새 전화). 채점은 지금까지의 말을 한 통화로 모아서 해요")
                continue
            mic = Mic(loop, args.device_in)
            print("● 말씀하세요. 말하고 가만히 계시면 상담원이 답해요...")
            try:
                ok, _ = await play_turn(sess, live_frames(await wait_for_speech(mic, stop) or [], mic), player)
            finally:
                mic.close()
            if not ok:
                break
    else:
        try:
            await loop.run_in_executor(None, input, "\nEnter 를 누르면 대화가 시작돼요. 시작한 뒤에는 Enter 없이 그냥 말하면 되고, 끝낼 때 Enter 를 다시 누르세요 > ")
        except EOFError:
            return
        mic = Mic(loop, args.device_in)
        watch_enter(loop, stop)
        try:
            await hands_free(sess, mic, player, stop)
        finally:
            mic.close()


async def main():
    global VOICE_RMS, BARGE_RMS
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--server", default="ws://127.0.0.1:9000", help="상담원 서버 주소")
    ap.add_argument("--mock", default="http://127.0.0.1:8100", help="전산실(모의 함수 서버) 주소")
    ap.add_argument("--relay-port", type=int, default=8200, help="함수 중계가 받을 포트")
    ap.add_argument("--caller", help="발신자 번호 (예: 01058003641). 없으면 번호 모르는 손님")
    ap.add_argument("--now", help='대화 당시 날짜·시각 (예: "2026-10-09 (금) 08:29"). 없으면 지금. 연습 데이터는 9~10월이라 대본마다 맞춰야 반품 기간 판단이 맞아요')
    ap.add_argument("--expect", help="말 번호별 정답 함수 호출 JSON (함수 점수용). 형식은 맨 위 설명 참고")
    ap.add_argument("--manual", action="store_true", help="말할 때마다 Enter 를 누르는 방식 (기본은 Enter 한 번 뒤 계속 이어지는 대화)")
    ap.add_argument("--vad-rms", type=int, default=VOICE_RMS, help="이 볼륨 이상이면 말소리로 본다 (잡음에 걸리면 올리고, 말이 안 걸리면 낮추세요)")
    ap.add_argument("--barge-rms", type=int, default=BARGE_RMS, help="상담원이 말하는 중에 이 볼륨 이상이 0.24초 이어지면 끼어들기로 본다")
    ap.add_argument("--no-barge-in", dest="barge_in", action="store_false", help="끼어들기(상담원 말 끊기)를 끈다. 스피커로 들으면 상담원 소리에 반응할 수 있어 끄는 게 좋아요")
    ap.add_argument("--wav", nargs="+", help="마이크 대신 이 wav 들을 차례로 말한다 (16kHz 모노 16-bit)")
    ap.add_argument("--no-play", action="store_true", help="스피커로 재생하지 않는다")
    ap.add_argument("--no-judge", action="store_true", help="채점할 때 응대 채점(OpenAI)을 건너뛴다")
    ap.add_argument("--no-score", action="store_true", help="끝나고 채점하지 않는다")
    ap.add_argument("--device-in", type=int, help="마이크 번호")
    ap.add_argument("--device-out", type=int, help="스피커 번호")
    ap.add_argument("--list-devices", action="store_true", help="마이크·스피커 목록을 보여 주고 끝낸다")
    ap.add_argument("--list-samples", action="store_true", help="전산실에 기록이 있는 번호 예시를 보여 주고 끝낸다")
    args = ap.parse_args()
    VOICE_RMS, BARGE_RMS = args.vad_rms, args.barge_rms
    if args.list_samples:
        return print_samples()
    if args.list_devices:
        import sounddevice as sd
        return print(sd.query_devices())

    relay, player = Relay(args.mock, args.relay_port), Player(not args.no_play, args.device_out)
    relay.server.handle_error = lambda request, client_address: None    # 손님이 말을 끊어 서버가 함수 호출을 중단하면 중계가 쓰는 연결 오류 로그가 나는데, 정상이라 숨긴다
    sess, loop, stop = Session(args, relay), asyncio.get_running_loop(), asyncio.Event()
    print(f"상담원 서버 {args.server} 와 대화합니다. 기록 폴더: {sess.run_dir}")
    try:
        await converse(args, sess, player, loop, stop)
    except (KeyboardInterrupt, asyncio.CancelledError):            # Ctrl+C 나 창 닫기 직전에 끝내도, 지금까지의 말은 채점한다
        print("\nCtrl+C 로 끝냈어요.")
        task = asyncio.current_task()
        if task is not None and hasattr(task, "uncancel"):
            task.uncancel()
    if sess.n and not args.no_score:
        print("\n대화가 끝났어요. 대회 채점기와 같은 규칙으로 채점합니다 (응대 채점은 말마다 채점 AI를 불러서 몇 초~수십 초 걸려요)...", flush=True)
        try:
            await live_score.score_run(sess.run_dir, judge=not args.no_judge)
        except Exception as e:                                     # 채점이 실패해도 기록은 남아 있다 (나중에 live_score.py 로 다시 채점)
            print(f"  ! 채점하지 못했어요: {e!r}\n    기록은 남아 있어요. 나중에 다시 채점: python eval/live_score.py --latest")
    elif sess.n:
        print("\n채점은 나중에: python eval/live_score.py --latest")


if __name__ == "__main__":
    asyncio.run(main())
