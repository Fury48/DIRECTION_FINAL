# 인계: TTS 출력 경로와 pace (말 끝 대기·재생 속도)

다른 에이전트가 이 기능을 그대로 구현하거나 다른 서버로 옮길 수 있도록, **지금 `voice_agent` 구현** 그대로 정리한 문서다.

- 기준 브랜치: `feat/voice-agent-gateway-pipeline`, 커밋 `381d6a1` 이후
- 기준 파일
  - `voice_agent/agent/tts.py`: 합성·캐시·16k 변환
  - `voice_agent/agent/audio.py`: `StreamResampler`, `LeadTrimmer`, `time_stretch`
  - `voice_agent/agent/pace.py`: 속도 계산, 대기 조절, 재생 속도
  - `voice_agent/agent/session.py`: 연결점
  - `voice_agent/agent/config.py`: 설정값

| 커밋 | 내용 |
|---|---|
| `3bc6bfa` | 말 끝 대기 조절 도입 |
| `5e2a6b2` | 대기 배율 1.0~1.2배 (줄이지 않음) |
| `fdfec43` | 이전 턴 기록이 있을 때만 적용, 상한 분리, 측정 오차 수정, 재생 속도 조절 추가 |
| `381d6a1` | 변환 끝 잘림·홀수 바이트 수정, 녹음 문장 뒤 쉼 이중 확장 수정, 시연 화면 표시, 시연 기본 대기 0.8초·1.0초 |

> 같은 폴더의 `tts_speed_and_pace_handoff.md`는 다른 설계다(기본 1.2배속, 측정 모드에서도 배속, 본인 쉼 상한 버그 포함). 지금 구현과 맞지 않으므로 이 문서를 기준으로 삼는다.

---

## 1. 목적

1. **TTS 출력**: 게이트웨이 TTS(24kHz PCM)를 대회 규격 16kHz 16-bit mono로 바꾸고, 원래 속도(1.0배)로 캐시·전송한다.
2. **말 끝 대기 조절**: 느리게 말하거나 말 중간에 길게 쉬는 사람에게는 말 끝 확정을 조금 더 기다린다 (1.0~1.2배, 본인 쉼 반영).
3. **재생 속도 조절**: 느리게 말하는 사람에게는 상담원 음성을 음높이를 유지한 채 느리게(0.85~1.0배) 들려주고, 문장 사이 쉼도 그만큼 늘린다.

## 2. 반드시 지킬 원칙

| 원칙 | 이유 |
|---|---|
| TTS 음성은 **1.0배(원래 속도)로 만들고 캐시한다**. 배속은 보내기 직전에만 적용한다 | 캐시에 배속을 넣으면 다시 바꿀 때 두 번 변환돼 음질이 떨어진다 |
| 같은 통화의 **이전 턴 기록이 있을 때만** pace를 적용한다. 첫 턴은 기록만 한다 | 채점은 질문 하나뿐이고 이전 대화는 글로만 온다. 그래서 모드 설정과 상관없이 채점에서는 구조적으로 발동하지 않는다 |
| 대기는 늘리기만 하고 줄이지 않는다 (속도 배율 1.0~1.2) | 줄이면 끼어든다. 끼어들면 그 문항의 속도 점수가 0점이다 |
| 재생은 느리게만 한다 (0.85~1.0배). 빠르게 하지 않는다 | 빠른 사람에게 빨리 말해 줄 이득보다 알아듣기 어려워지는 손해가 크다. 0.85배 미만은 WSOLA 울림이 들릴 수 있다 |
| 재생 속도는 **시연 모드에서만** 바꾼다. 측정 모드는 늘 1.0배 | 채점기는 우리 음성을 받아써(gpt-4o-transcribe) 응대를 평가한다 |
| 빨리 확정하는 규칙에만 대기 조절을 적용한다 | 이미 1.6~2.4초 기다리는 규칙(쉼표 끝, 번호 읽는 중, 연결어미, 인사)은 충분히 길다 |
| 근거가 부족하면 손대지 않는다 | 짧은 말과 번호 위주의 말로는 속도를 믿을 수 없다 |
| 반응 속도를 잃지 않는다 | 변환은 10초 문장 약 20ms다. 게이트웨이 TTS는 문장 단위로 한 번에 오므로 문장을 다 받고 변환해도 첫 소리가 늦어지지 않는다 |
| 기능 오류가 대답을 막으면 안 된다 | 계산·변환에 실패하면 원래 대기 시간과 원래 음성을 쓴다 |

## 3. TTS 출력 경로 (`tts.py`)

```
게이트웨이 /audio/speech/ (gemini-3.8-flash-tts, 원시 PCM 24kHz, 스트리밍 없음: 약 3초 뒤 한 번에 옴)
  → StreamResampler(24000 → 16000)        # audio.py, 조각 단위로 이어지는 리샘플러
  → LeadTrimmer(thr=silence_rms)          # 앞쪽 무음 제거 (첫 소리를 앞당김)
  → apply_gain(tts_gain=1.0)
  → 16kHz PCM 조각 (1.0배)
  → 디스크 캐시 runtime/tts_cache/gemini_<모델>_<목소리>/<sha1>.pcm
```

- 캐시 키: `sha1(gemini_tts_api | tts_style | tts_style_prefix | 글)` 앞 20자. 글은 `for_tts()`로 읽는 형태로 바꾼 뒤의 글이다(번호는 한글 숫자 등).
- 미리 합성하는 문장(첫마디, 고정 답, 조회 뒤 첫 문장, 제품별 첫 문장, 직접 답 후보 등 123개)은 항상 캐시한다. 실시간 답 음성은 `persist_dynamic_tts=False`라 저장하지 않는다.
- 제출 ZIP에 캐시 폴더를 함께 넣어 운영진 서버에서 다시 합성하지 않는다 (`ops/make_submission.py`).
- 실패 처리
  - 첫 음성 6초 제한(`tts_first_audio_timeout_s`), 재시도 1회(`tts_retries`)
  - 이미 소리를 보낸 뒤 끊기면 재시도하지 않는다 (같은 말 반복 방지)
  - 문장 합성이 끝내 실패하면 그 문장만 건너뛴다. 대체 음성으로 뜻을 바꾸지 않는다
- 답 문장은 문장 단위로 나눠 동시에 합성한다(`TTSJob`). 첫 단위는 첫 문장 하나로 두고, 길면 쉼표에서 앞부분만 먼저 합성한다(`facts.short_head`).
- 재생 속도는 이 경로에 넣지 않는다. 6.2절 ⑦에서 보내기 직전에 적용한다.

## 4. 계산 방법

### 4.1 발음 속도 (글자/초, `pace.turn_rate`)

```
발음 속도 = 받아쓴 글자 수 ÷ 실제 소리가 난 시간(초)
```

- **글자 수**: 한글 음절과 영문자만 센다. 숫자, 공백, 문장부호는 빼고 센다.
- **소리가 난 시간**: 고객 차례(`phase == "listening"`)에서 음량(RMS)이 무음 기준 이상인 20ms 조각들의 합이다.
  - 쉼은 넣지 않는다.
  - 상담원이 말하는 동안의 소리(에코·잡음)는 세지 않는다.
- **언제 재나**: 확정 받아쓰기가 나온 뒤(`_finalize_transcript`). 받아쓰기는 소리보다 0.3~0.6초 늦어, 확정 직전 글자로 재면 끝 글자가 빠져 속도가 실제보다 느리게 나온다.
- **재지 않는 경우** (None): 글자 8자 미만 / 소리 1.5초 미만 / 숫자 개수가 글자 수의 절반 초과

### 4.2 통화 기록 (`pace.Profile`, 통화 내내 유지)

- **누적 속도**: 지수 평균 `새 값 = 0.6 × 이전 값 + 0.4 × 이번 값`. 1~12자/초 밖의 값은 버린다.
- **끼어들기에서 배운 쉼**: 서버가 확정했는데 고객이 이어 말한 경우의 간격(확정한 말 끝 → 다시 말하기 시작). 200~4000ms만, 최근 5개.
- **`ready()`**: 누적 속도가 있거나 배운 쉼이 하나라도 있으면 참. 첫 턴에는 늘 거짓.

### 4.3 말 끝 대기 (`pace.adjust`)

```
조건: pace_adapt가 켜짐, 규칙이 punct / silence / complete, Profile.ready()

속도 = 이번 턴 발음 속도 (못 재면 누적 속도)
본인 쉼 = max(이번 턴 말 중간 쉼, 배운 쉼)

대기 = 원래 대기 × clamp(기준 속도 / 속도, 1.0, 1.2)                   # 속도 배율
대기 = max(대기, min(본인 쉼 × 1.25, 원래 대기 + 모드별 상한))          # 본인 쉼 (상한을 따로 둔다)
대기 = max(대기, 원래 대기 × 1.0)                                       # 줄이지 않음
```

- 두 상한을 나누는 이유: 본인 쉼을 속도 상한(1.2배)에 묶으면 학습이 막힌다. 시연 기본 대기가 0.6초였을 때는 상한이 0.72초라, 0.75초 쉬다 끊긴 사람이 다음 턴에도 또 끊겼다. 기본 0.8초에서도 상한 0.96초를 넘게 쉬는 사람은 같은 문제가 생기므로, 본인 쉼은 "원래 대기 + 0.6초"까지 따로 허용한다.
- 모드별 상한: 시연 +0.6초, 측정 +0.3초. 측정 쪽은 채점에서 `ready()`가 거짓이라 쓰이지 않는다.
- 시연 모드 기본 대기 0.8초(`live_commit_fast_ms`, 문장부호 끝)의 예:

| 상황 | 대기 |
|---|---|
| 첫 턴 | 0.80초 (기록만) |
| 5.3자/초 이상 | 0.80초 |
| 4.8자/초 | 약 0.88초 |
| 4.4자/초 이하 | 0.96초 (속도 상한 1.2배) |
| 지난번 0.78초 쉬고 이어 말함 | 0.975초 (본인 쉼 × 1.25) |
| 배운 쉼이 아주 김 | 1.4초 (본인 쉼 상한 +0.6초) |

### 4.4 재생 속도 (`pace.play_speed`)

```
조건: 시연 모드, pace_play 켜짐, 누적 속도가 있음 (= 이전 턴에서 잰 속도)
재생 속도 = clamp(누적 속도 / 기준 속도, 0.85, 1.0) 를 0.05 단위로 반올림
문장 사이 쉼 = 원래 쉼 ÷ 재생 속도   (문장 사이 150ms, 쉼표 조각 사이 80ms)
```

| 누적 속도 | 재생 속도 | 문장 사이 쉼 |
|---|---|---|
| 5.3자/초 이상 | 1.0배 | 150ms |
| 4.8자/초 | 0.9배 | 167ms |
| 4.4자/초 이하 | 0.85배 (하한) | 176ms |

- 답을 시작할 때(`_respond`) 턴마다 한 번 정한다.
- 0.05 단위로 맞추는 이유: 같은 녹음 문장의 변환 결과를 다시 쓰려고(세션별 캐시, 최대 200개).

### 4.5 음높이 유지 속도 변환 (`audio.time_stretch`, WSOLA)

- 16kHz PCM, 30ms 창(Hanning)을 15ms 간격으로 겹쳐 이어 붙인다.
- 각 창은 앞 창과 파형이 가장 잘 이어지는 위치를 ±10ms 안에서 상호상관으로 찾는다. 겹친 부분은 창 합으로 나눠 정규화한다.
- 끝에 0을 (창 + 탐색 폭 + 간격)만큼 덧대 마지막 조각까지 쓰고, 결과를 `round(입력 길이 / 속도)`로 자른다. 덧대지 않으면 끝 약 40ms(어미 끝)가 잘린다.
- 홀수 바이트는 마지막 1바이트를 버린다. 약 0.1초 미만 입력과 1.0배는 그대로 돌려준다.
- 속도: 10초 문장 약 20ms, 1.5초 첫마디 수 ms.

## 5. 근거 데이터

**연습 음성 78개** (합성 음성, 쉼 기준 RMS 300, 20ms 조각)

| 발음 속도 3등분 | 평균 속도 | 말 중간 최대 쉼 평균 | 상위 10% 쉼 |
|---|---|---|---|
| 느린 1/3 | 4.1자/초 | 418ms | 630ms |
| 중간 1/3 | 5.3자/초 | 368ms | 610ms |
| 빠른 1/3 | 6.1자/초 | 419ms | 660ms |

- 합성 음성은 속도와 쉼이 무관하다. 그래서 채점에는 쓸모가 없고, 사람과 대화하는 시연에서만 의미가 있다.
- 기준 속도 5.3자/초는 합성 음성의 중앙값이다. 사람 목소리로 다시 맞춰야 한다(8절).

**느린 고객 3턴 실측** (연습 음성을 0.75배속으로 만들어 실제 모델·시연 모드로 대화, `fdfec43`, 당시 시연 기본 대기 0.6초)

| 턴 | 잰 속도 | 말 끝 대기 | 재생 속도 | 결과 |
|---|---|---|---|---|
| 1 | - | 0.60초 (기록만) | 1.0배 | 정상 |
| 2 | 3.7자/초 | 0.72초 | 0.85배 | 고객이 0.78초 쉬어 끼어듦 → 이어 말한 것으로 합쳐 처리, 쉼 학습 |
| 2 (이어 말함) | 4.4자/초 | 0.975초 | 0.85배 | 정상 |
| 3 | 4.0자/초 | 0.975초 | 0.85배 | 끼어들기 없음 |

- 말 끝 확정 → 첫 소리: 7~16ms. 느리게 재생해도 반응 속도 손해가 없었다.
- 같은 서버로 측정 모드 2문항: 대기 조절·재생 속도 모두 발동하지 않음, 첫 소리 0.98~0.99초로 변화 없음.

## 6. 서버에 연결하는 자리

### 6.1 세션 상태

| 상태 | 수명 | 뜻 |
|---|---|---|
| `pace` (`Profile`) | 통화 전체 | 누적 발음 속도, 끼어들기에서 배운 쉼 |
| `turn_voiced_ms` | 턴마다 초기화 | 이번 턴(고객 차례)에 소리가 난 시간 |
| `turn_pauses` | 턴마다 초기화 | 이번 턴의 말 중간 쉼 목록 |
| `carry_voiced_ms` | 다음 턴 시작 때 한 번 | 끊긴 말을 이어 붙일 때 앞말의 소리 시간 |
| `play_speed` | 답마다 | 이번 답의 재생 속도 |
| `paced_cache` | 통화 전체 | (음성 길이, 해시, 속도) → 변환한 음성 |

### 6.2 연결점 일곱 곳

**① 음성 20ms 조각을 받을 때** (`_on_audio`)

```python
if voiced:
    if speech_started and voiced_run_ms == 0 and last_voiced_audio_ms is not None:
        gap = audio_ms - dt - last_voiced_audio_ms          # 쉬었다가 다시 말하기 시작함
        if gap >= s.pace_min_pause_ms and phase == "listening":
            turn_pauses.append(gap)
    voiced_run_ms += dt
    if phase == "listening":                               # 상담원이 말하는 동안의 소리는 세지 않는다
        turn_voiced_ms += dt
```

**② 턴을 새로 시작할 때** (`_reset_turn`)

```python
turn_voiced_ms = (voiced_run_ms if keep_speech else 0) + carry_voiced_ms
carry_voiced_ms = 0
turn_pauses = []
```

**③ 말 끝 규칙에서 대기 시간을 정한 직후** (`_commit_rule`)

```python
base_thr = thr
pace_info = {}
if pace.enabled(s, mode) and rule_name in ("punct", "silence", "complete") and profile.ready():
    rate = pace.turn_rate(text, turn_voiced_ms, s) or profile.rate
    own = max(max(turn_pauses, default=0), profile.pause_hint())
    thr, pace_info = pace.adjust(thr, rule_name, rate=rate, own_pause_ms=own, s=s, mode=mode)
if silence_ms < thr:
    return None
# Jev 응답을 기다리는 추가 대기의 조건 비교에는 늘린 값이 아니라 base_thr를 쓴다
...
if pace_info:
    tl.info["pace"] = pace_info          # 실제로 확정하는 순간에만 기록 (return 직전)
return rule_name
```

- 기록 항목: `cps`, `factor`, `own_pause_ms`, `base_ms`, `thr_ms`. 받아쓴 글은 남기지 않는다.

**④ 확정 받아쓰기가 나왔을 때** (`_finalize_transcript`)

```python
final = (carry_text + " " + final) if carry_text else final     # 끊긴 말 이어 붙이기 (시연 모드)
profile.note_rate(pace.turn_rate(final, turn_voiced_ms, s))
```

**⑤ 확정 직후 고객이 이어 말해 앞말과 합칠 때** (`_finish_interrupted`, 시연 모드)

```python
carry_voiced_ms = max(0, turn_voiced_ms - voiced_run_ms)
gap = (audio_ms - voiced_run_ms) - speech_end_audio_ms   # 확정한 말 끝 → 다시 말하기 시작
profile.note_pause(gap)
```

- 합치는 조건: 상담원이 첫마디만 했고, 고정 답·실제 답·함수 실행이 아직 없을 때.

**⑥ 답을 시작할 때** (`_respond`)

```python
play_speed = pace.play_speed(profile, s, mode)    # 시연 모드 + 이전 턴 속도가 있을 때만 1.0 미만
```

**⑦ 음성을 보낼 때**

```python
# 녹음 문장(첫마디, 고정 답, 미리 합성 문장): _speak
await send_pcm(paced(pcm + raw_gap))     # 원래 길이의 쉼을 붙여 함께 변환 (미리 늘려 두면 이중으로 늘어난다)

# 실시간 합성 문장: _answer
if play_speed < 1.0:
    whole = b"".join([c async for c in job.iterate()])     # 문장 음성을 다 받은 뒤 (게이트웨이는 어차피 한 번에 온다)
    await send_pcm(paced(whole))
    await send_pcm(silence(sentence_gap_ms / play_speed))  # 늘린 쉼은 따로 보낸다
else:
    async for c in job.iterate():
        await send_pcm(c)
```

- `paced(pcm)`: `play_speed`가 1.0이면 그대로. 아니면 캐시를 보고 없으면 `time_stretch`. 실패하면 원래 음성.
- 보낸 길이(끼어들 때 어디까지 들려줬는지 추정)는 변환 뒤 길이로 계산되므로 따로 맞출 것이 없다.
- 시연 화면: `turn.commit`의 `pace`로 "대기 800→960ms (3.7자/초)", `turn.metrics`의 `play_speed`로 "느린 말투 → 상담원 재생 0.85배, 문장 사이 쉼 176ms"를 보여 준다.

### 6.3 대기 조절 대상 규칙

| 규칙 이름 | 원래 대기 (측정 / 시연) | 조절 |
|---|---|---|
| `punct` 문장부호로 끝남 | 1.0초 / 0.8초 | 함 |
| `complete` 판정 모델이 완결로 봄 | 1.0초 / 0.8~1.0초 | 함 |
| `silence` 판정 없음 | 1.1초 | 함 |
| `comma`, `partial_number`, `connective`, `preamble`, `noun`, `incomplete`, `very_incomplete` | 1.6~2.4초 (시연 `noun`은 1.0초) | 안 함 |

## 7. 설정값 (`agent/config.py`)

| 이름 | 기본값 | 뜻 |
|---|---|---|
| `gemini_tts_model` | `gemini-3.8-flash-tts` | TTS 모델 (게이트웨이 `/audio/speech/`) |
| `gemini_voice` | `Kore` | 목소리 |
| `tts_first_audio_timeout_s` | 6.0 | 첫 음성 제한 |
| `tts_retries` | 1 | 재시도 횟수 |
| `tts_gain` | 1.0 | 음량 |
| `persist_dynamic_tts` | False | 실시간 답 음성을 디스크에 저장할지 |
| `sentence_gap_ms` / `comma_gap_ms` | 150 / 80 | 문장 사이 / 쉼표 조각 사이 쉼 |
| `live_commit_fast_ms` / `live_commit_mid_ms` | 800 / 1000 | 시연 모드 기본 대기 |
| `pace_adapt` | `"all"` | 대기 조절: `off` / `live` / `all`. 이전 턴 기록이 있어야 발동해 채점에서는 발동 안 함 |
| `pace_ref_cps` | 5.3 | 기준 발음 속도 (글자/초) |
| `pace_min_factor` / `pace_max_factor` | 1.0 / 1.2 | 대기 속도 배율 범위 |
| `pace_max_extra_live_ms` / `pace_max_extra_measure_ms` | 600 / 300 | 본인 쉼으로 늘어나는 최대 폭 |
| `pace_min_chars` / `pace_min_voiced_ms` | 8 / 1500 | 속도를 재는 최소 근거 |
| `pace_pause_margin` | 1.25 | 본인 쉼의 이 배수만큼은 기다린다 |
| `pace_min_pause_ms` | 250 | 이보다 짧은 끊김은 쉼이 아니다 |
| `pace_play` | True | 느린 사람에게 상담원 음성도 느리게 (시연 모드만) |
| `pace_play_min_speed` | 0.85 | 가장 느린 재생 속도 |

## 8. 기준 속도 맞추는 법

1. 서버 세 개를 띄운다: `python -m agent.server`(9000), `python mock_tools/server.py --db data/mock_db.json`(8100), `python demo/serve.py`(8080).
2. 시연 화면 http://localhost:8080 에서 평소 속도로 몇 문장, 일부러 천천히 몇 문장 말한다.
3. 둘째 턴부터 커밋 표시의 "대기 800→960ms (3.7자/초)"와 로그의 재생 속도를 본다.
4. 평소 속도 `cps`보다 약간 낮은 값을 `pace_ref_cps`로 정한다. 평소에도 자주 늘어나거나 음성이 느려지면 낮추고, 천천히 말해도 반응이 없으면 높인다.
5. 사람 목소리 없이 재현: `python bench/live_check.py --turns P0137-01,P0121-01,P0077-01 --barge 0 --slow 0.75`

## 9. 시험 (`tests/test_units.py`)

| 시험 | 확인 내용 |
|---|---|
| `PaceTest.test_turn_rate_needs_evidence` | 짧은 말, 소리 부족, 번호 위주의 말은 None. 16자 / 5초 = 3.2자/초 |
| `PaceTest.test_adjust_range` | 빠른 사람 그대로, 느린 사람 1.2배, 0.75초 쉬던 사람 0.94초(상한에 안 묶임), 본인 쉼 상한 +0.6초, 긴 규칙 그대로, 측정 상한 +0.3초 |
| `PaceTest.test_profile` | 이상값은 버리고 지수 평균, 가장 긴 쉼 기억 |
| `PaceTest.test_commit_rule_waits_longer_for_slow_speaker` | 첫 턴은 느려도 그대로, 이전 턴이 느렸으면 기본 × 1.2, 배운 쉼 0.75초면 0.95초, 채점(질문 하나)은 1.0초 유지 |
| `PaceTest.test_audio_tracks_voiced_time_and_pauses` | 소리 1.7초, 쉼 1개(약 600ms) 기록 |
| `SlowPlaybackTest.test_stretch_keeps_pitch` | 0.85·0.9배에서 길이 1/속도, 음높이 유지 |
| `SlowPlaybackTest.test_stretch_is_fast` | 10초 문장 변환 0.15초 미만 |
| `SlowPlaybackTest.test_play_speed_rules` | 첫 턴 1.0, 빠른 사람 1.0, 4.8자/초 → 0.9, 아주 느리면 0.85, 측정 모드 늘 1.0 |
| `SlowPlaybackTest.test_paced_cache_and_fallback` | 같은 문장은 한 번만 변환, 1.0이면 원래 음성, 실패 시 원래 음성 |
| `Feedback2Test.test_stretch_length_and_odd_bytes` | 길이가 정확히 `round(입력 / 속도)`, 홀수 바이트도 오류 없음 |
| `LiveContinuationTest.*` | 끊긴 말 합치기 조건, 합친 뒤 판단 |

```
cd voice_agent
python -m unittest tests.test_units -v
```

## 10. 주의할 점

- **다른 서버로 옮길 때**: 캐시를 배속된 음성으로 저장하지 않는다. 1.0배로 저장하고 보낼 때 한 번만 바꾼다.
- **무음 기준 고정(RMS 300)**: 작은 목소리나 잡음 환경에서는 약한 음절이 쉼으로 잡혀 대기가 늘 수 있다. 끼어들기가 줄어드는 방향이라 위험하지는 않지만 답이 늦어질 수 있다.
- **첫 턴에는 적용되지 않는다**: 느린 사람의 첫 질문에서는 끼어들 수 있다. 그때 끊긴 말 이어 붙이기(시연 모드)가 대화를 이어 주며 그 쉼을 배운다.
- **0.85배보다 낮추지 않는다**: WSOLA 특유의 울림이 들릴 수 있다. 낮추려면 먼저 직접 들어 본다.
- **측정 모드에서 재생 속도를 바꾸지 않는다**: 채점기가 우리 음성을 받아써 평가한다.
