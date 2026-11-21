# pace 연계: 말하는 사람의 속도에 맞춰 '말 끝' 대기 시간 조절

다른 에이전트가 이 기능을 그대로 구현하거나 다른 서버로 옮길 수 있도록 정리한 문서다.
기준 구현은 `voice_agent/agent/pace.py`와 `voice_agent/agent/session.py`이고, 브랜치 `feat/voice-agent-gateway-pipeline`의 커밋 `3bc6bfa`(도입)와 `5e2a6b2`(배율 1.0~1.2로 조정)에 있다.

## 1. 목적

음성 상담 서버는 고객이 조용해진 시간을 재서 "말이 끝났다"고 확정한 뒤 답한다.
느리게 말하는 사람은 문장 사이 쉼도 길어서, 같은 대기 시간을 쓰면 말 도중에 끼어들게 된다.
이 기능은 말하는 사람의 속도와 쉼 버릇을 재서 **느린 사람에게만 대기 시간을 조금 늘린다.**

## 2. 반드시 지킬 원칙

| 원칙 | 이유 |
|---|---|
| 늘리기만 하고 줄이지 않는다 (배율 1.0~1.2) | 줄이면 끼어든다. 끼어들면 그 문항의 속도 점수가 0점이다 |
| 빨리 확정하는 규칙에만 적용한다 | 이미 1.6~2.4초 기다리는 규칙(쉼표 끝, 번호 읽는 중, 연결어미, 인사)은 충분히 길다 |
| 근거가 부족하면 손대지 않는다 | 짧은 말과 번호 위주의 말로는 속도를 믿을 수 없다 |
| 시연 모드에서만 켠다 (기본값) | 측정용 합성 음성은 속도와 쉼이 무관했다 (아래 4절). 최종 채점은 측정 모드다 |
| 기능 오류가 대답을 막으면 안 된다 | 계산할 수 없으면 원래 대기 시간을 쓴다 |

## 3. 계산 방법

### 3.1 발음 속도 (글자/초)

```
발음 속도 = 받아쓴 글자 수 ÷ 실제 소리가 난 시간(초)
```

- **글자 수**: 한글 음절과 영문자만 센다. 숫자, 공백, 문장부호는 빼고 센다. 숫자는 사람마다 읽는 길이가 달라서다.
- **소리가 난 시간**: 음량(RMS)이 무음 기준 이상인 20ms 조각들의 합이다. 쉼은 넣지 않는다. 발화 전체 시간으로 나누면 쉼이 섞여 값이 흔들린다.
- **재지 않는 경우**(None을 돌려준다):
  - 글자 8자 미만
  - 소리 1.5초 미만
  - 숫자 개수가 글자 수의 절반을 넘음 (번호 위주의 말)

### 3.2 본인 쉼

- **이번 발화 중간의 쉼**: 말소리가 다시 시작될 때 직전 소리와의 간격이다. 250ms 미만의 끊김은 쉼으로 치지 않는다.
- **끼어들기에서 배운 쉼**: 서버가 말 끝으로 확정했는데 고객이 이어 말한 경우, 확정 시점의 말 끝부터 다시 말하기 시작한 시점까지의 간격이다. 통화 동안 최근 5개를 기억한다.
- 위 두 값 중 큰 쪽을 쓴다.

### 3.3 대기 시간

```
배율 = clamp(기준 속도 / 발음 속도, 1.0, 1.2)
대기 = 원래 대기 × 배율
대기 = max(대기, 본인 쉼 × 1.25)
대기 = clamp(대기, 원래 대기 × 1.0, min(원래 대기 × 1.2, 원래 대기 + 상한))
```

- 발음 속도는 이번 턴 값을 먼저 쓰고, 재지 못하면 통화에서 누적한 값을 쓴다.
- 통화 누적값은 지수 평균이다: `새 값 = 0.6 × 이전 값 + 0.4 × 이번 값`. 1~12자/초 밖의 값은 버린다.
- 원래 대기 0.6초인 말(시연 모드, 문장부호 끝)의 예:

| 말하는 사람 | 대기 |
|---|---|
| 5.3자/초 이상 (기준 이상) | 0.60초 (그대로) |
| 4.8자/초 | 약 0.66초 |
| 4.4자/초 이하 | 0.72초 (1.2배, 상한) |

## 4. 근거 데이터

연습 음성 78개(합성 음성)를 쉼 기준 RMS 300, 20ms 조각으로 분석한 결과다.

| 발음 속도 3등분 | 평균 속도 | 말 중간 최대 쉼 평균 | 상위 10% 쉼 |
|---|---|---|---|
| 느린 1/3 | 4.1자/초 | 418ms | 630ms |
| 중간 1/3 | 5.3자/초 | 368ms | 610ms |
| 빠른 1/3 | 6.1자/초 | 419ms | 660ms |

- 합성 음성은 속도와 쉼이 무관하다. 그래서 측정 모드는 끄고, 실제 사람이 말하는 시연 모드에만 켠다.
- 기준 속도 5.3자/초는 이 합성 음성의 중앙값이다. 사람 목소리로 다시 맞춰야 한다(7절).
- 말 중간 쉼이 1.0초를 넘은 문항은 78개 중 1개였다 (P0130-01, 1.8초).

## 5. 설정값

| 이름 | 기본값 | 뜻 |
|---|---|---|
| `pace_adapt` | `"live"` | `off` / `live`(시연만) / `all`(측정 포함) |
| `pace_ref_cps` | 5.3 | 기준 발음 속도. 이보다 느리면 늘린다 |
| `pace_min_factor` | 1.0 | 배율 하한. 1.0 = 줄이지 않음 |
| `pace_max_factor` | 1.2 | 배율 상한 |
| `pace_max_extra_live_ms` | 600 | 시연 모드에서 늘어나는 최대 폭 |
| `pace_max_extra_measure_ms` | 300 | 측정 모드에서 늘어나는 최대 폭 (`all`일 때만 의미) |
| `pace_min_chars` | 8 | 속도를 재는 최소 글자 수 |
| `pace_min_voiced_ms` | 1500 | 속도를 재는 최소 소리 시간 |
| `pace_pause_margin` | 1.25 | 본인 쉼의 이 배수만큼은 기다린다 |
| `pace_min_pause_ms` | 250 | 이보다 짧은 끊김은 쉼이 아니다 |

## 6. 서버에 연결하는 자리

통화 한 건을 맡는 세션 객체에 아래 상태와 연결점이 필요하다.

### 6.1 상태

| 상태 | 수명 | 뜻 |
|---|---|---|
| `pace` (`Profile`) | 통화 전체 | 누적 발음 속도, 끼어들기에서 배운 쉼 |
| `turn_voiced_ms` | 턴마다 초기화 | 이번 턴에 소리가 난 시간 |
| `turn_pauses` | 턴마다 초기화 | 이번 턴의 말 중간 쉼 목록 |
| `carry_voiced_ms` | 다음 턴 시작 때 한 번 | 끊긴 말을 이어 붙일 때 앞말의 소리 시간 |

### 6.2 연결점 다섯 곳

**① 음성 20ms 조각을 받을 때** (`session.py` `_on_audio`)

```python
if voiced:
    if speech_started and voiced_run_ms == 0 and last_voiced_audio_ms is not None:
        gap = audio_ms - dt - last_voiced_audio_ms          # 쉬었다가 다시 말하기 시작함
        if gap >= s.pace_min_pause_ms and phase == "listening":
            turn_pauses.append(gap)
    voiced_run_ms += dt
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
if pace.enabled(s, mode) and rule_name in ("punct", "silence", "complete"):
    rate = pace.turn_rate(text, turn_voiced_ms, s) or profile.rate
    own = max(max(turn_pauses, default=0), profile.pause_hint())
    thr, info = pace.adjust(thr, rule_name, rate=rate, own_pause_ms=own, s=s, mode=mode)
if silence_ms < thr:
    return None
```

- 판정 모델(Jev) 응답을 기다리는 추가 대기의 조건 비교에는 늘린 값이 아니라 `base_thr`를 쓴다.
- 확정할 때 `info`를 로그에 남긴다 (`cps`, `factor`, `own_pause_ms`, `base_ms`, `thr_ms`). 받아쓴 글은 남기지 않는다.

**④ 말 끝을 확정할 때** (`_commit`)

```python
profile.note_rate(pace.turn_rate(text, turn_voiced_ms, s))
```

**⑤ 확정 직후 고객이 이어 말해 앞말과 합칠 때** (`_finish_interrupted`, 시연 모드)

```python
carry_voiced_ms = max(0, turn_voiced_ms - voiced_run_ms)
gap = (audio_ms - voiced_run_ms) - speech_end_audio_ms   # 확정한 말 끝 → 다시 말하기 시작
profile.note_pause(gap)                                  # 200~4000ms만 기록
```

### 6.3 조절 대상 규칙

| 규칙 이름 | 원래 대기 (측정 / 시연) | 조절 |
|---|---|---|
| `punct` 문장부호로 끝남 | 1.0초 / 0.6초 | 함 |
| `complete` 판정 모델이 완결로 봄 | 1.0초 / 0.6~0.9초 | 함 |
| `silence` 판정 없음 | 1.1초 | 함 |
| `comma`, `partial_number`, `connective`, `preamble`, `noun`, `incomplete`, `very_incomplete` | 1.6~2.4초 | 안 함 |

## 7. 기준 속도 맞추는 법

1. 시연 화면(`python demo/serve.py` → http://localhost:8080)에서 평소 속도로 몇 문장, 일부러 천천히 몇 문장 말한다.
2. 각 턴의 `turn.commit` 이벤트에 붙은 `pace` 값을 본다. 대기를 늘리지 않은 턴에는 비어 있다.
3. 평소 속도 `cps`보다 약간 낮은 값을 `pace_ref_cps`로 정한다. 평소에도 자주 늘어나면 값을 낮추고, 천천히 말해도 안 늘어나면 높인다.

## 8. 시험 (`tests/test_units.py`의 `PaceTest`)

| 시험 | 확인 내용 |
|---|---|
| `test_turn_rate_needs_evidence` | 짧은 말, 소리 부족, 번호 위주의 말은 None. 16자 / 5초 = 3.2자/초 |
| `test_adjust_range` | 빠른 사람은 그대로, 느린 사람은 1.2배, 본인 쉼 반영, 상한, 긴 규칙은 그대로, 측정 모드는 꺼짐 |
| `test_profile` | 이상값은 버리고 지수 평균, 가장 긴 쉼 기억 |
| `test_commit_rule_waits_longer_for_slow_speaker` | 시연 모드 0.6초 → 느린 사람 0.72초, 배운 쉼 0.5초 → 0.625초, 측정 모드 1.0초 유지 |
| `test_audio_tracks_voiced_time_and_pauses` | 1.0초 말 + 0.6초 쉼 + 0.5초 말 + 0.1초 끊김 + 0.2초 말 → 소리 1.7초, 쉼 1개(약 600ms) |

```
cd voice_agent
python -m unittest tests.test_units -v
```

## 9. 주의할 점

- 받아쓰기 글자는 소리보다 0.3~0.6초 늦게 따라온다. 말하는 도중에는 속도가 실제보다 느리게 나와 대기가 늘어날 수 있다. 상한 1.2배가 이 오차를 막아 준다.
- 끊긴 말을 합친 턴은 앞말의 소리 시간을 더해야 속도가 맞다 (6.2절 ②의 `carry_voiced_ms`).
- 무음 기준 음량이 고정(RMS 300)이라 작은 목소리나 잡음 환경에서는 약한 음절이 쉼으로 잡힐 수 있다. 그러면 본인 쉼이 커져 대기가 상한까지 늘어난다. 이 경우 끼어들기는 줄어드는 방향이라 위험하지 않다.
- 측정 모드에서 켜려면(`pace_adapt="all"`) 연습 음성 재생 시험으로 1.5초를 넘는 첫 소리가 늘지 않는지 먼저 확인한다.
