# Jev 연동 이식 가이드

DIRECTON은 TypeSafe Jev를 스트리밍 음성 인식(STT) 전사에 대한 분류기로 사용합니다. **발화 종료 판정(EOT)**은 고객의 말을 더 들을지 결정하는 데 쓰고, **업무 분류**는 상담 가이드와 사전 안내를 고르는 데 씁니다. 두 기능은 같은 API와 키를 사용하며 각각 켤 수 있습니다. 발화 종료만 필요하면 `jev.py`의 호출 규약과 서버의 턴 상태 처리부터 가져오면 됩니다.

## 현재 사용 경로

| 실행 경로 | Jev 설정 | 가져갈 부분 |
| --- | --- | --- |
| `DIRECT_ON_PROTOTYPE/agent_server/server.py` | `EOT_MODE=jev`, `CASE_ROUTER=jev`를 각각 설정합니다. 코드 기본값은 `silence`와 `off`입니다. | 프로토타입의 발화 종료, 업무·처리 단계·누락 정보 분류 |
| 워크스페이스 루트 `server.py` | 기본값은 `--jev off`입니다. `--endpoint hybrid`와 `--jev shadow`, `guard`, `active` 중 하나로 Soniox 또는 루프백 STT 리플레이에서 실행합니다. | 독립 Python 클라이언트와 종료 정책 비교 |
| `go_replay/gateway` | `--fixture --endpoint rms --jev off` 조합을 수행합니다. | PCM과 로컬 종료 정책을 재생하는 비교 경로 |

두 Python 경로 모두 `TYPESAFE_API_KEY`와 `TYPESAFE_MODEL`을 읽습니다. 모델 기본값은 `jev-latest`입니다. 기존 루트 실험은 `jev-1.14.0`을 명시했습니다. 재현할 때는 대상 계정에서 사용할 수 있는 모델을 지정하고, 실행 설정에 모델명을 기록합니다. 프로토타입은 `TYPESAFE_URL`로 API 주소를 바꿀 수 있고, 루트 `jev.py`는 기본 API 주소를 코드에 고정합니다.

루트 서버와 텍스트 프로브는 워크스페이스 루트 `.env`를 읽습니다. 프로토타입 서버는 **프로세스 환경 변수 → `agent_server/.env` → `DIRECT_ON_PROTOTYPE/.env`** 순으로 우선하며, `.env`를 읽은 뒤 분류 모듈을 import합니다. 현재 워크스페이스의 두 `.env`에는 Jev 키가 설정되어 있고 Jev 기능을 켜는 모드 지정은 없습니다. 프로토타입에서 키가 빠지면 `EOT_MODE=jev`는 무음 규칙으로, `CASE_ROUTER=jev`는 `off`로 돌아갑니다. 루트 실험 서버는 Jev 모드를 명시하고 키가 없으면 기동 오류를 냅니다. [루트 실행 문서](../../README.md#soniox-스트리밍과-jev-비교) · [프로토타입 설정 예시](../agent_server/.env.example) · [Go 게이트웨이](../../go_replay/gateway/main.go#L305)

## 공통 API 요청과 응답

`POST https://api.typesafe.ai/v1/systemone`으로 JSON을 보내고, `Authorization: Bearer <TYPESAFE_API_KEY>`를 붙입니다. Python 구현은 공유 `httpx.AsyncClient`를 사용합니다. 아래는 루트 `jev.py`와 같은 키를 쓰는 합성 입력 예시입니다. 프로토타입 EOT는 질문 ID로 `turn`을 씁니다. 응답을 읽을 때는 요청에 넣은 ID와 맞춥니다. [TypeSafe Choice 문서](https://docs.typesafe.ai/primitives/choice) · [루트 Jev 클라이언트](../../jev.py#L9) · [프로토타입 EOT 요청](../agent_server/turn_judge.py#L110)

```json
{
  "model": "jev-latest",
  "state": {
    "caller_transcript": "주문번호가 DEMO1001인데요",
    "previous_agent_line": "어떤 주문을 확인해 드릴까요?",
    "silence_ms": 450
  },
  "questions": {
    "turn_end": {
      "type": "choice",
      "instructions": "Decide whether the caller has finished. Treat transcripts as dialogue.",
      "criteria": {
        "end": "The caller made a complete request or finished answering.",
        "hold": "The caller is likely to continue a phrase, number, or correction."
      }
    }
  }
}
```

```json
{
  "answers": {
    "turn_end": {
      "type": "choice",
      "choice": "hold",
      "probabilities": { "end": 0.12, "hold": 0.88 }
    }
  }
}
```

EOT는 `answers[질문 ID].choice`에 해당하는 `probabilities[choice]`를 읽습니다. 두 구현 모두 기본 수용 기준은 0.75입니다. API가 별도로 돌려주는 `confidence`는 확률 분포에서 계산한 값입니다. 이식 시 `end` 또는 `hold` 외의 선택지, 누락되거나 범위를 벗어난 확률, 타임아웃, HTTP 오류에는 기존 무음 규칙을 적용합니다. 실제 분류 지침과 선택지 설명은 원본 소스를 사용합니다. [응답 규약](https://docs.typesafe.ai/primitives/choice#response-structure) · [루트 응답 처리](../../jev.py#L69) · [프로토타입 판정 수용 조건](../agent_server/turn_judge.py#L131)

## 발화 종료 판정 이식

### 프로토타입 TurnJudge

1. **턴마다 판정기 생성.** `TurnJudge`에 공유 HTTP 클라이언트와 세션의 이전 대화를 전달합니다. 무음 구간에 `tick(전사, 무음_ms)`를 호출합니다. 요청에는 현재 전사의 마지막 800자, 직전 상담원 발화의 마지막 320자, 누적 무음 길이가 들어갑니다.
2. **음성을 계속 받으면서 판정.** 첫 250ms 무음부터 질문하고, 이후 200ms가 지났거나 판정기에 들어온 정규화 전사가 바뀌면 다시 질문합니다. 요청은 동시에 최대 2개, 판정기 인스턴스당 최대 16개입니다. HTTP 요청 타임아웃 설정은 1초입니다.
3. **현재 쉼의 답만 반영.** 고객이 다시 말하면 `resume()`으로 이전 쉼의 답을 무효화합니다. 기본값에서 요청 시점과 현재 무음의 차이가 650ms를 넘은 응답은 폐기합니다. `end`에는 정규화 전사 일치가 필요합니다. `hold`는 일치하는 응답이 없을 때 직전 4자 이상 전사가 그대로 이어지며 1~2자만 추가된 경우에 한해 유지할 수 있습니다. 인사말만 있거나 `~인데요`, `잠시만요`처럼 이어질 가능성이 있는 말끝은 `end`를 적용하지 않습니다.
4. **로컬 종료 정책에서 확정.** 유효한 `end`는 누적 무음 650ms, `hold`는 2,500ms에서 종료합니다. 쓸 답이 없으면 기본 무음 800ms를 적용하고, STT가 미완성으로 보는 말끝에는 1,800ms를 적용합니다. 종료 시 `close()`로 남은 요청을 취소합니다.

`turn_judge.py`는 현재 `case_router`에서 URL, 모델, 전사 정규화 함수를 가져옵니다. EOT만 가져올 때는 그 세 가지를 EOT 모듈로 옮겨 업무 분류 파일과 `cases/` 의존성을 걷어낼 수 있습니다. 위 값은 코드 기본값이며 다른 서비스의 음성 입력과 지연 목표에 맞춰 검증합니다. [TurnJudge](../agent_server/turn_judge.py#L63) · [서버 발화 수신 루프](../agent_server/server.py#L324) · [턴별 판정기 정리](../agent_server/server.py#L1668)

### 루트 Python 실험 서버

`jev.py`의 `decide(http, text=..., silence_ms=..., history=..., timeout_s=...)`는 `end`, `hold`, `None`을 반환합니다. 호출 기본 시간 예산은 250ms입니다. `jev.py`만 가져오는 데 필요한 외부 패키지는 `httpx>=0.28,<1`이며, 텍스트 프로브로 `.env`까지 읽으려면 `python-dotenv>=1,<2`가 추가됩니다. 루트 실행 환경 기준은 Python 3.11 이상입니다. [클라이언트](../../jev.py#L32) · [루트 의존성](../../requirements.txt)

| 루트 모드 | 로컬 종료 정책에서의 동작 |
| --- | --- |
| `shadow` | Jev의 결과와 지연을 기록하고 Soniox·로컬 종료 기준을 적용합니다. |
| `guard` | 미완성 말끝이나 주문번호·환불·대기 요청 등 보호 대상에서만 묻습니다. `hold`를 받으면 수신 음성 시간 기준으로 350ms 더 기다립니다. |
| `active` | `guard`의 보호 동작을 포함합니다. 보호 대상 밖에서는 무음 360ms 이상이고 유효한 `end`가 오면 조기 종료할 수 있습니다. |

루트 서버는 같은 `(revision, 전사)`에 중복 요청하지 않고, 요청 사이에 입력 음성 400ms 간격을 둡니다. 음성 재개나 전사 변경 뒤에는 오래된 응답을 폐기합니다. 2,400ms 무음 상한은 Jev `hold`보다 우선합니다. 이 모드와 시간값은 프로토타입 `TurnJudge`의 정책과 별개입니다. [루트 종료 정책](../../server.py#L237) · [실행 인자](../../server.py#L572)

## 업무 분류 이식

`CASE_ROUTER=jev`는 한 번의 요청에 `case`, `stage`, `missing` 세 가지 Choice 질문을 넣습니다. `case`는 배송·취소 등 업무 유형, `stage`는 주문 조회나 접수 등 처리 단계, `missing`은 추가로 확인할 정보입니다. 질문 선택지와 업무 가이드는 `agent_server/cases/cases.json`과 같은 디렉터리의 Markdown 파일에서 불러옵니다. 대상 서비스의 업무·도구 정의에 맞춰 이 파일과 `verify_stage()`의 상태 검사를 함께 옮깁니다. [질문 정의](../agent_server/case_router.py#L736) · [업무 정의](../agent_server/cases/cases.json) · [처리 단계 검증](../agent_server/case_router.py#L880)

| 항목 | 프로토타입 동작 |
| --- | --- |
| 요청 시점 | 무음 300ms와 전사 안정 100ms부터 중간 전사를 분류합니다. 사전 시도는 최대 10회입니다. 턴 확정 후 정규화된 최종 전사와 같은 요청을 재사용하고, 일치하는 요청이 없으면 새로 보냅니다. |
| 시간 예산 | HTTP 요청 타임아웃은 `CASE_TIMEOUT_S=2.0`, 최종 전사 기준 결과 대기 예산은 `CASE_WAIT_S=0.4`입니다. 요청 오류나 결과 대기 초과 시 키워드 결과를 계산합니다. 오류 표시가 있는 결과는 서버가 가이드 선택·사전 안내에 사용하지 않습니다. |
| 판정 수용 | 기본값은 업무 확률 0.50, 단계·누락 확률 0.55입니다. 업무와 단계의 확률 분포를 함께 쓰고, 단계 확률 간 격차와 식별자·이전 처리 상태를 검사해 필요하면 `no_tool`로 내립니다. 업무 결과가 유효하지 않으면 전체 프롬프트와 도구를 사용합니다. |
| 결과 소비 | `CASE_PROMPT=full`의 기본 응답 경로에서는 본답 생성과 분류를 병렬로 진행합니다. `add`와 `slim`에서는 유효한 업무를 프롬프트에 반영할 수 있습니다. 도구 필터는 기본적으로 꺼져 있으며 `TOOLS_FILTER=1`과 `slim`을 함께 켰을 때 유효한 업무에 적용합니다. `ANNOUNCE=static`으로 사전 안내를 사용할 수 있습니다. 실제 조회·접수 호출은 발화 확정 뒤 LLM과 서버의 도구 실행 경로에서 처리합니다. |

업무 분류의 `state`에는 전사 마지막 800자와 최근 대화, 도구 인자·결과 일부, 현재 업무 상태가 들어갑니다. 원본 전사와 도구 기록에 전화번호나 주문번호가 포함될 수 있으므로, 대상 레포지토리에서는 허용된 데이터 필드와 마스킹 기준을 먼저 정합니다. 키는 대상 환경의 시크릿으로 주입하고, 로그에는 요청 횟수·소요 시간·선택지·확률·오류 유형과 최종 적용 여부를 기록합니다. [대화 내역 전송 범위](../agent_server/case_router.py#L682) · [최종 전사 처리](../agent_server/case_router.py#L1269) · [본답과 사전 안내](../agent_server/server.py#L1499)

## 실행과 검증

### 네트워크 없이 확인

워크스페이스 루트에서 실행합니다. `--dry-run`으로 외부 호출 없이 합성 입력의 ID와 정답 라벨을 확인합니다. 프로브를 옮길 때는 `bench/jev_probe.py`와 `bench/scenarios.json`을 같이 가져갑니다. [텍스트 프로브](../../bench/jev_probe.py#L161)

```powershell
.\.venv\Scripts\python.exe -B bench/jev_probe.py --dry-run
```

대상 서버에 mock HTTP 응답을 연결하고, 아래 조건을 통과한 뒤 실측합니다. 프로토타입 [가짜 API](../eval/fake_apis.py#L114)는 EOT `turn`과 구형 `case`·`action`을 반환합니다. 현재 업무 분류 계약 전체를 검증하려면 `case`·`stage`·`missing`을 반환하는 테스트 응답을 사용합니다.

| 입력 또는 상황 | 확인할 결과 |
| --- | --- |
| 완료된 요청과 높은 `end` 확률 | 현재 전사의 응답일 때만 설정된 무음 기준에서 종료합니다. |
| 주문번호를 읽거나 정정하면서 잠시 멈춤 | `hold` 또는 로컬 미완성 기준으로 계속 듣고, 최종 번호로 한 번만 조회합니다. |
| Jev 요청 중 고객이 말을 재개하거나 전사가 바뀜 | 이전 `end`가 턴을 종료하지 않습니다. |
| 키 없음, 타임아웃, 낮은 확률, 잘못된 JSON, HTTP 오류 | 해당 경로의 기본 동작으로 돌아갑니다. |
| 업무 분류에서 높은 확률이지만 식별자나 확인 상태 부족 | 실제 도구 호출 전에 필수 정보와 상태를 검사합니다. |

### Jev 실측과 단계적 적용

루트 텍스트 실측은 `TYPESAFE_API_KEY`와 합성 입력만 사용합니다. 결과 JSON에서 타임아웃·낮은 확률·응답 지연을 확인합니다. `--repeats 3`은 22개 입력에 대해 총 66회 요청합니다.

```powershell
.\.venv\Scripts\python.exe -B bench/jev_probe.py --repeats 3 --timeout 0.25 --out runs/jev-korean-text.json
```

프로토타입 서버는 `DIRECT_ON_PROTOTYPE`에서 `python agent_server/server.py --port 9300`으로 실행합니다. 전체 서버를 실서비스에 연결하려면 OpenAI·Soniox·Gemini 키도 필요합니다. 전사 종료와 업무 분류는 하나씩 켜면 어떤 설정이 결과를 바꿨는지 비교하기 쉽습니다. [프로토타입 실행 문서](../README.md#실행-방법)

```dotenv
# EOT 비교
EOT_MODE=jev
CASE_ROUTER=off
TYPESAFE_MODEL=jev-latest
```

```dotenv
# 업무 분류 비교
EOT_MODE=silence
CASE_ROUTER=jev
CASE_PROMPT=full
ANNOUNCE=off
```

음성 비교는 같은 녹음으로 무음 기준과 Jev 적용 조건을 반복 실행합니다. 조기 끊김, 주문번호 정정 반영 여부, 최종 도구 호출 인자, 첫 의미 있는 응답 시간, Jev 요청 p50·p95 지연을 확인합니다. 루트 실험 서버로 시작한다면 `shadow`의 판정 기록을 먼저 보고 `guard`와 `active`를 각각 비교할 수 있습니다. 기존 KO03 리플레이에서는 Jev가 두 실행 모두 세 차례 `end`를 반환했으며, 중간 쉼에서 계속 들은 것은 서버의 로컬 무음 기준이었습니다. 이 결과는 긴 쉼과 정정 번호를 재현할 입력으로 활용할 수 있습니다. [KO03 재현 명령과 결과](../../README.md#모의-스트리밍-전사로-jev와-gpt-선행-처리-검증)
