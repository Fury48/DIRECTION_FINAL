#!/usr/bin/env bash
# 상담 서버 시작 — 최종 측정 때 운영진이 이 파일을 그대로 실행한다. 9000번 포트에서 WebSocket을 받으면 된다.
# 우리 상담원 서버(agent_server/server.py)를 띄운다. API 키는 맨 위 폴더의 .env (또는 agent_server/.env) 에서 읽는다.
cd "$(dirname "$0")"
exec python3 agent_server/server.py --port 9000
