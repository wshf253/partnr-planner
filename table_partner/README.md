# PARTNR 식탁 준비와 뒤정리

같은 사람과 여러 날 식탁을 차리는 실험이다. Qwen은 사람이 이미 나르기 시작한 접시를 함께 맡을지, 깨끗한 컵을 나를지 결정한다. 이동·집기·놓기·세척은 PARTNR의 기존 스킬이 실행한다. 파트너별 관찰 이력은 실행 중 메모리에 유지하고 매일 JSONL에 남긴다. 모델 가중치를 업데이트하는 과정은 없다.

## 연구실에서 실행하기

아래 명령은 `wshf253/partnr-planner`의 저장소 루트에서 실행한다. Habitat 환경과 HSSD, 식기, 로봇·사람 에셋은 저장소의 설치 절차를 따른다. [PARTNR 설치 문서](../INSTALLATION.md)

**터미널 1**에서 체크포인트를 지원하는 vLLM 환경으로 서버를 실행한다. 스크립트는 `/lab/ckpt/jsj412/Qwen3.8-27B`를 읽어 `qwen3.8-27B`라는 이름으로 서비스한다. 기본 주소는 `http://127.0.0.1:8010/v1`이다. GPU 두 장에 분산하려면 명령 뒤에 `--tensor-parallel-size 2`를 추가한다.

```bash
bash table_partner/serve_qwen.sh
```

**터미널 2**에서 PARTNR 환경으로 실행한다. 데이터셋은 한 번 생성해 세 파트너와 두 메모리 조건에 재사용한다.

```bash
conda activate habitat-llm
python -m table_partner generate --days 7 --seed 17
python -m table_partner run --days 7 --output outputs/table_partner/qwen-memory.jsonl
python -m table_partner run --days 7 --memory off --output outputs/table_partner/qwen-no-memory.jsonl
```

기존 실행 환경의 `PARTNR_LLM_BASE_URL`도 지원한다. 별도 서버를 우선 지정할 때는 `QWEN_BASE_URL`에 서버의 OpenAI 호환 `/v1` 주소를 지정한다. 서비스 이름이 다르면 `QWEN_MODEL`을, 인증이 필요하면 `QWEN_API_KEY`를 설정한다. 실험 로그 파일은 새 이름을 사용한다. 특정 파트너만 실행하려면 `--partner C`를 추가한다. 각 실행은 빈 이력으로 출발한다.

## 에피소드 구성

| 일차 | 실험 조건 |
|---|---|
| 1일 | 접시 두 개를 차리고 치운다. 사람이 먼저 접시를 놓으며 세 유형의 행동이 같다. |
| 2·3일 | 적은 일과 많은 일이 남은 상황에서 로봇이 한 번씩 접시 나르기에 합류한다. 사람이 맡을 수 있는 접시와 다른 일을 모두 남겨 둔다. |
| 4일 이후 | 날짜별로 식기의 수·출발 위치·컵의 청결 상태·이미 차려진 접시·준비 마감을 생성한다. 남은 일이 적은 날과 많은 날을 번갈아 배정한다. Qwen이 매일 한 번 개입 여부를 결정한다. |

A는 접시를 유지한다. B는 개입을 보면 컵으로 전환한다. C는 개입 직전 아직 식탁에 놓아야 할 식기가 5개 이상일 때 전환한다. 이 기준과 파트너 유형은 로봇 입력에서 숨긴다. 첫 버전은 반응을 결정적으로 설정했다. 이후의 분담은 관찰한 행동을 따라 정하고, 더러운 컵은 사람만 씻는다. 두 에이전트는 서로 다른 식기를 맡아 동시에 움직인다.

기본 장면은 `103997919_171031233`이다. 거실의 `table_1`을 식탁, 주방의 `table_0`을 뒤정리 장소로 쓴다. 시작 식기는 주방과 거실의 `table_2` 및 식탁에 배치한다. PARTNR 생성기가 배치한 물체의 실제 핸들을 평가식과 스킬에 연결한다. 식사 시점은 모든 목표 식기가 식탁에 모이고 컵이 깨끗한 순간으로 처리한다. 이후 같은 식기를 주방 탁자로 옮긴다.

## 결과 읽기

JSONL 한 줄은 한 파트너의 하루다. `observation`에는 로봇이 본 정보, `decision`에는 `join` 또는 `separate`, `evidence`에는 개입 전 남은 수와 관찰된 `stay` 또는 `switch`가 남는다. `probe=true`인 두 날의 선택은 실험자가 고정했다. 모델의 선택을 비교할 때는 4일차부터 같은 파트너와 날짜의 두 실행을 짝지으면 된다. `memory=off`에서는 Qwen에 전달하는 이력이 매번 비어 있다.

`ready_at`은 식탁 준비까지 걸린 스텝 수, `deadline`은 그날의 손님 도착 시점, `final_step`은 뒤정리까지 걸린 스텝 수다. `on_time`은 준비 마감 준수, `task_state_success`는 순서와 마지막 배치에 대한 성공, `success`는 두 조건을 모두 충족했음을 뜻한다. Habitat에서는 실제 시뮬레이터 스텝을 쓰며 기본 마감은 300·400·500스텝 중 하나다. 이 값은 초기 실험 설정으로, 같은 마감을 쓰려면 예를 들어 `--deadline 400`을 지정한다. 늦어도 뒤정리를 진행한다. 스킬 실패는 `error`에 기록된다.

## GPU 없이 로직 확인하기

```bash
python -m unittest -v table_partner.test_scenario
python -m table_partner run --backend symbolic --robot reference --output outputs/table_partner/reference.jsonl
```

기호 환경의 시간은 공동 운반에 한 단위, 세척이 포함되면 추가 한 단위다. 규칙 기반 로봇은 가장 비슷한 작업량에서 관찰한 전환 빈도로 선택한다. 모델 서버만 준비된 경우 `--backend symbolic --robot qwen`으로 물리 환경 없이 같은 의사결정을 시험할 수 있다.
