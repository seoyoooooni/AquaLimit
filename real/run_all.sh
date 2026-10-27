#!/bin/bash
# AquaLimit 최종 계산 전체. 10코어 맥 기준 약 3시간 반.
# 중간에 멈추거나 오류가 나면 같은 명령으로 다시 실행: 이미 끝난 단계는 건너뛴다.
#   실행:  cd ~/Desktop/AquaLimit/real && bash run_all.sh 2>&1 | tee log_all.txt
cd "$(dirname "$0")"
source .venv/bin/activate 2>/dev/null || true
export PYTHONUNBUFFERED=1
step() {
  local name="$1" done_file="$2"; shift 2
  if [ -f "$done_file" ]; then echo "== [$name] 이미 끝남, 건너뜀 ($done_file)"; return 0; fi
  echo "== [$name] 시작 $(date '+%m/%d %H:%M')"
  if "$@"; then echo "== [$name] 끝 $(date '+%H:%M')"; else echo "!! [$name] 실패 — 위 오류를 Claude에게 보여 주세요"; exit 1; fi
}
step "1/12 검증 모델 v3 (2022년까지 보정 → 2023~26 GPS 채점)" results_v3/RESULTS.md                 python3 run_real.py
step "2/12 응답 검증: 자연 실험"                              results_validate_v3/RESULTS.md        python3 validate_response.py
step "3/12 운영 모델 v3ops (모든 자료로 보정)"                 results_v3ops/RESULTS.md              env AQ_VARIANT=v3ops python3 run_real.py
step "4/12 취수 금지 후 침하 원인 (운영 모델)"                 results_validate_v3ops/RESULTS.md     env AQ_VARIANT=v3ops python3 validate_response.py
step "5/12 정책 비교: 전면 금지 vs AquaLimit + 원인 분석"       results_policy_v3ops/RESULTS.md       env AQ_VARIANT=v3ops python3 policy_real.py
step "6/12 사업자 계획 E1~E4 (실제 신고 우물)"                 results_plan_v3ops/RESULTS.md         env AQ_VARIANT=v3ops python3 run_plan.py
step "7/12 날씨 시나리오 (기온·수로 배분율)"                   results_plan_v3ops/RESULTS_weather.md env AQ_VARIANT=v3ops python3 run_weather.py
step "8/12 표본 밖 평가"                                      results_plan_v3ops/RESULTS_oos.md     env AQ_VARIANT=v3ops python3 eval_plan_oos.py
step "9/12 위치 입력 AI"                                      results_ai_v3ops/RESULTS_loc.md       env AQ_VARIANT=v3ops python3 surrogate_loc.py
step "10/12 입력별 기여: Landsat 빼기"                        results_v3_nolandsat/results_e3only.json env AQ_VARIANT=v3_nolandsat python3 run_real.py --e3-only
step "11/12 입력별 기여: 기상 빼기"                           results_v3_noweather/results_e3only.json env AQ_VARIANT=v3_noweather python3 run_real.py --e3-only
step "12/12 입력별 기여: 수로 배분율 빼기"                     results_v3_noalloc/results_e3only.json   env AQ_VARIANT=v3_noalloc python3 run_real.py --e3-only
# 10/22~10/23 추가 단계(위 12단계가 끝난 결과를 쓴다)
step "13 자연 실험 1년 단위 (검증·운영 모델)"                  results_validate_v3ops/RESULTS_annual.md bash -c "python3 validate_annual.py && AQ_VARIANT=v3ops python3 validate_annual.py"
step "14 금지 반경 비교 (선형 응답)"                           results_policy_v3ops/RESULTS_radius.md   env AQ_VARIANT=v3ops python3 policy_ban_radius.py
step "15 금지 반경·계획 MODFLOW 확인·보정 (같은 신뢰도)"        results_policy_v3ops/RESULTS_refine.md   env AQ_VARIANT=v3ops python3 refine_plan.py
step "16 연습 시험 다시 채점 (정답 OPERA)"                     RESCORE_PRACTICE.md                      bash -c "AQ_VARIANT=v3 python3 rescore_practice.py && python3 rescore_practice.py --table"
step "17 구조 오차 다시 재기"                                  results_v3/RESULTS_uncertainty.md        env AQ_VARIANT=v3 python3 rescore_uncertainty.py
python3 scoreboard.py > /dev/null
echo "== 전부 끝 $(date '+%m/%d %H:%M')"
