#!/bin/bash
# 모델 구조 후보를 '연습 시험'(2019년까지 보정 → 2020~2022 위성 예측)으로 비교한다. 후보당 약 5~10분.
#   실행: cd ~/Desktop/AquaLimit/real && bash run_dev.sh 2>&1 | tee -a log_dev.txt
#   결과: SCOREBOARD.md (연습: 수로 RMSE가 v3보다 작고, 추세 연장보다 작으면 좋은 후보)
cd "$(dirname "$0")"
source .venv/bin/activate 2>/dev/null || true
export PYTHONUNBUFFERED=1
for v in ${@:-v4-corc v4-pfrac v4-emult v4-kv9}; do
  if [ -f "results_$v/results_dev.json" ]; then echo "== $v 이미 있음"; continue; fi
  echo "== [$v] 연습 시험 시작 $(date '+%H:%M')"
  AQ_VARIANT=$v python3 run_real.py --dev || echo "!! $v 실패"
done
python3 scoreboard.py > /dev/null && cat SCOREBOARD.md
