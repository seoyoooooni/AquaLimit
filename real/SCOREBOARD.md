# 채점표

개선 여부는 **연습 시험**(2019년까지 보정 → 2020~2022 위성 예측, 수로 2km 칸)으로만 판단한다. **최종 시험**(2022년까지 보정 → 2023~2026 GPS)은 모델을 확정할 때만 본다. 단위 mm.

같은 연습 시험에서 '추세 연장'(2015~2019 위성 추세를 그대로 늘림)보다 작아야 물리모델이 데이터 추세보다 낫다고 말할 수 있다.

**10/23 연습 시험 정답 변경: DWR InSAR → NASA OPERA.** 연습 기간(2020~22) GPS가 있는 유일한 곳 P544에서 GPS 27.8mm, OPERA 24.9mm, DWR 13.6mm 침하 — DWR가 절반만 잡는다(2023년 이후 GPS와 반대로 가는 문제가 2020년 무렵부터 시작). 같은 보정 결과(H2019)를 다시 채점만 했다(rescore_practice.py, RESCORE_PRACTICE.md). DWR 정답 열은 기록으로 남긴다. 'GPS P544' = 연습 기간 P544 GPS 월 값 36개 RMSE(추세 연장 4.1mm, 변화 0 12.5mm).

| 폴더 | 변형 | 파라미터 수 | 최종: GPS E3 | 최종: 90% 포함률 | 최종: 범위 폭 | 연습(OPERA): 수로 RMSE | 연습(OPERA): 추세 연장 | 연습(OPERA): 편향 | 연습: GPS P544 | 연습(DWR, 10/22까지): 수로 RMSE | 연습(DWR): 추세 연장 | 연습(DWR): 편향 | 범위 끝 파라미터 | 자연 실험 방향 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| results_uniform | uniform | 15 | 20.5 | 93.5 | 65.3 |  |  |  |  |  |  |  |  |  |
| results_v3 | v3 | 21 | 21.2 | 99.2 | 97.4 | 14.0 | 10.4 | 2.0 | 21.0 | 25.1 | 27.2 | 17.9 | mSk_z0, Kf, rch | 1/4 |
| results_v3-opera | v3-opera | 21 | 21.8 |  |  | 15.5 | 10.4 | 2.8 | 29.7 | 26.9 | 27.2 | 18.7 | mSk_z0, mSk_z3, mSk_z6, Kf, rch, rf |  |
| results_v3_noalloc | v3_noalloc | 21 | 24.7 |  |  |  |  |  |  |  |  |  |  |  |
| results_v3_nolandsat | v3_nolandsat | 21 | 19.2 |  |  |  |  |  |  |  |  |  |  |  |
| results_v3_noweather | v3_noweather | 21 | 23.5 |  |  |  |  |  |  |  |  |  |  |  |
| results_v3ops | v3ops | 23 |  |  |  |  |  |  |  |  |  |  |  | 4/4 |
| results_v3ops-opera | v3ops-opera | 23 |  |  |  |  |  |  |  |  |  |  |  | 4/4 |
| results_v4 | v4 | 21 |  |  |  | 15.5 | 10.4 | 6.2 | 29.2 | 29.3 | 27.2 | 22.1 | mSk_z0, Kf, rch, rf |  |
| results_v4-aq | v4-aq | 23 |  |  |  | 24.9 | 10.4 | 19.4 | 28.2 | 43.3 | 27.2 | 35.2 | mSk_z0, Kf, rch, rf |  |
| results_v4-corc | v4-corc | 21 |  |  |  | 15.3 | 10.4 | 6.1 | 29.5 | 29.1 | 27.2 | 22.0 | mSk_z0, Kf, rch, rf |  |
| results_v4-emult | v4-emult | 22 |  |  |  | 15.0 | 10.4 | 5.5 | 27.0 | 28.6 | 27.2 | 21.4 | mSk_z0, Kf, rch, rf, emult |  |
| results_v4-kv9 | v4-kv9 | 28 |  |  |  | 29.7 | 10.4 | 22.4 | 55.2 | 47.7 | 27.2 | 38.4 | mSk_z0, rch, rf |  |
| results_v4-oil | v4-oil | 23 |  |  |  | 15.6 | 10.4 | 9.1 | 29.7 | 32.1 | 27.2 | 25.0 | mSk_z0, Kf, rch, rf |  |
| results_v4-oil-aq | v4-oil-aq | 25 |  |  |  | 22.9 | 10.4 | 17.8 | 28.3 | 41.4 | 27.2 | 33.7 | mSk_z0, Kf, rch, rf |  |
| results_v4-oil-hw | v4-oil-hw | 23 |  |  |  | 15.7 | 10.4 | 9.2 | 30.1 | 32.3 | 27.2 | 25.2 | mSk_z0, Kf, rch, rf |  |
| results_v4-oil-opera | v4-oil-opera | 23 |  |  |  | 15.2 | 10.4 | 6.1 | 29.9 | 29.6 | 27.2 | 22.1 | mSk_z0, mSk_z3, mSk_z6, Kf, rch, rf |  |
| results_v4-oil-pp | v4-oil-pp | 47 |  |  |  | 16.6 | 10.4 | 11.7 | 23.6 | 33.7 | 27.2 | 27.4 | mSk_p0, mSk_p1, mSk_p4, mSk_p6, mSk_p9, mSk_p12, mSk_p13, mSk_p16, mSk_p32, Kf, rch, rf |  |
| results_v4-oil-pp-opera | v4-oil-pp-opera | 47 |  |  |  | 15.8 | 10.4 | 9.8 | 24.6 | 31.8 | 27.2 | 25.4 | mSk_p0, mSk_p1, mSk_p4, mSk_p5, mSk_p6, mSk_p8, mSk_p9, mSk_p12, mSk_p13, mSk_p16, mSk_p23, mSk_p32, Kf, rch, rf |  |
| results_v4-pfrac | v4-pfrac | 22 |  |  |  | 15.5 | 10.4 | 6.4 | 29.4 | 29.5 | 27.2 | 22.3 | mSk_z0, Kf, rch, rf |  |
| results_v4-pp | v4-pp | 45 |  |  |  | 15.9 | 10.4 | 10.9 | 23.0 | 32.9 | 27.2 | 26.3 | mSk_p0, mSk_p1, mSk_p4, mSk_p9, mSk_p10, mSk_p16, mSk_p21, mSk_p23, mSk_p32, Kf, rch, rf |  |
| results_v4-pp-opera | v4-pp-opera | 45 |  |  |  | 15.1 | 10.4 | 9.0 | 25.3 | 30.9 | 27.2 | 24.3 | mSk_p0, mSk_p1, mSk_p4, mSk_p5, mSk_p7, mSk_p8, mSk_p9, mSk_p10, mSk_p12, mSk_p16, mSk_p21, mSk_p22, mSk_p23, mSk_p32, Kf, rch, rf |  |
