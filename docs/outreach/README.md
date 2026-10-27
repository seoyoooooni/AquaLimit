# 고객 검증 메일 (초안)

[사업 모델](../BUSINESS_MODEL.md) 7절의 첫 가설("GSA·수로 운영자가 금지 규칙의 충분성 점검·이행 점검에 돈을 낸다")을 확인하려는 메일이다. 팔려는 메일이 아니라 **질문하는 메일**이다. 답이 오면 고객 문제를 맞게 짚었는지 확인하고, 안 오면 2주 뒤 한 번만 다시 보낸다.

| 받는 곳 | 왜 | 공개된 연락 경로 |
|---|---|---|
| WDWA GSA (Westside District Water Authority) | 수로 MP 195~215 취수 금지를 만들고 6개월마다 연장 여부를 정하는 곳(최근 2026.5.14 재연장) | 사무실 (661) 633-9022 Ext.5 / 이사회 안건에 적힌 문의 메일 tmaggart@westsidewa.org (담당자 안내 요청) |
| DWR CASP (California Aqueduct Subsidence Program) | 수로 침하 보수·관측 사업 주체. 우리가 검증에 쓴 수로 GPS 관측망을 설치 중 | CASP 관측 사업 환경 검토 문서의 공개 담당 메일 Philip.Meyer@water.ca.gov (기술 담당자 안내 요청) |

보내기 전에 할 일
1. 첨부: [AquaLimit_one_pager.pdf](AquaLimit_one_pager.pdf) (영문 1장: 숫자 5개 + 금지 전후 위성 지도 + 한계). 두 메일에 같이 붙인다.
2. 보내는 사람 이름·학교·팀원 이름을 채운다.
3. "시뮬레이션 결과"라는 표현을 지우지 않는다. 실제 계량 기록 없이 만든 결과다.

---

## 메일 1. WDWA GSA

**Subject:** Student research on the CASP buffer moratorium — three questions (and free results to share)

Dear WDWA GSA staff,

I am a student in Korea on a team building AquaLimit, a tool that turns satellite subsidence data into well-by-well pumping plans. For a national space hackathon we chose the California Aqueduct between Mileposts 195 and 215 as our test area, because your GSA has taken some of the clearest action anywhere on aqueduct subsidence.

Using only public data (NASA OPERA InSAR, Landsat, DWR well completion reports, and the new CASP GPS stations), we calibrated a MODFLOW 6 model and found three things you may find useful. All of them are simulations, not measurements of any individual well:

1. With the 2.5-mile moratorium in place, our model expects roughly 45% of aqueduct points to stay under a 0.5 in/yr rate in an average 2026 irrigation season, and about 32% in a hot, dry one. Most of the remaining subsidence comes from pumping outside the buffer and from older clay compaction that is still catching up.
2. In an average-year scenario, keeping every aqueduct point under 0.5 in/yr by choosing reductions well by well cost about 20% less in our model than widening the buffer to the smallest fixed radius that also holds in a hot, dry year (about 6.9 km). We checked both options directly in the MODFLOW ensemble at the same reliability.
3. In NASA OPERA satellite data, subsidence inside the buffer did not slow relative to areas outside it during the first year after December 2024. We can share a map of the 1-km cells that are still sinking fastest, which might help you decide where to check flowmeters first.

Could you help us with three short questions?

- When the Board reviews the moratorium every six months, what information decides whether to renew, change, or widen it?
- Is a subsidence threshold for the Aqueduct under discussion for your GSP, and if so, is it similar to Wheeler Ridge-Maricopa's 0.5 inch per year?
- Would a satellite-based screening list of where to check compliance first be useful, or do the flowmeter reports already cover this?

I would be happy to send a one-page summary and the map. Even a two-line reply would help us a lot. Thank you for your time.

Best regards,
[Name], [University], AquaLimit team
[email] · [project link]

---

## 메일 2. DWR CASP

**Subject:** Independent check of TRE Altamira InSAR vs. your new Aqueduct GPS (MP 195–215) — sharing results

Dear Mr. Meyer,

I found your name on the CEQA filing for the CASP Subsidence and Groundwater Monitoring Project. If this is not the right contact, I would be grateful if you could forward this to CASP technical staff.

I am a student on a team building AquaLimit, a research tool that links satellite subsidence data to groundwater pumping decisions. Our test area is the California Aqueduct between Mileposts 195 and 215. We used the Aqueduct GPS stations P544, Q198, Q204, Q208, and Q213 (Nevada Geodetic Laboratory solutions) as independent validation data.

Two findings may be useful to your program:

1. **Data quality:** From 2023 onward, the DWR TRE Altamira InSAR series in our 500 m cells shows uplift at P544 and Q204, while GPS shows continued subsidence. The gap seems to start earlier: from December 2019 to December 2022, P544 GPS subsided 27.8 mm, OPERA 24.9 mm, and TRE 13.6 mm. NASA OPERA DISP-S1 for the same cells matches the direction of all five GPS stations. The median error is 8.3 mm for OPERA and 17.0 mm for TRE.
2. **Seasonal signal:** The Aqueduct GPS stations rise several millimeters in summer and drop in winter, while OPERA in the same cells shows summer subsidence. Comparing rates over periods that are not whole years can therefore over- or under-state changes. For example, measured over whole years, post-moratorium rates at Q204 and Q208 are similar to the year before, not faster.

We would welcome your view on two questions:

- Which Aqueduct reaches are the current priorities for subsidence monitoring and groundwater coordination with GSAs?
- Is there interest at CASP in tools that turn monitoring data into recommended pumping reductions near the Aqueduct, and who would use them?

I can send our comparison figures and the processing scripts. Thank you very much for your time.

Best regards,
[Name], [University], AquaLimit team
[email] · [project link]

---

## 한국어 요약 (팀 공유용)

**메일 1 (WDWA GSA):** 학생 팀이라고 밝히고, 공개 자료만으로 만든 시뮬레이션 결과 세 가지를 공유한다.
1. 금지만으로는 평년 45%, 가뭄 32% 지점만 기준(연 0.5인치)을 지킨다.
2. 우물별로 줄일 양을 고르는 방식이, 가뭄에도 지키도록 금지 반경을 넓히는 방식(6.9km)보다 약 20% 싸다(평년 시나리오, 두 방식 모두 MODFLOW로 같은 신뢰도 확인).
3. 위성 자료상 금지 후 1년 동안 금지구역 안이 밖보다 덜 나아졌다. 가장 빨리 꺼지는 칸 지도를 줄 수 있다.

질문은 세 가지다. 6개월마다 연장을 결정할 때 무엇을 보는지, 수로 침하 기준을 정하는 논의가 있는지, 위성 기반 '먼저 확인할 곳' 목록이 쓸모 있는지.

**메일 2 (DWR CASP):** 우리가 그쪽 GPS로 검증했다고 밝히고, 그쪽에 도움이 될 자료 품질 발견 두 가지를 먼저 준다.
1. DWR가 공급하는 InSAR가 2023년 이후 GPS와 반대 방향이고, OPERA는 맞는다. 어긋남은 2020년 무렵부터 시작된 것으로 보인다(P544 2019.12→2022.12: GPS 27.8, OPERA 24.9, DWR 13.6mm).
2. 수로 GPS에는 계절 변동이 있어 연 단위로 비교해야 한다.

질문은 두 가지다. 우선순위 구간이 어디인지, '관측 → 감량 권고' 도구에 관심이 있는지.

**답장 기록:** 받은 날짜, 누가, 핵심 답(연장 결정 근거, 기준값, 이행 점검 수요)을 [사업 모델](../BUSINESS_MODEL.md) 7절 '상태'에 적는다.
