# 선행연구 조사 A: LLM 기반 변이 테스트·테스트 생성

착수 → 완료 (2026-10-20). 후보 10건, 모두 직접 열어 원문(초록·본문 일부)에서 인용함.

## 주제 1. LLM이 변이를 죽이는 테스트 생성

### 1. Meta ACH — Mutation-Guided LLM-based Test Generation at Meta
- 제목 / Foster, Gulati, Harman, Harper, Mao, Ritchey, Robert, Sengupta (Meta) / 2025 / https://arxiv.org/abs/2501.12862 (보조: https://engineering.fb.com/2025/09/30/security/llms-are-the-key-to-mutation-testing-and-better-compliance/)
- 인용(arXiv 초록): "From these currently uncaught faults, ACH generates tests that can catch them, thereby \`killing' the mutants and consequently hardening the platform against regressions."
- 인용(arXiv 초록): "ACH also deploys an LLM-based equivalent mutant detection agent that achieves a precision of 0.79 and a recall of 0.47 (rising to 0.95 and 0.96 with simple pre-processing)."
- 대상: Android Kotlin 클래스 10,795개 (Meta 7개 플랫폼)
- 실행 판정 근거(Meta 블로그): "mutation testing reveals whether a test fails after inserting a mutation, indicating that the tests are not effectively checking the code's behavior." / "ACH also automatically generates unit tests that kill the mutants"
- 확인한 페이지: arXiv 초록 전체 + Meta 블로그 본문 일부

### 2. MuTAP — Effective Test Generation Using Pre-trained Large Language Models and Mutation Testing
- 제목 / Dakhel, Nikanjam, Majdinasab, Khomh, Desmarais / 2023 (arXiv, IST 저널판 2024) / https://arxiv.org/abs/2308.16557
- 인용: "Our goal is achieved by augmenting prompts with surviving mutants, as those mutants highlight the limitations of test cases in detecting bugs."
- 인용: "Furthermore, MuTAP achieves a Mutation Score (MS) of 93.57% on synthetic buggy code, outperforming all other approaches in our evaluation."
- 대상: Python (초록에서 Pynguin과 대조 언급)
- 실행 판정: 초록에서 테스트 실행 절차는 명시 안 됨 (Mutation Score 평가 수치만 있음) → "초록에서 확인 못 함"
- 확인한 페이지: arXiv 초록 전체

### 3. MUTGEN — Mutation-Guided Unit Test Generation with a Large Language Model
- 제목 / Guancheng Wang, Qinghua Xu, Lionel Briand, Kui Liu / 2025 (IEEE TSE 수락) / https://arxiv.org/abs/2506.02954
- 인용: "we propose MUTGEN, a mutation-guided, LLM-based test generation approach that incorporates mutation feedback directly into the prompt."
- 인용: "as demonstrated in our findings where some test suites achieve 100% coverage but only 4% mutation score"
- 대상: 204 subjects, 2 benchmarks (초록에 언어 명시 없음) → "원문에서 확인 못 함"
- 실행 판정: 초록에서 확인 못 함 (변이 점수를 측정 지표로 사용한다는 것만 명시)
- 확인한 페이지: arXiv 초록 전체

### 4. Mutation Testing via Iterative LLM-Driven Scientific Debugging
- 제목 / Straubinger, Kreis, Lukasczyk, Fraser / 2025 / https://arxiv.org/html/2503.08182v1
- 인용: "In the resulting approach, LLMs form hypotheses about how to kill specific mutants, and then iteratively generate and refine tests until they succeed, all with detailed explanations for each step."
- 인용: "The results demonstrate that LLMs, although requiring higher computational cost, consistently outperform Pynguin in generating tests with better fault detection and coverage."
- 대상: Python (Pynguin 비교, 모델 gpt-4o-mini-2024-07-18)
- 실행 판정: 실행 기반 반복으로 해석됨 — "iteratively generate and refine tests until they succeed" (평가 항목에 mutation score 포함)
- 확인한 페이지: arXiv HTML 본문 일부(초록)

## 주제 2. LLM으로 변이 생성·동치 변이 판정

### 5. Large Language Models for Equivalent Mutant Detection: How Far Are We?
- 제목 / Tian, Shu, Wang, Cao, Kamei, Chen / 2024 (ISSTA) / https://arxiv.org/abs/2408.01760
- 인용: "we conduct an empirical study on 3,302 method-level Java mutant pairs to comprehensively investigate the effectiveness and efficiency of LLMs for equivalent mutant detection."
- 인용: "LLM-based techniques significantly outperform existing techniques (i.e., the average improvement of 35.69% in terms of F1-score), with the fine-tuned code embedding strategy being the most effective."
- 대상: Java
- 실행 판정: 초록에서 확인 못 함 (동치성 판정 연구로 실행 기반 아님)
- 확인한 페이지: arXiv 초록 전체

### 6. On the Use of Large Language Models in Mutation Testing
- 제목 / Bo Wang, Mingda Chen, Youfang Lin, Mike Papadakis, Jie M. Zhang / 2024 (arXiv v2) / https://arxiv.org/html/2406.09843v2
- 인용: "LLMs generate more diverse mutations that are behaviorally closer to real bugs, which leads to approximately 19% higher fault detection than current approaches (i.e., 93% vs. 74%)."
- 인용: "A mutant is called killed or live depending on whether the test execution leads to a different program output (observable behavior) from that of the original program or not."
- 대상: Java (Defects4J 2.0 605 bugs, ConDefects 246 bugs; 총 851 real bugs)
- 실행 판정: 예 — 위 killed/live 정의가 테스트 실행 기반임을 명시
- 확인한 페이지: arXiv HTML 본문 일부(초록)

### 7. Re-evaluating Detection of Equivalent Mutants using LLMs: We Should Properly Measure How Far We Are
- 제목 / Tandon, Dündar, Gebru, Marinov, Lou, Wang / 2026 (PACMSE, ISSTA) / https://api.semanticscholar.org/graph/v1/paper/DOI:10.1145/3832250 (ACM 원문 dl.acm.org는 403으로 fetch 불가)
- 인용: "Contrary to prior findings, the proposed LLM-based EMD techniques suffer substantial performance degradation on the two additional datasets."
- 인용: "the studied LLMs tend to rely on a method-wise majority-voting shortcut rather than reasoning about the semantic effects of mutations."
- 대상: Java 외 추가 데이터셋 (원 Tian 2024와 다른 연산자·언어·프로젝트)
- 실행 판정: EMD(동치성 판정) 연구라 실행 기반 아님. 초록에 동치 판정 방식 명시 안 됨
- 확인한 페이지: Semantic Scholar API 경유 초록만 (ACM 원문 403)

## 주제 3. 실행 없이 킬/결과 예측 (및 그 한계)

### 8. Predicting Test Results without Execution
- 제목 / Andre Hora / 2024 (SIGSOFT FSE Companion) / https://api.semanticscholar.org/graph/v1/paper/DOI:10.1145/3663529.3663794 (데이터셋: https://github.com/andrehora/predicting-test-results-gpt-4)
- 인용: "we explore the capability of LLMs to predict test results without execution."
- 인용: "GPT-4 has a precision of 88.8%, recall of 71%, and accuracy of 81% in the test result prediction." / "Our findings suggest that GPT-4 still needs significant progress in predicting test results."
- 대상: Python Standard Library 테스트 200개 (ast, calendar, csv, gzip, string)
- 실행 판정: 실행 없이 예측하는 연구 자체 (패스/실패 예측이며 변이 킬 직접 대상은 아님)
- 확인한 페이지: Semantic Scholar API 초록 + GitHub README (ACM 페이지 403)

### 9. Contextual Predictive Mutation Testing (MutationBERT)
- 제목 / Jain, Alon, Groce, Le Goues / 2023 / https://arxiv.org/html/2309.02389v1
- 인용: "Research on Predictive Mutation Testing takes a different approach to scalable mutation testing, using machine learning to predict whether a mutant will be detected or not without actually running the tests."
- 인용: "On the cross project test set, the gold mutation score is 0.77. Seshat differs from this value significantly, with a mutation score of 0.63 (error of 0.14). MutationBERT is much closer, predicting a mutation score of 0.72 (error of 0.05)."
- 대상: 초록에서 언어 명시 안 됨 (CodeBERT 파인튜닝 기반) → "원문에서 확인 못 함"
- 실행 판정: 아니요 — "without actually running the tests" (변이 점수 추정)
- 확인한 페이지: arXiv HTML 본문 일부(초록)

## 주제 4. 서베이 (2023년 이후)

### 10. Software Testing with Large Language Models: Survey, Landscape, and Vision
- 제목 / Junjie Wang, Yuchao Huang, Chunyang Chen, Zhe Liu, Song Wang, Qing Wang / 2023 (arXiv, IEEE TSE 수락) / https://arxiv.org/abs/2307.07221
- 인용: "This paper provides a comprehensive review of the utilization of LLMs in software testing. It analyzes 102 relevant studies that have used LLMs for software testing, from both the software testing and LLMs perspectives."
- 인용: "The paper presents a detailed discussion of the software testing tasks for which LLMs are commonly used, among which test case preparation and program repair are the most representative."
- 대상: 범용 서베이
- 실행 판정: 해당 없음 (서베이)
- 확인한 페이지: arXiv 초록 전체

## 마지막 (3줄)

(가) 확인한 후보 10건 (모두 열어 인용 확보). 열지 못한 것: ACM DL 2건(위 7·8의 DOI 페이지, 403), figshare 재평가 페이지(202 응답), PDF 본문 3건(pdftotext 미설치로 텍스트 추출 불가 → arXiv 초록/API로 대체 확인).
(나) 후보를 못 찾은 주제: 없음 (주제 1~4 모두 ≥1건 확인).
(다) 추가로 읽어야 할 문서: LLMutantKiller (dl.acm.org/doi/10.1145/3832098), MutLLM (ieeexplore.ieee.org/document/11541746), GEM-LLM (sciencedirect.com/science/article/pii/S2667305326000153), A Comprehensive Review on Equivalent Mutant Detection Using ML (rodin.uca.es/handle/10498/37666, 2025), Tandon 외 원문 (dl.acm.org/doi/10.1145/3832250), MuTAP 저장소 (github.com/ExpertiseModel/MuTAP), Hora FSE 2024 원문(ACM 403).