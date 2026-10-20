# 선행연구 조사 B: GPU/Triton 커널의 테스트·검증·변이

착수 → 갱신 완료 (2026-10-20). 후보 10개, 전부 실제로 페이지를 열어 원문 확인함.

## 주제 1·2·3·4 후보 (확인한 URL만, 인용은 페이지 그대로)

### 1. MUTGPU — Massively Parallel, Highly Efficient, but What about the Test Suite Quality? Applying Mutation Testing to GPU Programs
- Qianqian Zhu, Andy Zaidman (TU Delft) / 2020 / ICST'20 / https://repository.tudelft.nl/record/uuid:dc9e6893-9a4c-4ecf-bc18-61fa7e5128d7
- 원문: "In addition to conventional mutation operators, we propose nine GPU-specific mutation operators based on the core syntax differences between CPU and GPU programming."
- 원문: "The results show that mutation testing can effectively evaluate the test quality of GPU programs: conventional mutation operators can guide the engineers to write simple direct tests, while GPU-specific mutation operators can lead to more intricate test cases which are better at revealing GPU-specific weaknesses."
- 대상: CUDA / 변이: 다룸(위 문장) / 확인: repository record 초록 전체 (논문 PDF는 다운로드 성공했으나 현 환경에 pdftotext가 없어 텍스트 미추출)

### 2. Testing and Mutation Testing for GPU Kernels (석사 논문)
- Yujie Liu (University of Twente) / 2023 / https://hgpu.org/?p=28799
- 원문: "In this research, we conduct a feasibility study on applying testing and mutation testing to GPU programming, adapting existing research methodology, hypotheses, experiments, and optimization methods to the specific use case of the GPU kernel, and exploring the GPU-native mutation testing theories and techniques."
- 원문: "We also discovered that mutation testing is able to quantify the testing quality for GPU kernels."
- 대상: GPU(언어 비특정, CUDA 중심) / 변이: 다룸(위 문장) / 확인: hgpu.org 요약 초록 전체 (essay.utwente.nl 기록 페이지는 JS 셸이라 제목 외 추출 불가)

### 3. Measuring the Checker: Mutation Analysis for GPU-Kernel Benchmark Oracles (KernelBench-M)
- Mingzhe Du, Anh Tuan Luu, Dong Huang, See-Kiong Ng / 2026 / arXiv:2609.22220 / https://arxiv.org/html/2609.22220v1
- 원문: "We introduce mutation analysis as an adequacy metric for kernel-benchmark oracles: deterministic rules inject 10,303 compilable faults into verified CUDA implementations of 188 KernelBench problems, 7,384 of them with an independent kill witness; any test protocol is scored by the fraction it detects."
- 원문: "The official check misses one in six witnessed faults (16.9%), deterministically, and the misses are skewed by family: 8.7% of arithmetic faults escape but 78.6% of precision faults do."
- 대상: CUDA (KernelBench 188문제, 규칙 124개) / 변이: 다룸(위 문장) / 확인: arxiv HTML — 초록 + 본문 일부(§1·§4·§10 요약)

### 4. KernelBench (LLM GPU 커널 생성 벤치마크)
- ScalingIntelligence / GitHub 레포 README / https://github.com/ScalingIntelligence/KernelBench
- 원문: "are correct ✅: check against reference torch operators `n_correctness` times on randomized inputs."
- 원문: "Since we need to capture **both** correctness and performance, we define a metric `fast_p`: fraction of tasks that are both correct and have a speedup greater than threshold `p`; speedup is computed as the ratio of PyTorch reference wall-clock time to generated kernel time."
- 대상: CUDA / 변이: 원문(README)에서 확인 못 함 — 정답성 판정은 무작위 입력·allclose 방식 / 확인: README 본문 일부

### 5. KernelBench-Verified: Do LLM-Generated Kernels Actually Beat PyTorch?
- Yunxiang Zhang 외 6인 / 2026 / arXiv:2607.16241 / https://arxiv.org/abs/2607.16241
- 원문: "concerning algorithmic correctness, models often exploit the narrow test distribution by hardcoding bypasses for specific tensor values."
- 원문: "We introduce KernelBench-Verified, an extended evaluation framework that incorporates a TF32-enabled baseline and a four-distribution hidden test suite."
- 대상: CUDA / 변이: 원문(초록)에서 확인 못 함 — 테스트 충분성은 은닉 분포 4종으로 접근 / 확인: 초록만

### 6. TritonBench: Benchmarking Large Language Model Capabilities for Generating Triton Operators
- Jianling Li 외 10인 / 2025 / ACL Findings / arXiv:2502.14752 / https://arxiv.org/abs/2502.14752
- 원문: "TritonBench features two evaluation channels: a curated set of 184 real-world operators from GitHub and a collection of operators aligned with PyTorch interfaces."
- 원문: "Unlike conventional code benchmarks prioritizing functional correctness, TritonBench also profiles efficiency performance on widely deployed GPUs aligned with industry applications."
- 대상: Triton / 변이: 원문(초록)에서 확인 못 함 — 초록에 테스트 입력 구성·충분성 서술 없음 / 확인: 초록만

### 7. Test-Input Generation for Tensor Programs: What Actually Finds Kernel Bugs
- Dipankar Sarkar / 2026 / arXiv:2606.27396 / https://arxiv.org/abs/2606.27396
- 원문: "Test-input generation for tensor kernels is folkloric. Most projects pick a representative shape and dtype, run a fixed-shape allclose-style check, and ship. We make the choices explicit and measure them."
- 원문: "Boundary-only shape sampling is the operationally safe winner: 78% recall on the 10 buggy kernels with 0% FP on the 16 controls."
- 대상: tensor kernels(Triton 명시 없음, GPU 인스턴스 RTX 3060) / 변이: 단어 자체는 없음 — "10 LLM-style buggy variants seeded with documented transcription patterns"의 시드 결함에 대한 재현율 측정 / 확인: 초록만

### 8. Triton 공식 문서 — Using the Interpreter (TRITON_INTERPRET)
- triton-lang.org 프로그래밍 가이드 3장 Debugging / https://triton-lang.org/main/programming-guide/chapter-3/debugging.html
- 원문: "This setting causes all Triton kernels to bypass compilation and be simulated by the interpreter using numpy equivalents of Triton operations. The interpreter processes each Triton program instance sequentially, executing operations one at a time."
- 원문: "It does not support operations on `bfloat16` numeric types." / "It does not support indirect memory access patterns such as: ptr = tl.load(ptr) x = tl.load(ptr)"
- 대상: Triton / 변이: 해당 없음 — 공식 문서가 밝힌 인터프리터 한계 2종 / 확인: 해당 절 본문 전체

### 9. Triton 이슈 #10658 — Float-to-int casts for out-of-range/NaN inputs are unspecified and diverge between JIT and interpreter
- triton-lang/triton GitHub / 2026 / https://github.com/triton-lang/triton/issues/10658
- 원문: "The same Triton program can produce different results in JIT vs. interpreter, making interpreter-based testing unreliable."
- 원문: "JIT path (`python/triton/language/semantic.py`): casts are lowered directly to `arith.fptosi` / `arith.fptoui` ... Interpreter path (`python/triton/runtime/interpreter.py`): uses `numpy.ndarray.astype()`, which has its own dtype-dependent behavior that differs from LLVM semantics."
- 대상: Triton / 변이: 해당 없음 — 인터프리터-컴파일 결과 불일치(HW f16→i8 표 비교표 포함) / 확인: 이슈 본문 전체

### 10. Triton 이슈 #10291 — [jit] [semantic inconsistency] `tl.argmax` outputs diffferently between jit and inpreter when meeting `nan`
- triton-lang/triton GitHub / 2026 / https://github.com/triton-lang/triton/issues/10291
- 원문: "`tl.argmax` appears to handle `NaN` differently between normal compiled execution and Triton interpreter mode."
- 원문: "compiled mode returns index 2, which is the position of the largest finite value 8.0." / "Interpreter mode returns index 0, which is the position of the NaN."
- 대상: Triton / 변이: 해당 없음 — 동일 입력에 대해 컴파일=2, 인터프리터=0 (재현 코드 포함) / 확인: 이슈 본문 전체

## 판정 불가·열었으나 내용 추출 실패
- https://azaidman.github.io/publications/zhuICST2020.pdf — 다운로드 성공(HTTP 200)했으나 현 환경에 pdftotext가 없어 텍스트 미추출(후보 1의 대체 페이지로 확인)
- https://essay.utwente.nl/essays/97661 — 301 리다이렉트 후 JS 응용 셸, 메타데이터 추출 불가(후보 2의 대체 페이지로 확인)

## 마지막 정리
(가) 확인한 후보 10개 / 열지 못한 URL 2개(MUTGPU PDF, UT 논문 기록 페이지 — 둘 다 대체 페이지로 확인).
(나) 4개 주제 중 후보를 못 찾은 주제: 없음. 단 주제 3은 후보가 1개뿐(arXiv:2606.27396)이고 초록에 Triton 명시가 없음.
(다) 추가로 읽을 문서: arXiv:2606.20128(The Correctness Illusion in LLM-Generated GPU Kernels — 주제 2 근접), arXiv:2610.05683(RESOLVE — KernelBench로 검증, 이슈 검출), arXiv:2608.12004(RealisticTritonBench — 수동 평가 스크립트의 보상 해킹 지적), triton-lang/triton #5895(TF32 vs 인터프리터 fp32 정밀도 차, 9.5e-07 vs 0.0187), triton-lang/triton/issue 6281(tl.sum int64 업캐스트 오동작).