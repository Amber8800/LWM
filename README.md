# 로컬 한국어 회의록

RTX 4070 Ti SUPER 한 장에서 WAV·MP3·M4A 녹음을 처리하는 Windows CLI입니다. 결과 폴더에는 원본 시각과 화자 ID가 있는 전사본, 근거 발화가 연결된 회의록, 단계별 실행 시간이 저장됩니다.

## 브라우저에서 사용

프로젝트의 **회의 녹취.cmd**를 더블클릭하면 기본 브라우저에서 로컬 웹앱이 열립니다. 녹음을 놓고 STT 모델과 출력 형식을 고른 뒤 **녹취 시작**을 누르세요. 기본값은 Whisper Large V3와 녹취록 TXT입니다. 회의록을 선택하면 Gemma 3 12B가 요약한 Markdown도 다운로드할 수 있습니다.

기본 화자 모델은 NVIDIA Nemotron 3이며 최대 8명을 자동 감지합니다. 기존 Community-1 + ECAPA를 선택한 경우 실제로 말한 사람 수를 알면 **발언자 수**에 입력하세요. 빈칸이면 자동 감지합니다. **화자 분리 세부 설정**에서 코사인 거리 임계값을 조정할 수 있습니다. 기본값은 0.45이며 값을 높이면 같은 화자로 인정하는 거리 범위가 넓어집니다. 인원을 지정하면 반복해서 관측된 목소리를 기준으로 구간을 배정하고, 매칭 근거가 약한 구간은 미확정으로 표시합니다.

웹앱 주소는 http://127.0.0.1:8765 입니다. 실행 창을 닫으면 서버가 종료됩니다. 처리한 결과는 runs/web 아래에 저장되며 같은 서버가 실행 중이면 페이지를 새로고침해도 최근 작업을 볼 수 있습니다.

## 현재 PC에서 실행

프로젝트 루트에서 PowerShell을 엽니다. 실제 파일 경로를 따옴표 안에 넣습니다.

~~~powershell
.\.venv\Scripts\python -m lwm doctor
.\.venv\Scripts\python -m lwm run "D:\녹음\주간회의.m4a" --stt large-v3
~~~

출력 위치는 명령 마지막에 표시됩니다. 기본 폴더는 runs입니다. 평가 전에는 Large V3를 기준 모델로 사용합니다. 기본값은 --diarizer nemotron입니다. 기존 군집 방식을 사용하려면 --diarizer community를 선택하세요. --speakers 4 또는 --threshold를 단독 지정한 기존 명령도 community를 선택합니다. --threshold는 코사인 거리 기준의 화자 군집 임계값이고 초기값은 0.45입니다. 개발용 회의에서 보정한 값을 최종 평가에도 동일하게 적용하세요.

| 명령 | 용도 |
|---|---|
| asr 녹음.wav --stt komix-v2 | 선택한 모델로 타임스탬프가 있는 전사만 생성 |
| run 녹음.mp3 --transcript-only | 화자 분리와 전사까지 실행 |
| run 녹음.m4a --selection 평가결과/benchmark.json | 평가에서 선택한 모델·정밀도·프롬프트로 회의록 생성 |
| summarize 결과/transcript.json --names 이름표.json | 전사본을 재사용해 이름을 반영한 회의록 생성 |
| benchmark 평가셋.json --split test --output 평가결과 | 두 모델을 동일한 조건으로 3회씩 실행 |
| score 정답전사.json 결과/transcript.json | CER·용어 점수·DER·정렬된 어절의 화자 정답률 계산 |
| score-review 검수.json | 사람이 판정한 결정·할 일·담당자·근거의 정확도 집계 |

각 명령 앞에는 위 예시의 **.\\.venv\\Scripts\\python -m lwm**을 붙입니다. 이름표는 예를 들어 {"SPEAKER_01":"민수","SPEAKER_02":"지수"}를 UTF-8 JSON으로 저장합니다. --context 4096으로 회의록 문맥 예산을 줄일 수 있고, STT는 --precision int8_float16을 지원합니다. 양자화 설정은 별도 품질 평가 대상으로 취급합니다.

## 새 PC 준비

Python 3.11, NVIDIA 드라이버 및 uv가 설치된 Windows 환경에서 다음을 실행합니다.

~~~powershell
uv venv --python 3.11 .venv
uv pip install --python .venv/Scripts/python.exe --index-url https://download.pytorch.org/whl/cu128 torch==2.8.0 torchaudio==2.8.0
uv pip install --python .venv/Scripts/python.exe -e '.[gpu]'
.\setup-llama.ps1
.\setup-nemotron.ps1
.\.venv\Scripts\python -m lwm prepare large-v3 komix-v2 nemotron minutes
~~~

설치 스크립트는 버전이 고정된 공식 llama.cpp Windows CUDA 패키지의 SHA-256을 확인합니다. FFmpeg는 환경의 실행 파일 또는 imageio-ffmpeg에 포함된 실행 파일을 사용합니다. 이미 설치한 실행 파일은 LWM_FFMPEG, LWM_LLAMA_SERVER로 지정할 수 있습니다.

선택형 기존 화자 모델은 `python -m lwm prepare segment embed`로 추가합니다. Community-1과 Gemma는 공식 저장소에서 계정별 사용 조건 동의가 필요합니다. prepare는 Hugging Face 로그인, HF_TOKEN 환경 변수, 또는 프로젝트의 .env 안에 있는 HF_TOKEN이나 HUGGING_FACE_ACCESS_TOKEN을 사용할 수 있습니다. 모델 준비가 끝나면 오프라인으로 실행할 수 있습니다. 복사 배포 시 models 아래의 완성된 모델 폴더와 manifest.json을 함께 옮깁니다. GPU 추론은 로컬 경로만 읽으며 요약 서버는 127.0.0.1에 임시로 열립니다. [Community-1](https://huggingface.co/pyannote/speaker-diarization-community-1) · [Gemma](https://huggingface.co/google/gemma-3-12b-it-qat-q4_0-gguf)

### 모델 출처

| 이름 | 구성 |
|---|---|
| large-v3 | OpenAI 원본을 로컬에서 FP16 CTranslate2로 변환합니다. [원본](https://huggingface.co/openai/whisper-large-v3) |
| komix-v2 | seastar105의 OpenAI Whisper Medium 한국어 미세조정본을 같은 방식으로 변환합니다. 배포 용도에 따른 이용 조건은 원 배포자에게 확인해야 합니다. [원본](https://huggingface.co/seastar105/whisper-medium-komixv2) |
| nemotron | NVIDIA Nemotron-3-Diarization Q8_0과 NeMo-Speech.cpp 0.3.0 CUDA 런타임을 사용합니다. 녹음 한 개에 스트림 한 개를 열고 5초씩 읽어 화자 정보를 이어갑니다. [공식 가중치](https://huggingface.co/nvidia/Nemotron-3-Diarization) |
| segment | pyannote Community-1에서 segmentation/pytorch_model.bin만 가져옵니다. [공식 구성](https://huggingface.co/pyannote/speaker-diarization-community-1/tree/main) |
| embed | SpeechBrain ECAPA-TDNN VoxCeleb 가중치로 각 창의 단독 발화에서 임베딩을 추출하고, 회의 전체에서 평균 연결 군집화를 수행합니다. 짧거나 불확실한 구간은 미확정 처리합니다. [모델](https://huggingface.co/speechbrain/spkrec-ecapa-voxceleb) |
| minutes | Google Gemma 3 12B IT QAT Q4_0의 텍스트 GGUF를 사용합니다. [공식 GGUF](https://huggingface.co/google/gemma-3-12b-it-qat-q4_0-gguf) |

manifest.json에는 원본 배포처·리비전·라이선스 표기와 실제 파일 해시가 있습니다. 런타임에 필수 파일의 해시를 검사합니다. 원본 Community-1의 화자 임베딩과 PLDA는 배포 목록에서 제외됩니다. 이 프로그램의 화자 분리 성능은 교체한 구성으로 평가해야 합니다.

## 결과와 재개

- **transcript.txt**는 메모장에서 열 수 있는 UTF-8 녹취록입니다. 각 줄에 시작 시각, 화자, 발언을 표시합니다. 일반 실행과 --transcript-only 실행 모두 생성합니다.
- **transcript.json**에는 기준 PCM의 밀리초 단위 시각, 발화 ID, 화자 구간, 어절별 화자 후보·겹말 표시 및 원문이 있습니다. 실명은 이름표로 관리합니다.
- **minutes.md**는 주요 논의, 결정 사항, 담당자별 할 일, 근거 시각과 발화 ID를 표시합니다. **minutes.json**에는 인용한 발화 ID와 전사본에서 직접 가져온 원문이 있습니다. 생성 단계에서 해당 묶음에 포함된 ID만 선택할 수 있습니다. 인용의 존재는 자동 검증하고 의미가 맞는지는 검수자가 판단합니다.
- **metrics.json**은 단계별 경과 시간·재사용 여부·GPU의 샘플링 피크·전체 RTF를 담습니다. VRAM 수치는 화면 출력과 다른 프로세스를 포함한 장치 전체의 관측치이며 짧은 순간 피크는 놓칠 수 있습니다. 운영 예산은 14 GiB입니다.
- 같은 입력과 설정으로 재실행하면 디코딩, 완료된 화자 단계, 약 10분 단위 STT 결과와 요약 묶음을 재사용합니다. 녹음·모델·설정이 달라지면 대응되는 캐시를 분리합니다. 소스와 중간 PCM·발화도 로컬 디스크에 남으므로 결과 및 캐시 폴더를 녹음과 같은 기준으로 관리하세요.

## 품질 평가

회의 ID별로 개발·최종 평가를 분리합니다. 적어도 12개 실제 회의에서 원거리 마이크, 겹말, 한영 혼용, 사내 용어 등을 포함하고 사람이 교정한 전사를 작성합니다. 아래 예시에서 audio는 JSON 파일 기준 상대 경로입니다. terms의 한 배열이 같은 용어의 허용 표기입니다. 기준 전사에 등장한 횟수로 분모를 계산합니다.

~~~json
{
  "terms": [["GPU", "지피유"], ["CI/CD", "씨아이씨디"]],
  "meetings": [
    {"id":"회의-01","split":"test","audio":"녹음/회의-01.wav","reference":"GPU를 교체하기로 했습니다."}
  ]
}
~~~

기존 클로바노트 전사가 있으면 해당 회의에 clova_text 문자열을 추가해 동일한 기준으로 비교합니다. 두 모델 모두 FP16·beam 5·배치 1로 동일한 최대 30초 음성 창을 한 번씩 디코딩합니다. 약 25초 지점에서 음량이 낮은 곳을 골라 자르고, 양끝 문맥과 어절 중점으로 경계를 합칩니다. 한국어 생성과 단어 타임스탬프를 함께 사용합니다. 보고서에는 원문 대비 편집 수, 띄어쓰기 포함·제외 CER, 용어 정답률·오생성, 3회 개별 출력, 로딩·추론 시간, 전체 작업자 시간 중앙값, RTF, 피크 VRAM을 기록합니다. CER 최저 모델에서 CER 차이가 0.5%p 이내이고 용어 정답률 하락이 1%p 이내인 후보 중 RTF가 낮은 모델을 선택합니다. 용어가 없는 평가셋에서는 용어 지표가 비어 있으므로 실제 선정용 데이터에는 기술 용어를 포함하세요.

화자 평가용 정답은 transcript.json과 같은 turns 및 utterances[].words 형식으로 교정합니다. score는 최적 화자 매핑, 경계 허용 0초, 겹말 포함으로 측정합니다. 단어 화자 정답률은 글자가 일치해 정렬된 어절만 분모에 넣습니다. 겹말·500 ms 이하의 화자 교대 어절을 구분해 집계합니다. reference_timing_speaker_accuracy는 정답 어절 시각을 사용했을 때의 화자 정답률입니다. 변환 전후 전사와 단어 시각은 tests/check_conversion.py에 30초 이하 검증 파일, --stt와 --output을 지정해 대조할 수 있습니다.

회의록 검수자는 각 시스템 항목에 대응하는 정답 ID 또는 null, 근거 유효 여부를 기록합니다. 할 일에는 담당자 지정 여부, 정답 일치 여부, 근거 유무도 표시합니다. 목표는 결정·할 일 정밀도 각각 90% 이상, 근거 없는 담당자 지정 0건, 2시간 회의 전체 처리 RTF 1.0 이하입니다.

~~~json
{
  "decisions": {"reference_ids":["D1"],"predictions":[{"matched_reference":"D1","evidence_valid":true}]},
  "actions": {"reference_ids":["A1"],"predictions":[{"matched_reference":"A1","evidence_valid":true,"owner_assigned":true,"owner_correct":true,"owner_supported":true}]}
}
~~~

회귀 테스트는 프로젝트 루트에서 실행합니다.

~~~powershell
.\.venv\Scripts\python -m unittest discover -s tests -v
~~~
