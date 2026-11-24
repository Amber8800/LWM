"use strict";
const el = id => document.getElementById(id);
let selected = null;
let busy = true;
let timer;
const models = {"large-v3":"Whisper Large V3", "komix-v2":"Whisper Medium KoMix V2", "buzzasr-korean":"BuzzASR Korean · 빔 5"};
const diarizers = {nemotron:"NVIDIA Nemotron 3", community:"Community-1 + ECAPA"};
function controls(locked) {
  busy = locked;
  el("browse").disabled = el("model").disabled = locked;
  el("diarizer").disabled = locked;
  const community = el("diarizer").value === "community";
  el("community-settings").hidden = !community;
  el("speakers").disabled = el("threshold").disabled = locked || !community;
  document.querySelectorAll("[name=mode]").forEach(x => x.disabled = locked);
  el("start").disabled = locked || !selected;
}
function error(message) {
  el("error").textContent = message;
  el("error").hidden = !message;
}
function choose(files) {
  if (busy || !files.length) return;
  const file = files[0];
  if (files.length !== 1 || !/\.(wav|mp3|m4a)$/i.test(file.name) || !file.size) {
    error("내용이 있는 WAV, MP3, M4A 파일 한 개를 선택하세요.");
    return;
  }
  selected = file;
  el("file-name").textContent = file.name;
  el("file-meta").textContent = (file.size / 1048576).toLocaleString("ko-KR", {maximumFractionDigits:2}) + " MB";
  el("browse").textContent = "파일 변경";
  error(""); controls(false);
}
el("browse").onclick = () => el("file").click();
el("file").onchange = e => choose(e.target.files);
for (const event of ["dragover","dragleave","drop"]) {
  el("drop").addEventListener(event, e => {
    e.preventDefault();
    el("drop").classList.toggle("drag", !busy && event === "dragover");
    if (event === "drop") choose(e.dataTransfer.files);
  });
}
function show(job) {
  el("job").hidden = false;
  const percent = job.state === "done" ? 100 : job.percent;
  el("progress-panel").hidden = !Number.isFinite(percent);
  if (Number.isFinite(percent)) {
    el("progress").value = percent;
    el("percent").textContent = percent + "%";
    el("progress-detail").textContent = job.detail || "";
  }
  el("status").textContent = job.state === "error" ? "처리 실패" : job.message;
  el("job-model").textContent = [models[job.model], diarizers[job.diarizer || "community"], job.speakers && ("발언자 " + job.speakers + "명")].filter(Boolean).join(" · ");
  el("job-name").textContent = job.name;
  error(job.state === "error" ? job.message : "");
  const downloads = el("downloads");
  downloads.replaceChildren();
  for (const kind of job.files) {
    const a = document.createElement("a");
    a.className = "download";
    a.href = "/api/jobs/" + job.id + "/" + kind;
    a.textContent = kind === "txt" ? "녹취록 TXT 다운로드" : "회의록 MD 다운로드";
    downloads.append(a);
  }
  controls(job.state === "busy");
  clearTimeout(timer);
  if (busy) timer = setTimeout(() => poll(job.id), 1200);
  else if (job.files.length) document.dispatchEvent(new Event("recordings-changed"));
}
async function state() {
  const res = await fetch("/api/state");
  if (!res.ok) throw Error();
  const data = await res.json();
  if (data.latest) show(data.latest);
  else { el("job").hidden = true; error(""); controls(false); }
}
async function poll(id) {
  try {
    const res = await fetch("/api/jobs/" + id);
    if (res.status === 404) return await state();
    if (!res.ok) throw Error();
    show(await res.json());
  } catch {
    error("서버 연결을 확인하는 중입니다.");
    timer = setTimeout(() => poll(id), 2500);
  }
}
el("form").onsubmit = async e => {
  e.preventDefault();
  if (busy || !selected) return;
  const threshold = el("threshold").valueAsNumber;
  const diarizer = el("diarizer").value;
  if (diarizer === "community" && !(threshold > 0 && threshold < 2)) {
    el("threshold").closest("details").open = true;
    error("목소리 거리 임계값은 0보다 크고 2보다 작게 입력하세요.");
    el("threshold").focus();
    return;
  }
  const model = el("model").value;
  const mode = document.querySelector("[name=mode]:checked").value;
  controls(true); error("");
  el("job").hidden = false;
  el("job-name").textContent = selected.name;
  el("job-model").textContent = models[model];
  el("status").textContent = "파일 보내는 중";
  el("progress-panel").hidden = true;
  el("downloads").replaceChildren();
  try {
    const res = await fetch("/api/jobs", {method:"POST", headers:{
      "X-LWM-Upload":"1", "X-Filename":encodeURIComponent(selected.name),
      "X-Mode":mode, "X-Model":model, "X-Diarizer":diarizer,
      ...(diarizer === "community" ? {"X-Speakers":el("speakers").value, "X-Speaker-Threshold":String(threshold)} : {})
    }, body:selected});
    const job = await res.json();
    if (res.status === 409) { await state(); return; }
    if (!res.ok) throw Error(job.error || "파일 전송에 실패했습니다.");
    show(job);
  } catch (err) {
    error(err.message);
    el("status").textContent = "전송 상태 확인 필요";
    controls(false);
  }
};
state().catch(() => { error("로컬 서버에 연결할 수 없습니다. 페이지를 새로고침하세요."); });

el("form").addEventListener("change", () => {
  controls(busy);
  el("pipeline-diarizer").textContent = diarizers[el("diarizer").value];
  el("pipeline-stt").textContent = models[el("model").value];
  const minutes = document.querySelector("[name=mode]:checked").value === "minutes";
  el("pipeline-llm").textContent = minutes ? "Gemma 3 12B · Q4_0" : "사용 안 함";
});
