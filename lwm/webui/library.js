"use strict";

(() => {
  const get = id => document.getElementById(id);
  const dialog = get("record-dialog");
  let records = [];
  let current = null;
  let request = 0;
  let listRequest = 0;
  const date = new Intl.DateTimeFormat("ko-KR", {dateStyle: "medium", timeStyle: "short"});

  function description(record) {
    const parts = [date.format(record.date * 1000)];
    if (Number.isFinite(record.duration)) {
      const seconds = Math.floor(record.duration);
      parts.push(Math.floor(seconds / 60) + "분 " + seconds % 60 + "초");
    }
    if (record.model) parts.push(models[record.model] || record.model);
    if (record.diarizer) parts.push(diarizers[record.diarizer] || record.diarizer);
    return parts.join(" · ");
  }

  function render() {
    const trash = get("record-filter").value === "trash";
    const query = get("record-search").value.trim().toLocaleLowerCase();
    const matches = records.filter(r => r.trashed === trash &&
      [r.title, r.source].some(v => v.toLocaleLowerCase().includes(query)));
    get("record-count").textContent = matches.length + "건";
    get("library-message").textContent = matches.length ? "" :
      (query ? "검색 결과가 없습니다." : (trash ? "휴지통이 비어 있습니다." : "저장된 녹취가 없습니다."));
    const list = get("record-list");
    list.replaceChildren();
    for (const record of matches) {
      const li = document.createElement("li");
      const button = document.createElement("button");
      const title = document.createElement("strong");
      const info = document.createElement("span");
      title.textContent = record.title;
      info.className = "meta";
      info.textContent = description(record) + (record.minutes ? " · 회의록" : "");
      button.append(title, info);
      button.onclick = () => openRecord(record);
      li.append(button);
      list.append(li);
    }
  }

  async function switchView(saved) {
    const version = ++listRequest;
    get("new-panel").hidden = saved;
    get("saved-panel").hidden = !saved;
    get("new-view").setAttribute("aria-pressed", String(!saved));
    get("saved-view").setAttribute("aria-pressed", String(saved));
    if (!saved) return;
    get("library-message").textContent = "저장된 녹취를 불러오는 중…";
    try {
      const response = await fetch("/api/recordings");
      if (!response.ok) throw Error();
      const data = await response.json();
      if (version !== listRequest) return;
      records = data;
      render();
    } catch {
      if (version === listRequest) get("library-message").textContent = "녹취 목록을 불러오지 못했습니다. 지난 녹취를 눌러 다시 시도하세요.";
    }
  }

  function url(kind) {
    return "/api/recordings/" + current.id + "/" + kind;
  }

  async function showText(kind) {
    const version = ++request;
    const minutes = kind === "minutes";
    const link = get("record-download");
    get("read-transcript").setAttribute("aria-pressed", String(!minutes));
    get("read-minutes").setAttribute("aria-pressed", String(minutes));
    link.href = url(kind);
    link.textContent = minutes ? "MD 다운로드" : "TXT 다운로드";
    link.download = current.title + (minutes ? "-회의록.md" : "-녹취록.txt");
    get("record-error").textContent = "";
    get("record-text").textContent = "불러오는 중…";
    try {
      const response = await fetch(link.href);
      if (!response.ok) throw Error();
      const text = await response.text();
      if (version === request) get("record-text").textContent = text;
    } catch {
      if (version === request) get("record-text").textContent = "파일을 읽지 못했습니다.";
    }
  }

  function openRecord(record) {
    current = record;
    get("record-heading").textContent = record.title;
    get("record-info").textContent = [record.source, description(record)].filter(Boolean).join(" · ");
    get("record-trash").textContent = record.trashed ? "보관함으로 복원" : "휴지통으로 이동";
    get("record-delete").hidden = !record.trashed;
    get("title-form").hidden = true;
    get("read-minutes").hidden = !record.minutes;
    const audio = get("record-audio");
    audio.hidden = !record.audio;
    if (record.audio) audio.src = url("audio");
    dialog.showModal();
    showText("transcript");
  }

  async function edit(changes) {
    const editing = current.id;
    const controls = [get("record-trash"), get("record-delete"), get("record-rename"), get("title-form").querySelector("button")];
    controls.forEach(button => { button.disabled = true; });
    try {
      const response = await fetch("/api/recordings/" + editing, {
        method: "POST", headers: {"Content-Type": "application/json", "X-LWM-Upload": "1"},
        body: JSON.stringify(changes)
      });
      if (!response.ok) throw Error();
      const updated = await response.json();
      records = updated.deleted ? records.filter(r => r.id !== editing) :
        records.map(r => r.id === editing ? updated : r);
      render();
      if (updated.deleted) state().catch(() => {});
      if (current?.id !== editing) return;
      if (updated.deleted || "trashed" in changes) return dialog.close();
      current = updated;
      get("record-heading").textContent = updated.title;
      get("title-form").hidden = true;
      const minutes = get("read-minutes").getAttribute("aria-pressed") === "true";
      get("record-download").download = updated.title + (minutes ? "-회의록.md" : "-녹취록.txt");
      get("record-error").textContent = "";
    } catch {
      if (current?.id === editing) get("record-error").textContent = "변경 내용을 저장하지 못했습니다.";
    } finally {
      controls.forEach(button => { button.disabled = false; });
    }
  }

  get("new-view").onclick = () => switchView(false);
  get("saved-view").onclick = () => switchView(true);
  document.addEventListener("recordings-changed", () => {
    if (!get("saved-panel").hidden) switchView(true);
  });
  get("record-search").oninput = render;
  get("record-filter").onchange = render;
  get("record-close").onclick = () => dialog.close();
  dialog.onclose = () => {
    request++;
    current = null;
    const audio = get("record-audio");
    audio.pause();
    audio.removeAttribute("src");
    audio.load();
  };
  get("read-transcript").onclick = () => showText("transcript");
  get("read-minutes").onclick = () => showText("minutes");
  get("record-rename").onclick = () => {
    get("title-form").hidden = false;
    const input = get("record-title");
    input.value = current.title;
    input.select();
  };
  get("title-form").onsubmit = event => {
    event.preventDefault();
    const title = get("record-title").value.trim();
    if (title) edit({title});
  };
  get("record-trash").onclick = () => edit({trashed: !current.trashed});
  get("record-delete").onclick = () => {
    if (window.confirm("‘" + current.title + "’의 녹취록과 회의록을 영구 삭제할까요? 삭제 후 복원할 수 없습니다. 원본 녹음과 음성 캐시는 유지됩니다.")) {
      edit({delete: true});
    }
  };
})();
