// 掌柜智库 — 原生聊天壳(MVP 占位,RAG 管线接入后替换占位 fetch)。
const chat = document.getElementById("chat");
const form = document.getElementById("composer");
const input = document.getElementById("question");
const badge = document.getElementById("health-badge");

function push(role, text) {
  const div = document.createElement("div");
  div.className = `msg ${role}`;
  div.textContent = text;
  chat.appendChild(div);
  chat.scrollTop = chat.scrollHeight;
}

async function refreshHealth() {
  try {
    const res = await fetch("/api/health");
    const data = await res.json();
    const allOk = ["milvus", "mongo", "minio"].every(
      (k) => data[k] && data[k].status === "ok"
    );
    badge.textContent = allOk ? "已连接" : "部分不可用";
    badge.style.color = allOk ? "#58d68d" : "#f5b041";
  } catch (err) {
    badge.textContent = "后端离线";
  }
}

form.addEventListener("submit", async (e) => {
  e.preventDefault();
  const q = input.value.trim();
  if (!q) return;
  push("user", q);
  input.value = "";
  push("bot", "管线尚未接入(#3 起),占位应答。");
});

push("sys", "欢迎使用掌柜智库 — 请键入问题。");
refreshHealth();
setInterval(refreshHealth, 30_000);
