"use strict"

const main = document.querySelector("#view")
const warning = document.querySelector("#warning")
const connection = document.querySelector("#connection")
const stages = [
  ["V1-A", "Campaign"], ["V1-B", "A 到 G-B；合成准备可并行"],
  ["V1-C0", "A + B"], ["V1-C1", "C0"], ["V1-C2", "C1 + C0"],
  ["V1-C3", "C2 + C0"], ["V1-D0", "C3 + C0"], ["V1-D1", "D0"],
  ["V1-D2", "D1"], ["V1-D3", "D0 + D1 + D2；Policy-E 须有 E3"],
  ["V1-E0", "D0 + D1 + D2 归因"], ["V1-E1", "E0"], ["V1-E2", "E1 + 混合域 WM"], ["V1-E3", "E2"],
]
const dates = new Intl.DateTimeFormat("zh-CN", {
  timeZone: "Asia/Shanghai", year: "numeric", month: "2-digit", day: "2-digit", hour: "2-digit", minute: "2-digit", second: "2-digit", hour12: false,
})
const numbers = new Intl.NumberFormat("zh-CN", {maximumFractionDigits: 4})
const good = new Set(["TECHNICAL_PASS", "PASS", "RUNNING", "TRAINING", "IN_PROGRESS", "VALIDATING", "OBSERVED", "AVAILABLE"])
const caution = new Set(["FAILED", "BLOCKED", "EXITED", "PID_REUSED", "STALE", "PARTIAL", "UNREADABLE_OR_PARTIAL"])
let doc = null
let disconnected = false

function node(tag, text, className) {
  const el = document.createElement(tag)
  if (text !== null && text !== undefined) el.textContent = text
  if (className) el.className = className
  return el
}
function first(object, ...paths) {
  for (const path of paths) {
    const value = path.split(".").reduce((a, key) => a?.[key], object)
    if (value !== null && value !== undefined && value !== "") return value
  }
  return null
}
function display(value) {
  if (value === null || value === undefined || value === "") return "UNKNOWN"
  if (typeof value === "number") return Number.isFinite(value) ? numbers.format(value) : "UNKNOWN"
  if (typeof value === "boolean") return value ? "是" : "否"
  if (typeof value === "object") return JSON.stringify(value)
  return String(value)
}
function date(value) {
  if (value === null || value === undefined) return "UNKNOWN"
  const parsed = new Date(typeof value === "number" ? value * 1000 : value)
  return Number.isNaN(parsed.getTime()) ? "UNKNOWN" : dates.format(parsed)
}
function badge(value) {
  const state = display(value)
  return node("span", state, "badge" + (good.has(state) ? " good" : caution.has(state) ? " caution" : ""))
}
function header(title, note) {
  const group = node("div", null, "view-header")
  const text = node("div")
  text.append(node("h2", title), node("p", note, "view-caption"))
  group.append(text)
  return group
}
function metrics(rows) {
  const grid = node("dl", null, "metrics")
  for (const [label, value] of rows) {
    const cell = node("div", null, "metric")
    const rendered = display(value)
    cell.append(node("dt", label), node("dd", rendered, ["UNKNOWN", "NOT_RUN"].includes(rendered) ? "missing" : ""))
    grid.append(cell)
  }
  return grid
}
function table(title, labels, rows, note) {
  const section = node("section", null, "panel")
  section.append(node("h3", title))
  if (note) section.append(node("p", note, "panel-note"))
  if (!rows.length) {
    section.append(node("p", "NOT_RUN", "empty"))
    return section
  }
  const wrap = node("div", null, "table-wrap")
  const grid = node("table")
  const head = node("thead")
  const heading = node("tr")
  for (const label of labels) {
    const cell = node("th", label)
    cell.scope = "col"
    heading.append(cell)
  }
  head.append(heading)
  const body = node("tbody")
  for (const row of rows) {
    const line = node("tr")
    for (const value of row) {
      const cell = node("td")
      if (value instanceof Node) cell.append(value)
      else cell.textContent = display(value)
      line.append(cell)
    }
    body.append(line)
  }
  grid.append(head, body)
  wrap.append(grid)
  section.append(wrap)
  return section
}
function facts(title, object, fields) {
  const entries = fields ? fields.map(([label, ...keys]) => [label, first(object, ...keys)]) : Object.entries(object || {})
  return table(title, ["项目", "记录"], entries.slice(0, 60))
}
function rows(object) {
  if (Array.isArray(object)) return object
  return Object.entries(object || {}).map(([id, value]) =>
    value && typeof value === "object" ? {id, ...value} : {id, state: value})
}
function stage(key) {
  const value = first(doc.ledger, "stages." + key, "stage_states." + key)
  return value && typeof value === "object" ? first(value, "state", "status") : value
}
function iteration(record) {
  const n = first(record, "learner_iteration", "iteration", "iter", "metrics.learner_iteration")
  return typeof n === "number" && Number.isFinite(n) && n >= 0 ? n : null
}
function measure(record, ...keys) {
  return first(record, ...keys.map(key => "metrics." + key), ...keys)
}
function points(records, keys) {
  const values = new Map()
  for (const entry of records) {
    const value = measure(entry, ...keys)
    const step = iteration(entry)
    if (step === null) continue
    values.set(step, typeof value === "number" && Number.isFinite(value) ? value : null)
  }
  return [...values.entries()].sort((a, b) => a[0] - b[0])
}
function chart(title, note, groups, gate) {
  const section = node("section", null, "panel")
  section.append(node("h3", title), node("p", note, "panel-note"))
  const valid = groups.flatMap(group => group.values).filter(([, value]) => value !== null)
  if (!valid.length) {
    section.append(node("p", "NOT_RUN · 当前记录没有可绘制的数据。", "empty"))
    return section
  }
  const canvas = node("canvas", null, "chart")
  canvas.setAttribute("role", "img")
  canvas.setAttribute("aria-label", title)
  const legend = node("ul", null, "chart-key")
  for (const {label} of groups) legend.append(node("li", label))
  if (gate !== undefined) legend.append(node("li", "审核门 " + gate + " mm"))
  section.append(canvas, legend)
  requestAnimationFrame(() => draw(canvas, groups, valid, gate))
  return section
}
function draw(canvas, groups, valid, gate) {
  if (!canvas.isConnected || canvas.clientWidth < 120) return
  const width = canvas.clientWidth
  const height = canvas.clientHeight
  const scale = Math.min(window.devicePixelRatio || 1, 2)
  canvas.width = Math.round(width * scale)
  canvas.height = Math.round(height * scale)
  const ctx = canvas.getContext("2d")
  ctx.scale(scale, scale)
  const xMin = Math.min(...valid.map(([step]) => step))
  const xMax = Math.max(xMin + 1, ...valid.map(([step]) => step))
  const values = valid.map(([, value]) => value)
  if (gate !== undefined) values.push(gate)
  const low = Math.min(...values)
  const high = Math.max(...values)
  const pad = Math.max((high - low) * .1, Math.abs(high) * .01, .0001)
  const left = Math.min(width / 4, 65)
  const right = width - 14
  const bottom = height - 28
  const x = step => left + (step - xMin) / (xMax - xMin) * (right - left)
  const y = value => bottom - (value - low + pad) / (high - low + 2 * pad) * (bottom - 12)
  ctx.font = "11px Inter, sans-serif"
  ctx.fillStyle = "#525e70"
  ctx.strokeStyle = "#d8dbe1"
  ctx.textAlign = "right"
  for (let i = 0; i < 5; i++) {
    const value = low - pad + i * (high - low + 2 * pad) / 4
    ctx.beginPath()
    ctx.moveTo(left, y(value))
    ctx.lineTo(right, y(value))
    ctx.stroke()
    ctx.fillText(display(value), left - 6, y(value) + 4)
  }
  ctx.textAlign = "left"
  ctx.fillText(display(xMin) + " 轮", left, height - 8)
  if (xMax - xMin > 1) {
    ctx.textAlign = "right"
    ctx.fillText(display(xMax) + " 轮", right, height - 8)
  }
  if (gate !== undefined) {
    ctx.strokeStyle = "#101b31"
    ctx.setLineDash([5, 4])
    ctx.beginPath()
    ctx.moveTo(left, y(gate))
    ctx.lineTo(right, y(gate))
    ctx.stroke()
    ctx.setLineDash([])
  }
  const colors = ["#355a91", "#007873", "#9f452b"]
  groups.forEach((group, index) => {
    ctx.strokeStyle = ctx.fillStyle = colors[index]
    ctx.lineWidth = 2
    ctx.beginPath()
    let contiguous = false
    for (const [step, value] of group.values) {
      if (value === null) contiguous = false
      else {
        if (contiguous) ctx.lineTo(x(step), y(value))
        else ctx.moveTo(x(step), y(value))
        contiguous = true
      }
    }
    ctx.stroke()
    for (const [step, value] of group.values) {
      if (value === null) continue
      ctx.beginPath()
      ctx.arc(x(step), y(value), 2.5, 0, Math.PI * 2)
      ctx.fill()
    }
  })
}
function budget() {
  const ledger = doc.ledger
  const caps = first(ledger, "available_budgets", "resource_budget", "budget.limits") || {}
  const used = first(ledger, "resource_usage", "consumed", "budget.used") || {}
  const reserved = first(ledger, "reserved", "budget.reserved") || {}
  const remaining = first(ledger, "remaining_budget", "budget.remaining") || {}
  const keys = [...new Set([...Object.keys(caps), ...Object.keys(used), ...Object.keys(reserved), ...Object.keys(remaining)])]
  return table("预算台账", ["资源", "上限", "已用", "预留", "剩余"],
    keys.map(key => [key, caps[key], used[key], reserved[key], remaining[key]]))
}
function overview() {
  const agents = rows(first(doc.agents, "agents", "work_packages") || first(doc.ledger, "agent_work_packages", "agents") || doc.agents)
  const run = doc.runtime
  return [header("总体进展", "阶段、运行证据、Agent 与预算"),
    metrics([["当前阶段", first(doc.ledger, "stage", "current_stage")], ["进程状态", run.state],
      ["实际 PID", run.pid], ["Run ID", run.run_id], ["最近心跳", date(run.heartbeat_at)],
      ["最后训练日志", date(run.last_training_log_at)]]),
    facts("运行身份", doc.ledger, [["Git SHA", "git_sha", "git_commit"], ["执行配置 SHA", "execution_sha256", "execution_config_sha256"],
      ["下一任务", "next_task", "next_legal_task"], ["台账更新", "updated_at"]]),
    table("阶段与前置", ["阶段", "前置", "状态", "技术门"], stages.map(([key, parent]) =>
      [key, parent, badge(stage(key)), first(doc.ledger, "stage_gate_evidence." + key, "gate_refs." + key)])),
    table("Agent 工作", ["Agent", "状态", "任务", "Run"], agents.map(agent =>
      [first(agent, "id", "agent", "name"), badge(first(agent, "state", "status")), first(agent, "task", "assignment", "scope"), agent.run_id])),
    table("待处理 Reviewer 发现", ["ID", "状态", "内容"], rows(first(doc.ledger, "reviewer_findings", "open_findings", "findings"))
      .filter(row => !["CLOSED", "RESOLVED"].includes(first(row, "state", "status")))
      .map(row => [row.id, badge(first(row, "state", "status")), first(row, "title", "summary", "finding")])),
    budget(),
    table("最近台账事件", ["上海时间", "事件", "阶段", "内容"], doc.events.slice(-20).reverse().map(row =>
      [date(first(row, "at", "recorded_at", "timestamp", "time")), first(row, "event_type", "kind", "event", "type", "action"),
        first(row, "stage", "facts.stage", "state.stage"), first(row, "summary", "message", "evidence_ref", "detail", "facts")]))]
}
function data() {
  const dataset = doc.data
  return [header("数据与环境", "RMB 处理、资格与并行规模预检"),
    metrics([["已处理拍", first(dataset, "processed_ticks", "ticks.processed")], ["合法参考", first(dataset, "legal_nominal_ticks", "ticks.legal")],
      ["计划拍", first(dataset, "planned_ticks", "ticks.planned")], ["S5 静态支持", first(dataset, "s5.targets", "s5_targets")],
      ["D10 动态支持", first(dataset, "d10.ticks", "d10_ticks")], ["实际并行环境", first(dataset, "environment_count", "qualification.environment_count")]]),
    facts("RMB 与谱系", dataset, [["Manifest", "manifest_sha256", "rmb_manifest_sha256"], ["Train", "splits.train", "train"],
      ["Validation", "splits.validation", "validation"], ["失败分布", "failure_breakdown", "failures"],
      ["动作类别", "motion_categories", "categories"], ["S5", "s5"], ["D10", "d10"]]),
    facts("FK、Servo、54 轴与安全", first(dataset, "tests", "physics_tests")),
    facts("并行预检", first(dataset, "qualification", "scale_preflight") || doc.ledger.preflight)]
}
function training() {
  const logs = doc.training
  const last = logs.at(-1) || {}
  const series = (label, ...keys) => ({label, values: points(logs, keys)})
  return [header("训练", "学习轮次、采样计数、回报与检查点"),
    metrics([["累计 learner iteration", iteration(last) ?? first(doc.ledger, "training.learner_iteration")],
      ["目标轮数", first(doc.ledger, "training.target_iterations", "target_iterations") ?? 30000],
      ["实际环境", measure(last, "environments", "num_envs", "environment_count")],
      ["物理 transitions", measure(last, "physical_transitions", "train_physical_transitions")],
      ["Imagined transitions", measure(last, "imagined_transitions")],
      ["轮/s", measure(last, "iteration_s", "iterations_per_second")], ["物理 transitions/s", measure(last, "physical_transitions_s", "physical_transitions_per_second")]]),
    chart("训练回报", "该图来自训练日志。独立物理精度见独立验证页。",
      [series("总回报", "reward_total", "reward", "return"), series("几何", "reward_geometry"), series("稳定", "reward_stability")]),
    chart("学习损失", "各损失按记录值绘制。", [series("World Model", "wm_loss"), series("Actor", "actor_loss"), series("Critic", "critic_loss")]),
    facts("更新、课程与权重", {last, ledger: doc.ledger}, [["WM 更新", "last.metrics.wm_updates", "last.wm_updates"],
      ["Actor 更新", "last.metrics.actor_updates", "last.actor_updates"], ["Critic 更新", "last.metrics.critic_updates", "last.critic_updates"],
      ["梯度范数", "last.metrics.gradient_norm", "last.gradient_norm"], ["学习率", "last.metrics.learning_rate", "last.learning_rate"],
      ["当前课程", "last.curriculum", "last.metrics.curriculum"], ["采样比例", "last.sampling", "last.metrics.sampling"],
      ["当前 checkpoint", "last.checkpoint", "ledger.current_checkpoint", "ledger.training.current_checkpoint"],
      ["最佳 checkpoint", "last.best_checkpoint", "ledger.best_checkpoint", "ledger.training.best_checkpoint"]]),
    facts("最新原始训练记录", first(last, "metrics") || last)]
}
function world() {
  return [header("World Model", "动力学资格、实际梯度与 watchdog"),
    facts("模型资格", doc.world_model, [["资格门", "gate", "status"], ["watchdog", "watchdog"], ["1 步", "steps.1", "one_step"],
      ["4 步", "steps.4", "four_step"], ["8 步", "steps.8", "eight_step"], ["q", "q_error"], ["qd", "qd_error"],
      ["十指误差", "fingertips"], ["腕部误差", "wrist_pose"], ["Pinch", "pinch"], ["Cross-hand", "cross_hand"], ["静态漂移", "static_drift"],
      ["动作梯度", "action_gradient", "gradient_gate"], ["54 轴激励", "excitation", "axis_coverage"],
      ["Imagination-return gap", "imagination_return_gap"], ["失格与恢复", "requalification", "recovery_history"]])]
}
const evalFields = [
  ["静态 P50 / P95 / P99", "static_p50_mm", "static_p95_mm", "static_p99_mm"],
  ["动态 P50 / P95 / P99", "dynamic_p50_mm", "dynamic_p95_mm", "dynamic_p99_mm"], ["S5 P95", "s5_p95_mm"],
  ["D10 P95", "d10_p95_mm"], ["困难片段 P95", "hard_p95_mm"], ["完整目标域 P95", "full_p95_mm"],
  ["3 / 5 / 10 / 15 mm 同拍命中", "hit_3_mm", "hit_5_mm", "hit_10_mm", "hit_15_mm"],
  ["每指误差", "per_finger_mm"], ["局部手型", "local_shape_mm"], ["Pinch", "pinch_mm"], ["Cross-hand", "cross_hand_mm"],
  ["腕相对 SE(3)", "relative_wrist"], ["停稳速度", "static_speed_mm_s"], ["恢复时间", "recovery_ms"], ["失败率", "failure_rate"],
  ["残差饱和", "saturation"], ["安全介入", "safety_interventions"], ["碰撞 / 硬停", "collisions", "hard_stops"],
  ["计分拍 / 计划拍", "scored_ticks", "planned_ticks"],
]
function validation() {
  const real = doc.evaluation.filter(entry => entry.evidence_kind === "ACTUAL_MUJOCO")
  const keys = ["B0", "CURRENT", "BEST"]
  const labels = ["B0 零残差", "当前 Policy", "最佳 Policy"]
  const grouped = keys.map(key => real.filter(entry => String(first(entry, "policy", "policy_role", "candidate_role")).toUpperCase() === key))
  const latest = grouped.map(records => records.at(-1) || {})
  const curve = field => grouped.map((records, index) => ({label: labels[index], values: points(records, [field])}))
  return [header("独立验证", "使用实际 MuJoCo 记录绘制物理精度"),
    chart("静态精度 · S5 max-ten P95", "误差单位为 mm，横轴为 learner iteration。", curve("s5_p95_mm"), 5),
    chart("动态精度 · D10 max-ten P95", "误差单位为 mm，横轴为 learner iteration。", curve("d10_p95_mm"), 10),
    table("实测对照", ["指标", ...labels], [
      ["轮次", ...latest.map(iteration)], ["Checkpoint", ...latest.map(entry => first(entry, "checkpoint_sha256", "checkpoint"))],
      ...evalFields.map(([label, ...fields]) => [label, ...latest.map(entry => fields.length === 1
        ? measure(entry, fields[0]) : fields.map(field => display(measure(entry, field))).join(" / "))]),
    ], "距离分位数单位为 mm。缺少物理计分时保留 UNKNOWN。"),
    facts("记录来源", doc.inputs.evaluation)]
}
function gib(bytes) {
  return typeof bytes === "number" ? display(bytes / 2 ** 30) + " GiB" : null
}
function hardware() {
  const h = doc.resources
  return [header("主机资源", "本机采样与已落盘资源预算"),
    metrics([["CPU 核心", h.cpu_count], ["1 分钟 load", h.load_average?.[0]], ["可用 RAM", gib(h.ram_available_bytes)],
      ["总 RAM", gib(h.ram_total_bytes)], ["可用磁盘", gib(h.disk?.free_bytes)], ["采样时间", date(h.observed_at)]]),
    table("GPU", ["索引", "型号", "负载 %", "已用 MiB", "总 MiB", "温度 °C"],
      h.gpu.map(row => [row.index, row.name, row["utilization.gpu"], row["memory.used"], row["memory.total"], row["temperature.gpu"]]), h.gpu_state),
    table("GPU 计算进程", ["PID", "进程", "显存 MiB"], h.gpu_processes.map(row => [row.pid, row.process_name, row.used_gpu_memory])),
    budget(), facts("资源告警", first(doc.status, "resource_alerts", "alerts"))]
}
function history() {
  return [header("迭代历史", "关联起因、配置差异、复验和回退"),
    table("独立审查与自主调整", ["轮次", "Checkpoint", "诊断", "配置差异", "Reviewer", "修改前后", "处置", "Run / Git"],
      doc.reviews.slice(-80).reverse().map(row => [iteration(row), first(row, "checkpoint", "checkpoint_sha256"),
        first(row, "diagnosis", "reason", "proposal"), first(row, "config_diff", "change"), first(row, "reviewer_result", "reviewer"),
        first(row, "before_after", "comparison", "metrics"), badge(first(row, "outcome", "decision")),
        [row.run_id, row.git_sha].filter(Boolean).join(" / ")]))]
}
function deployment() {
  const d = doc.deployment
  return [header("部署与 EgoPro", "现有 MuJoCo Teleop、双包与扩充支线"),
    metrics([["D0 离线门", stage("V1-D0")], ["现有链路", stage("V1-D1")], ["最终回归", stage("V1-D3")],
      ["EgoPro 准入", stage("V1-E0")], ["发布等级", first(d, "release_grade", "grade")]]),
    facts("运行与录包身份", d, [["现有软件图", "teleop_graph", "graph"], ["172809", "bags.172809", "bags.s0_sim_20261119_172809"],
      ["181022", "bags.181022", "bags.s0_sim_20261117_181022"], ["回退", "fallback", "rollback"], ["发布包", "release", "artifact"]]),
    table("T0 / T1 / T2 实际回放", ["录包", "方法", "轮次", "动态 P95 mm", "失败率", "证据"],
      doc.evaluation.filter(row => row.evidence_kind === "ACTUAL_TELEOP_MUJOCO").slice(-30).reverse().map(row =>
        [first(row, "bag_id", "bag"), first(row, "method", "policy"), iteration(row), measure(row, "dynamic_p95_mm"),
          measure(row, "failure_rate"), first(row, "evidence_ref", "artifact")])),
    facts("EgoPro", d, [["触发依据", "egopro.trigger"], ["新域覆盖", "egopro.coverage"], ["实际混合比例", "egopro.mix"],
      ["Policy-T / Policy-E 比较", "egopro.comparison"], ["最终选择", "egopro.selection"]])]
}
const pages = {overall: overview, data, training, world, validation, hardware, history, deployment}
function render() {
  const hash = window.location.hash.slice(1)
  const page = Object.hasOwn(pages, hash) ? hash : "overall"
  document.querySelectorAll("[data-view]").forEach(link => {
    if (link.dataset.view === page) link.setAttribute("aria-current", "page")
    else link.removeAttribute("aria-current")
  })
  if (doc === null) {
    main.replaceChildren(header("等待运行快照", "连接后会读取实际 Campaign 产物。"))
    return
  }
  const partial = Object.values(doc.inputs).filter(source => caution.has(source.state)).map(source => source.path)
  const message = disconnected ? "监控连接中断，保留最后一次有效快照。"
    : doc.runtime.state === "STALE" ? "运行心跳已过期。进度以最后一次有效记录为准。"
    : partial.length ? "部分文件正在写入或无法读取。涉及 " + partial.join("、") : ""
  warning.hidden = !message
  warning.textContent = message
  main.replaceChildren(...pages[page]())
}
async function refresh() {
  try {
    const response = await fetch("/api/snapshot", {cache: "no-store", signal: AbortSignal.timeout(6500)})
    if (!response.ok) throw new Error("Snapshot unavailable")
    doc = await response.json()
    disconnected = false
    connection.className = "badge good"
    connection.textContent = "已连接"
    document.querySelector("#refresh-time").textContent = "上海 " + date(doc.refreshed_at)
    document.querySelector("#campaign-id").textContent = first(doc.ledger, "campaign_id", "id") || "等待 Campaign 台账"
  } catch {
    disconnected = true
    connection.className = "badge caution"
    connection.textContent = "连接中断"
  }
  render()
  window.setTimeout(refresh, 5000)
}
window.addEventListener("hashchange", render)
let resizeTimer
window.addEventListener("resize", () => {
  window.clearTimeout(resizeTimer)
  resizeTimer = window.setTimeout(render, 150)
})
render()
refresh()
