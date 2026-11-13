"use strict";

const corpus = window.WHOLEBODY_DATASET;
const byId = id => document.getElementById(id);
const sides = ["left", "right"];
const labels = {
  hand_pose: "手型", arm_motion: "手臂运动", arm_hand: "臂手联动", bi_hand: "双手协同",
  train: "训练", validation: "验证", test: "测试",
  manus_left: "MANUS 左手", manus_right: "MANUS 右手",
  tracker_left: "Tracker 左腕 · 标定后", tracker_right: "Tracker 右腕 · 标定后",
  tracker_left_raw: "Tracker 左腕 · 原始", tracker_right_raw: "Tracker 右腕 · 原始",
  tracker_safety: "Tracker 安全状态", arm_left_state: "左臂关节反馈", arm_right_state: "右臂关节反馈",
  hand_left_state: "左手关节反馈", hand_right_state: "右手关节反馈", arm_command: "双臂控制指令",
  hand_left_command: "左手控制指令", hand_right_command: "右手控制指令",
  robot_state: "机器人状态", teleop_state: "遥操状态",
  human_gap: "人体数据缺口", arm_ik_or_hold: "IK 未接受或保持", hand_sdk_unavailable: "手部重定向缺失",
  joint_bounds: "关节限位修正过大", position_tracking: "腕部位置误差超限",
  orientation_tracking: "腕部姿态误差超限", static_contact: "模型静态穿透超限",
  nonfinite_joint: "关节参考不完整", bimanual_start_unaligned: "双腕起始相对向量超限",
  hand_step_speed: "手指变化速度超限",
};
const colors = {left: "#55e3d2", right: "#ffbe80", muted: "#9eb8c1", grid: "#2b4750", ink: "#28414c"};
const mono = '"IBM Plex Mono", Consolas, monospace';
const radToDeg = 180 / Math.PI;
const views = {
  human: {yaw: -.19, pitch: .38, zoom: 1, mode: "openvr"},
  robot: {yaw: -.24, pitch: .39, zoom: 1, mode: "robot"},
  left: {yaw: -.34, pitch: .43, zoom: 1, mode: "hand"},
  right: {yaw: -.34, pitch: .43, zoom: 1, mode: "hand"},
};
const defaults = Object.fromEntries(Object.entries(views).map(([key, view]) => [key, {...view}]));
const integer = new Intl.NumberFormat("zh-CN");
let selected, data, frame = 0, playing = false, playAt = 0, playFrame = 0;
let animation, resizeId, humanBounds, robotBounds, handBounds, jointBounds, episodeData, motion;

function frameCount() { return motion?.frames ?? selected.frames; }

function format(value, digits = 1) {
  return Number.isFinite(value) ? value.toFixed(digits) : "—";
}

function element(tag, content, className) {
  const node = document.createElement(tag);
  if (content !== undefined) node.textContent = content;
  if (className) node.className = className;
  return node;
}

function name(key) { return labels[key] || key; }
function seconds(t) { return format(t, 2) + " s"; }
function percentage(a, b) { return b ? format(100 * a / b, 1) + "%" : "—"; }

function unpack(field) {
  const text = atob(field.data);
  const bytes = new Uint8Array(text.length);
  for (let i = 0; i < text.length; i++) bytes[i] = text.charCodeAt(i);
  let values;
  if (field.kind === "u8") {
    values = bytes;
  } else if (field.kind === "i16") {
    const packed = new Int16Array(bytes.buffer);
    values = Float32Array.from(packed, v => v === -32768 ? NaN : v * field.scale);
  } else {
    values = new Float32Array(bytes.buffer);
  }
  return {values, shape: field.shape, width: field.shape.slice(1).reduce((a, b) => a * b, 1)};
}

function sample(key, index = frame) {
  const field = data?.fields[key];
  return field?.values.subarray(index * field.width, (index + 1) * field.width);
}

function present(key, index = frame) { return Boolean(sample(key, index)?.[0]); }
function validPoint(v) { return v && Number.isFinite(v[0]) && Number.isFinite(v[1]) && Number.isFinite(v[2]); }
function point(values, index, stride = 3) { return values?.subarray(index * stride, index * stride + stride); }
function scaleVector(v, k) { return [v[0] * k, v[1] * k, v[2] * k]; }
function add(a, b) { return [a[0] + b[0], a[1] + b[1], a[2] + b[2]]; }

function newBounds() {
  return {lo: [Infinity, Infinity, Infinity], hi: [-Infinity, -Infinity, -Infinity]};
}

function include(bounds, p) {
  if (!validPoint(p)) return;
  for (let i = 0; i < 3; i++) {
    bounds.lo[i] = Math.min(bounds.lo[i], p[i]);
    bounds.hi[i] = Math.max(bounds.hi[i], p[i]);
  }
}

function boundsFor(keys, stride, minimum) {
  const bounds = newBounds();
  for (const key of keys) {
    const arr = data.fields[key]?.values;
    if (!arr) continue;
    for (let i = 0; i < arr.length; i += stride) include(bounds, arr.subarray(i, i + 3));
  }
  if (!Number.isFinite(bounds.lo[0])) return {center: [0, 0, 0], radius: minimum, low: [0, 0, 0]};
  const center = bounds.hi.map((hi, i) => (hi + bounds.lo[i]) / 2);
  const radius = Math.max(minimum, .5 * Math.hypot(...bounds.hi.map((hi, i) => hi - bounds.lo[i])));
  return {center, radius, low: bounds.lo};
}

function surface(id) {
  const canvas = byId(id);
  const width = canvas.clientWidth;
  const height = canvas.clientHeight;
  const ratio = Math.min(window.devicePixelRatio || 1, 2);
  if (canvas.width !== Math.round(width * ratio) || canvas.height !== Math.round(height * ratio)) {
    canvas.width = Math.round(width * ratio);
    canvas.height = Math.round(height * ratio);
  }
  const ctx = canvas.getContext("2d");
  ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
  ctx.clearRect(0, 0, width, height);
  return {ctx, width, height};
}

function projector(surface, view, bounds, padding = .88, yOffset = 0) {
  const {width, height} = surface;
  const {center, radius} = bounds;
  const scale = padding * Math.min(width, height) * view.zoom / (2 * radius);
  const cy = Math.cos(view.yaw), sy = Math.sin(view.yaw);
  const cp = Math.cos(view.pitch), sp = Math.sin(view.pitch);
  return p => {
    const x = p[0] - center[0], y = p[1] - center[1], z = p[2] - center[2];
    const [u, v, up] = view.mode === "openvr" ? [-z, -x, y] : view.mode === "robot" ? [-x, y, z] : [x, y, z];
    const horizontal = u * cy - v * sy;
    const depth = u * sy + v * cy;
    return [width / 2 + scale * horizontal, height / 2 + yOffset - scale * (up * cp - depth * sp), depth * cp + up * sp];
  };
}

function line(ctx, a, b, color, width = 1) {
  ctx.strokeStyle = color;
  ctx.lineWidth = width;
  ctx.beginPath();
  ctx.moveTo(a[0], a[1]);
  ctx.lineTo(b[0], b[1]);
  ctx.stroke();
}

function dot(ctx, p, color, radius = 3) {
  ctx.fillStyle = color;
  ctx.beginPath();
  ctx.arc(p[0], p[1], radius, 0, Math.PI * 2);
  ctx.fill();
}

function grid(ctx, project, bounds, up, pitch) {
  const axes = [0, 1, 2].filter(axis => axis !== up);
  const center = bounds.center.slice();
  center[up] = bounds.low[up] - pitch * .1;
  const n = Math.min(12, Math.max(3, Math.ceil(bounds.radius / pitch)));
  for (let tick = -n; tick <= n; tick++) {
    for (const [i, j] of [axes, axes.slice().reverse()]) {
      const a = center.slice(), b = center.slice();
      a[i] += tick * pitch;
      b[i] += tick * pitch;
      a[j] -= n * pitch;
      b[j] += n * pitch;
      line(ctx, project(a), project(b), colors.grid);
    }
  }
}

function quaternionAxes(pose) {
  const norm = Math.hypot(pose[3], pose[4], pose[5], pose[6]);
  const [x, y, z, w] = Array.from(pose.slice(3, 7), v => v / norm);
  return [
    [1 - 2 * (y*y + z*z), 2 * (x*y + w*z), 2 * (x*z - w*y)],
    [2 * (x*y - w*z), 1 - 2 * (x*x + z*z), 2 * (y*z + w*x)],
    [2 * (x*z + w*y), 2 * (y*z - w*x), 1 - 2 * (x*x + y*y)],
  ];
}

function triad(ctx, project, pose, length, text) {
  if (!validPoint(pose)) return;
  const axes = quaternionAxes(pose);
  ctx.font = "10px " + mono;
  axes.forEach((axis, i) => {
    const end = project(add(pose, scaleVector(axis, length)));
    const color = ["#ffa1a4", "#a3f0a7", "#9dc3ff"][i];
    line(ctx, project(pose), end, color, 1.5);
    if (text) {
      ctx.fillStyle = color;
      ctx.fillText("XYZ"[i], end[0] + 2, end[1] - 2);
    }
  });
}

function path(ctx, project, key, side, start, stop, opacity, width, offset = 0, stride = 2) {
  const series = data.fields[key];
  if (!series) return;
  ctx.strokeStyle = colors[side];
  ctx.globalAlpha = opacity;
  ctx.lineWidth = width;
  ctx.beginPath();
  let joined = false;
  for (let i = start; i <= stop; i++) {
    const startIndex = i * series.width + offset;
    const pose = series.values.subarray(startIndex, startIndex + 3);
    if (!validPoint(pose)) {
      joined = false;
    } else if (!joined || i % stride === 0 || i === stop) {
      const [x, y] = project(pose);
      if (joined) ctx.lineTo(x, y);
      else ctx.moveTo(x, y);
      joined = true;
    }
  }
  ctx.stroke();
  ctx.globalAlpha = 1;
}

function canvasNote(surface, value, color = colors.muted, x = 11, y = 20) {
  surface.ctx.font = '12px "Noto Sans SC", "Microsoft YaHei", system-ui, sans-serif';
  surface.ctx.fillStyle = color;
  surface.ctx.fillText(value, x, y);
}

function drawHuman() {
  const s = surface("human-view");
  const {ctx} = s;
  const suffix = byId("tracker-source").value;
  const bounds = humanBounds[suffix];
  const project = projector(s, views.human, bounds, .9, 3);
  grid(ctx, project, bounds, 1, .15);
  for (const side of sides) {
    const key = "tracker_" + side + suffix;
    path(ctx, project, key, side, 0, frameCount() - 1, .28, 1.1, 0, 3);
    path(ctx, project, key, side, Math.max(0, frame - 200), frame, .92, 2.1);
    const pose = sample(key);
    const row = side === "left" ? 18 : 35;
    if (!validPoint(pose)) {
      canvasNote(s, (side === "left" ? "左腕" : "右腕") + "  此帧缺失", colors[side], 11, row);
      continue;
    }
    const axes = quaternionAxes(pose);
    const roll = Math.atan2(axes[1][2], axes[2][2]);
    const pitch = Math.asin(Math.max(-1, Math.min(1, -axes[0][2])));
    const yaw = Math.atan2(axes[0][1], axes[0][0]);
    dot(ctx, project(pose), colors[side], 4.5);
    triad(ctx, project, pose, .068, true);
    ctx.font = "10px " + mono;
    ctx.fillStyle = colors[side];
    ctx.fillText((side === "left" ? "L  " : "R  ") +
      Array.from(pose.slice(0, 3), v => format(v, 3)).join("  ") + " m   " +
      [roll, pitch, yaw].map(v => format(v * radToDeg, 0) + "°").join(" "), 11, row);
  }
}

const fingerColors = {Thumb: "#985433", Index: "#1d7665", Middle: "#31739d", Ring: "#65578a", Pinky: "#994c64"};

function drawHand(side) {
  const key = "manus_" + side;
  const s = surface(side + "-hand");
  const {ctx} = s;
  const top = data.topology[key];
  const pose = sample(key);
  const detail = byId("manus-" + side + "-detail");
  if (!pose || !present(key + "_valid") || !top) {
    detail.textContent = "此帧缺失";
    canvasNote(s, "无可用骨架", "#5a727b", 10, s.height / 2 + 8);
    return;
  }
  detail.textContent = top.length + " 节点 · 掌局部";
  const project = projector(s, views[side], handBounds[side], 1.22, 10);
  const points = top.map((_, i) => project(point(pose, i)));
  const byNode = new Map(top.map((row, i) => [row[0], i]));
  ctx.lineCap = "round";
  top.forEach(([id, parent, chain], i) => {
    if (id !== parent && byNode.has(parent)) line(ctx, points[byNode.get(parent)], points[i], fingerColors[chain] || colors.ink, 2);
  });
  top.forEach(([, , chain, joint], i) => dot(ctx, points[i], fingerColors[chain] || colors.ink, joint === "TIP" ? 3 : 2));
}

function drawRobot() {
  const s = surface("robot-view");
  const {ctx} = s;
  const pose = sample("robot_skeleton");
  if (!pose) {
    canvasNote(s, "本片段没有生成参考轨迹");
    return;
  }
  const focus = byId("robot-focus").value;
  const isFull = focus === "all";
  let bounds = robotBounds;
  if (!isFull) {
    const links = corpus.robotLinks.filter(link => link[2] === focus && link[3] === "finger");
    const palm = links[0][0];
    bounds = {center: Array.from(point(pose, palm)), radius: .21};
  }
  const project = projector(s, views.robot, bounds, isFull ? 1.11 : .99);
  if (isFull) grid(ctx, project, bounds, 2, .18);
  const points = Array.from({length: pose.length / 3}, (_, i) => project(point(pose, i)));
  const links = corpus.robotLinks
    .filter(link => isFull || (link[2] === focus && link[3] === "finger"))
    .slice().sort((a, b) => (points[a[0]][2] + points[a[1]][2]) - (points[b[0]][2] + points[b[1]][2]));
  ctx.lineCap = "round";
  ctx.globalAlpha = present("reference_usable") ? 1 : .56;
  for (const [a, b, side, kind] of links) {
    const base = kind === "arm" ? 6.5 : 2.0;
    const p = points[a], q = points[b];
    line(ctx, p, q, kind === "arm" ? "#42616b" : "#304e56", base + 3);
    line(ctx, p, q, colors[side], base);
    dot(ctx, q, colors[side], kind === "arm" ? 4 : 2);
  }
  ctx.globalAlpha = 1;
  if (isFull) {
    const targets = sample("target_tcp_world");
    const native = sample("native_tcp_world") ?? sample("robot_wrist_pose");
    sides.forEach((side, i) => {
      const target = point(targets, i, 7);
      const wrist = point(native, i, 7);
      path(ctx, project, "target_tcp_world", side, Math.max(0, frame - 160), frame, .6, 1.2, i * 7);
      if (validPoint(wrist)) triad(ctx, project, wrist, .053, false);
      if (!validPoint(target) || !validPoint(wrist)) return;
      const a = project(target), b = project(wrist);
      ctx.setLineDash([3, 4]);
      line(ctx, a, b, "#d3e2e6");
      ctx.setLineDash([]);
      ctx.strokeStyle = colors[side];
      ctx.lineWidth = 1.3;
      ctx.strokeRect(a[0] - 3, a[1] - 3, 6, 6);
    });
  }
  if (!present("reference_usable")) canvasNote(s, "此帧未通过参考检查 · 姿态仅供审阅", "#ffd2a7");
  else if (!present("in_library")) canvasNote(s, "此帧未入库 · 姿态仅供审阅", "#ffd2a7");
}

function jointIndex() { return Number(byId("joint-select").value); }

function setJointBounds() {
  const field = data?.fields.q_ref;
  if (!field) return;
  let low = Infinity, high = -Infinity;
  for (let i = jointIndex(); i < field.values.length; i += field.width) {
    const value = field.values[i] * radToDeg;
    if (Number.isFinite(value)) {
      low = Math.min(low, value);
      high = Math.max(high, value);
    }
  }
  if (!Number.isFinite(low)) low = high = 0;
  const pad = Math.max((high - low) * .07, .7);
  jointBounds = [low - pad, high + pad];
}

function drawJoint() {
  const s = surface("joint-chart");
  const {ctx, width, height} = s;
  const field = data.fields.q_ref;
  const joint = jointIndex();
  const value = sample("q_ref")?.[joint] * radToDeg;
  byId("joint-value").textContent = Number.isFinite(value) ? format(value, 2) + "°" : "—";
  if (!field || !jointBounds) return;
  const margin = {left: 43, right: 11, top: 35, bottom: 13};
  const [low, high] = jointBounds;
  const x = i => margin.left + (width - margin.left - margin.right) * i / Math.max(1, frameCount() - 1);
  const y = v => height - margin.bottom - (height - margin.top - margin.bottom) * (v - low) / (high - low);
  ctx.font = "9px " + mono;
  ctx.fillStyle = "#596c75";
  for (const tick of [low, (low + high) / 2, high]) {
    const py = y(tick);
    line(ctx, [margin.left, py], [width - margin.right, py], "#d5e2e3");
    ctx.fillText(format(tick, 0) + "°", 7, py + 3);
  }
  ctx.strokeStyle = joint < 7 || (joint >= 14 && joint < 34) ? "#167468" : "#ab5b23";
  ctx.lineWidth = 1.35;
  ctx.beginPath();
  let joined = false;
  for (let i = 0; i < frameCount(); i++) {
    const angle = field.values[i * field.width + joint] * radToDeg;
    if (Number.isFinite(angle)) {
      if (joined) ctx.lineTo(x(i), y(angle));
      else ctx.moveTo(x(i), y(angle));
      joined = true;
    } else {
      joined = false;
    }
  }
  ctx.stroke();
  line(ctx, [x(frame), margin.top], [x(frame), height - margin.bottom], "#596f75");
  if (Number.isFinite(value)) dot(ctx, [x(frame), y(value)], "#183039");
}

function drawAvailability() {
  const {ctx, width, height} = surface("availability");
  const laneHeight = height / 3;
  const laneWidth = width / frameCount();
  const lanes = [
    ["manus_left_valid", "manus_right_valid"],
    ["tracker_left_valid", "tracker_right_valid", "tracker_safety_valid"],
    ["in_library"],
  ];
  for (const [row, keys] of lanes.entries()) {
    ctx.fillStyle = "#f4dfd0";
    ctx.fillRect(0, row * laneHeight + 3, width, laneHeight - 6);
    ctx.fillStyle = ["#81bbb3", "#4a9a93", "#21685c"][row];
    let start = -1;
    for (let i = 0; i <= frameCount(); i++) {
      const valid = i < frameCount() && keys.every(key => data.fields[key]?.values[i]);
      if (valid && start < 0) start = i;
      if (!valid && start >= 0) {
        ctx.fillRect(start * laneWidth, row * laneHeight + 3, (i - start) * laneWidth, laneHeight - 6);
        start = -1;
      }
    }
  }
  line(ctx, [frame * laneWidth + 1, 0], [frame * laneWidth + 1, height], "#183039", 1.3);
}

function render() {
  if (!data) return;
  drawHuman();
  for (const side of sides) drawHand(side);
  drawRobot();
  drawJoint();
  drawAvailability();
  const inLibrary = present("in_library");
  const usable = present("reference_usable");
  const status = byId("frame-status");
  status.textContent = inLibrary ? "已入库" : usable ? "参考可用 · 未入库" : present("human_valid") ? "参考未通过" : "人体数据缺失";
  status.classList.toggle("warning", !inLibrary);
  const error = sample("arm_position_error_m");
  const angle = sample("arm_orientation_error_rad");
  const penetration = sample("static_penetration_m")?.[0] * 1000;
  byId("reference-metrics").textContent = error ?
    `TCP 误差 L ${format(error[0] * 1000)} / R ${format(error[1] * 1000)} mm · 姿态 L ${format(angle[0] * radToDeg)} / R ${format(angle[1] * radToDeg)}° · 穿透 ${format(penetration, 2)} mm` : "";
  const bits = sample("rejection_bits")?.[0] || 0;
  byId("frame-reasons").textContent = (data.reference.rejection_bit_names || [])
    .filter((_, i) => bits & (1 << i)).map(name).join(" / ");
  const playback = seconds(frame / corpus.rate) + " / " + seconds((frameCount() - 1) / corpus.rate);
  byId("timecode").textContent = motion ? playback + " · 原录制 " + seconds(sample("source_time_s")[0]) : playback;
  byId("seek").value = frame;
}

function setFrame(next) {
  if (!selected || !data) return;
  frame = Math.max(0, Math.min(frameCount() - 1, next));
  render();
}

function step(delta) { setPlaying(false); setFrame(frame + delta); }

function tick(now) {
  if (!playing) return;
  const next = Math.floor(playFrame + (now - playAt) * Number(byId("speed").value) * corpus.rate / 1000);
  if (next !== frame) setFrame(next);
  if (frame >= frameCount() - 1) setPlaying(false);
  else animation = requestAnimationFrame(tick);
}

function setPlaying(next) {
  cancelAnimationFrame(animation);
  playing = next && Boolean(data);
  byId("play").textContent = playing ? "暂停" : "播放";
  byId("play").setAttribute("aria-pressed", String(playing));
  if (!playing) return;
  if (frame >= frameCount() - 1) setFrame(0);
  playAt = performance.now();
  playFrame = frame;
  animation = requestAnimationFrame(tick);
}

function coverage(label, value) {
  const node = element("span");
  node.append(element("strong", value), document.createTextNode(label));
  return node;
}

function populateList() {
  const filter = byId("episode-filter").value;
  const list = byId("episode-list");
  list.replaceChildren();
  for (const e of corpus.episodes.filter(e => !filter || e.task === filter)) {
    const button = element("button", undefined, "episode-button");
    button.type = "button";
    button.dataset.episode = e.id;
    button.setAttribute("aria-current", String(selected?.id === e.id));
    const main = element("span", undefined, "row-main");
    const clock = e.start.slice(11, 19);
    const recorded = corpus.date.includes(" / ") ? e.start.slice(5, 10) + " " + clock.slice(0, 5) : clock;
    main.append(element("strong", name(e.task) + " " + e.trial), element("span", recorded, "technical"));
    const meta = element("span", undefined, "row-meta");
    const longGap = e.maxWristGapMs >= 100;
    const degraded = e.captureQuality === "degraded" || e.integrity !== "read" || longGap;
    const annotation = e.integrity !== "read" ? "读取异常" : longGap ? "腕部缺口" :
      e.captureQuality === "degraded" ? "质量下降" : e.split ? name(e.split) : "";
    meta.append(element("span", seconds(e.duration)), element("span", annotation, degraded ? "row-warning" : ""));
    button.append(main, meta);
    button.addEventListener("click", () => load(e));
    list.append(button);
  }
}

function load(e) {
  setPlaying(false);
  selected = e;
  episodeData = motion = undefined;
  data = undefined;
  frame = 0;
  for (const key of Object.keys(views)) Object.assign(views[key], defaults[key]);
  byId("episode-content").hidden = true;
  byId("episode-meta").textContent = e.start.slice(0, 10) + "  " + e.start.slice(11, 19) + "  UTC" + e.start.slice(19);
  byId("episode-title").textContent = name(e.task) + " " + e.trial;
  byId("episode-id").textContent = e.id;
  byId("episode-status").replaceChildren(
    coverage("同步帧", percentage(e.synced, e.frames)),
    coverage("入库片段", String(e.clips)),
    coverage("录制时长", format(e.duration, 1) + " s"),
  );
  byId("tracker-frame").textContent = e.trackerFrame + " / m / roll · pitch · yaw";
  byId("robot-focus").value = "all";
  byId("tracker-source").value = "";
  byId("tracker-source").options[1].disabled = false;
  const variants = byId("motion-variant");
  const original = element("option", "原始录制");
  original.value = "";
  variants.replaceChildren(original);
  for (const [i, clip] of (e.retimed || []).entries()) {
    const option = element("option", `慢速参考 ${i + 1} · ${clip.factor}× 时长 · ${seconds(clip.sourceFrame / corpus.rate)}`);
    option.value = clip.id;
    variants.append(option);
  }
  variants.hidden = variants.options.length < 2;
  byId("seek").max = e.frames - 1;
  for (const button of byId("episode-list").children) button.setAttribute("aria-current", String(button.dataset.episode === e.id));
  const state = byId("load-state");
  state.hidden = false;
  state.textContent = "读取录制数据…";
  const script = element("script");
  script.src = "episodes/" + e.id + ".js";
  script.onload = () => script.remove();
  script.onerror = () => {
    if (selected.id === e.id) state.textContent = "无法读取此录制。请将 episodes 文件夹与 index.html 放在同一目录。";
    script.remove();
  };
  document.body.append(script);
}

function renderQuality() {
  const table = byId("stream-table");
  table.replaceChildren();
  for (const [key, stream] of Object.entries(data.streams)) {
    const row = element("tr");
    const title = element("td", name(key));
    title.append(element("small", stream.topic));
    row.append(title);
    if (stream.status !== "recorded") {
      const status = element("td", {not_recorded: "未录制", read_error: "读取失败", empty: "零消息"}[stream.status] || stream.status, "missing");
      status.colSpan = 7;
      status.title = stream.error || "";
      row.append(status);
    } else {
      const gap = Math.max(stream.header_clock.interval_ms?.max ?? 0, stream.capture_start_to_first_ms ?? 0,
        stream.capture_last_to_stop_ms ?? 0);
      for (const value of [
        integer.format(stream.count), format(stream.header_clock.observed_hz, 2), format(gap, 2),
        stream.header_clock.gap_estimated_samples ?? "—", stream.source_stamps ? stream.source_duplicates : "—",
        stream.invalid_frames, format(stream.header_to_receive_ms?.p95, 3),
      ]) row.append(element("td", value));
      row.title = `包内间隔峰值 ${format(stream.header_clock.interval_ms?.max, 2)} ms · 尾部缺口 ${format(stream.capture_last_to_stop_ms, 2)} ms · 录制队列丢弃 ${stream.collector_dropped} 条`;
    }
    table.append(row);
  }
  const robotState = data.captureControl?.robot?.hardware_state_name;
  const teleopState = data.captureControl?.controller?.state_name;
  const captureState = [robotState, teleopState].filter(Boolean).join(" / ");
  byId("quality-note").textContent = selected.missingRobot.length ?
    `机器人反馈与指令缺少 ${selected.missingRobot.length} 路${captureState ? "，录制快照为 " + captureState : ""}。参考轨迹由离线重定向生成。` :
    "该录制包含机器人反馈和指令。运动视图显示离线参考轨迹。";
  const count = Object.values(data.streams).reduce((sum, stream) => sum + (stream.collector_dropped || 0), 0);
  const mismatch = Object.values(data.calibration.sides || {}).reduce((sum, side) => sum + (side.calibration_mismatch_count || 0), 0);
  const skew = data.alignment.manus_left_right_nearest_header_skew_ms?.p95;
  const qa = data.calibration.status === "checked" ? `标定重算超差 ${integer.format(mismatch)} 帧` : "缺少已核验的标定快照";
  byId("quality-foot").textContent =
    `录制队列丢弃 ${integer.format(count)} 条 · ${qa} · 双手最近时间戳偏差 P95 ${format(skew, 2)} ms。缺帧按消息间隔估计。接收延迟以 ROS 消息头为起点。`;
  byId("raw-report").href = "episodes/" + selected.id + "-quality.json";
  byId("reference-report").href = "episodes/" + selected.id + "-retarget.json";
}

function unpackFields(fields) {
  return Object.fromEntries(Object.entries(fields).map(([key, value]) => [key, unpack(value)]));
}

function displayMotion(payload, first) {
  data = payload;
  humanBounds = Object.fromEntries(["", "_raw"].map(suffix => [suffix, boundsFor(sides.map(side => "tracker_" + side + suffix), 7, .33)]));
  robotBounds = boundsFor(["robot_skeleton"], 3, .46);
  handBounds = Object.fromEntries(sides.map(side => [side, boundsFor(["manus_" + side], 3, .10)]));
  setJointBounds();
  renderQuality();
  byId("seek").max = frameCount() - 1;
  byId("tracker-source").value = "";
  byId("tracker-source").options[1].disabled = Boolean(motion);
  byId("load-state").hidden = true;
  byId("episode-content").hidden = false;
  setFrame(first);
}

window.wholebodyReceive = payload => {
  if (selected?.id !== payload.id) return;
  episodeData = {...payload, fields: unpackFields(payload.fields)};
  const first = episodeData.fields.in_library?.values.findIndex(v => v) ?? -1;
  const synced = episodeData.fields.human_valid.values.findIndex(v => v);
  displayMotion(episodeData, first >= 0 ? first : Math.max(synced, 0));
};

window.wholebodyReceiveClip = payload => {
  if (selected?.id !== payload.sourceEpisode || byId("motion-variant").value !== payload.id) return;
  motion = selected.retimed.find(clip => clip.id === payload.id);
  displayMotion({...episodeData, fields: unpackFields(payload.fields)}, 0);
};

function selectMotion() {
  setPlaying(false);
  const id = byId("motion-variant").value;
  if (!id) {
    const sourceFrame = motion?.sourceFrame ?? 0;
    motion = undefined;
    displayMotion(episodeData, sourceFrame);
    return;
  }
  const script = element("script");
  script.src = `clips/${id}.js`;
  script.onload = () => script.remove();
  script.onerror = () => {
    if (byId("motion-variant").value === id) {
      byId("motion-variant").value = motion?.id ?? "";
      byId("load-state").textContent = "无法读取此慢速参考。请确认 clips 文件夹与 index.html 在同一目录。";
      byId("load-state").hidden = false;
    }
    script.remove();
  };
  document.body.append(script);
}

function orbit(id, key) {
  const canvas = byId(id);
  const view = views[key];
  let drag;
  canvas.addEventListener("pointerdown", event => {
    drag = [event.clientX, event.clientY];
    canvas.setPointerCapture(event.pointerId);
  });
  canvas.addEventListener("pointermove", event => {
    if (!drag) return;
    view.yaw += (event.clientX - drag[0]) * .008;
    view.pitch = Math.max(-1.42, Math.min(1.42, view.pitch + (event.clientY - drag[1]) * .008));
    drag = [event.clientX, event.clientY];
    render();
  });
  for (const ev of ["pointerup", "pointercancel", "lostpointercapture"]) {
    canvas.addEventListener(ev, () => { drag = undefined; });
  }
  canvas.addEventListener("wheel", event => {
    event.preventDefault();
    view.zoom = Math.max(.45, Math.min(4, view.zoom * Math.exp(-event.deltaY * .001)));
    render();
  }, {passive: false});
  canvas.addEventListener("keydown", event => {
    const move = {ArrowLeft: [-.12, 0], ArrowRight: [.12, 0], ArrowUp: [0, -.12], ArrowDown: [0, .12]}[event.key];
    if (!move) return;
    event.preventDefault();
    view.yaw += move[0];
    view.pitch = Math.max(-1.42, Math.min(1.42, view.pitch + move[1]));
    render();
  });
}

function initialize() {
  if (!corpus?.episodes.length) {
    byId("load-state").hidden = false;
    byId("load-state").textContent = "没有找到录制数据。请检查 dataset.js 是否与此页在同一目录。";
    return;
  }
  const clips = corpus.episodes.reduce((sum, e) => sum + e.clips, 0);
  const frames = corpus.episodes.reduce((sum, e) => sum + e.libraryFrames, 0);
  byId("corpus").append(
    coverage("录制日期", corpus.date), coverage("段录制", String(corpus.episodes.length)),
    coverage("段入库", String(clips)), coverage("秒入库", format(frames / corpus.rate, 1)),
  );
  const filter = byId("episode-filter");
  const all = element("option", "全部动作");
  all.value = "";
  filter.append(all);
  for (const key of new Set(corpus.episodes.map(e => e.task))) {
    const option = element("option", name(key));
    option.value = key;
    filter.append(option);
  }
  filter.addEventListener("change", populateList);
  const joints = byId("joint-select");
  for (const [label, a, b] of [["左臂", 0, 7], ["右臂", 7, 14], ["左手", 14, 34], ["右手", 34, 54]]) {
    const group = element("optgroup");
    group.label = label;
    for (let i = a; i < b; i++) {
      const option = element("option", corpus.jointNames[i]);
      option.value = i;
      group.append(option);
    }
    joints.append(group);
  }
  joints.addEventListener("change", () => { setJointBounds(); render(); });
  byId("play").addEventListener("click", () => setPlaying(!playing));
  byId("motion-variant").addEventListener("change", selectMotion);
  byId("prev-frame").addEventListener("click", () => step(-1));
  byId("next-frame").addEventListener("click", () => step(1));
  byId("seek").addEventListener("input", event => { setPlaying(false); setFrame(Number(event.target.value)); });
  byId("speed").addEventListener("change", () => { if (playing) setPlaying(true); });
  byId("availability").addEventListener("click", event => {
    const rect = event.target.getBoundingClientRect();
    setPlaying(false);
    setFrame(Math.floor(frameCount() * (event.clientX - rect.left) / rect.width));
  });
  for (const id of ["tracker-source", "robot-focus"]) byId(id).addEventListener("change", render);
  for (const button of document.querySelectorAll("[data-reset]")) {
    button.addEventListener("click", () => {
      const key = button.dataset.reset;
      Object.assign(views[key], defaults[key]);
      if (key === "human") for (const side of sides) Object.assign(views[side], defaults[side]);
      render();
    });
  }
  orbit("human-view", "human");
  orbit("robot-view", "robot");
  for (const side of sides) orbit(side + "-hand", side);
  window.addEventListener("resize", () => {
    cancelAnimationFrame(resizeId);
    resizeId = requestAnimationFrame(render);
  });
  document.addEventListener("keydown", event => {
    if (!data || event.defaultPrevented || ["INPUT", "SELECT", "BUTTON", "A", "CANVAS"].includes(event.target.tagName)) return;
    if (event.code === "Space") {
      event.preventDefault();
      setPlaying(!playing);
    } else if (event.key === "ArrowLeft" || event.key === "ArrowRight") {
      event.preventDefault();
      step(event.key === "ArrowLeft" ? -1 : 1);
    }
  });
  populateList();
  load(corpus.episodes.find(e => e.task === "arm_hand" && e.clips) || corpus.episodes[0]);
}

initialize();
