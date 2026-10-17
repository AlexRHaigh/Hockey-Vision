"use strict";
// Hockey-Vision web app: one page, routed by the URL hash (#/players, #/player/<id>, ...), talking to
// the JSON API in api.py. Every piece of data goes into the page as text (h() below), never as HTML.

const app = document.getElementById("app");
const tooltip = document.getElementById("tooltip");
const SVG_NS = "http://www.w3.org/2000/svg";
const GROUPS = { F: "Forwards", D: "Defense", G: "Goalies" };
const POS_SHORT = { "Center": "C", "Left wing": "LW", "Right wing": "RW", "Defense": "D", "Goalie": "G" };

// --- DOM helpers ---

function h(tag, attrs, ...children) {
  const el = document.createElement(tag);
  setAttrs(el, attrs);
  append(el, children);
  return el;
}

function s(tag, attrs, ...children) {
  const el = document.createElementNS(SVG_NS, tag);
  setAttrs(el, attrs);
  append(el, children);
  return el;
}

function setAttrs(el, attrs) {
  for (const [k, v] of Object.entries(attrs || {})) {
    if (v === null || v === undefined || v === false) continue;
    if (k.startsWith("on")) el.addEventListener(k.slice(2), v);
    else if (k === "text") el.textContent = v;
    else el.setAttribute(k, v === true ? "" : v);
  }
}

function append(el, children) {
  for (const c of children.flat(Infinity)) {
    if (c === null || c === undefined || c === false) continue;
    el.append(c instanceof Node ? c : document.createTextNode(String(c)));
  }
}

function icon(path, cls) {
  return s("svg", { viewBox: "0 0 24 24", class: cls, "aria-hidden": "true" }, s("path", { d: path }));
}

const ICONS = {
  keep: "M9 16.2 4.8 12l-1.4 1.4L9 19 21 7l-1.4-1.4z",
  watch: "M1 21h22L12 2zm12-3h-2v-2h2zm0-4h-2v-4h2z",
  trade: "M12 2a10 10 0 1 0 0 20 10 10 0 0 0 0-20zm1 15h-2v-2h2zm0-4h-2V7h2z",
  hold: "M6 19h4V5H6zm8-14v14h4V5z",
  star: "M12 17.3 18.2 21l-1.6-7 5.4-4.7-7.2-.6L12 2 9.2 8.7 2 9.3l5.4 4.7L5.8 21z",
};

function clear(el) { el.replaceChildren(); return el; }

function toast(msg) {
  const t = document.getElementById("toast");
  t.textContent = msg;
  t.classList.add("show");
  clearTimeout(toast.timer);
  toast.timer = setTimeout(() => t.classList.remove("show"), 2200);
}

function showTip(evt, ...lines) {
  tooltip.replaceChildren();
  append(tooltip, lines.map((l, i) => i === 0 ? h("strong", {}, l) : h("div", { class: "muted" }, l)));
  tooltip.hidden = false;
  const r = evt.target.getBoundingClientRect();
  const x = evt.clientX || r.left + r.width / 2;
  const y = evt.clientY || r.top;
  const tw = tooltip.offsetWidth, th = tooltip.offsetHeight;
  tooltip.style.left = Math.min(window.innerWidth - tw - 8, Math.max(8, x - tw / 2)) + "px";
  tooltip.style.top = (y - th - 12 < 8 ? y + 16 : y - th - 12) + "px";
}

function hideTip() { tooltip.hidden = true; }

// --- API ---

async function api(method, path, body) {
  const res = await fetch(path, {
    method,
    headers: body ? { "Content-Type": "application/json" } : {},
    body: body ? JSON.stringify(body) : undefined,
  });
  const data = await res.json().catch(() => ({}));
  if (!res.ok) throw new Error(data.error || `${res.status} ${res.statusText}`);
  return data;
}

const get = (path) => api("GET", path);

// --- Theme ---

function storedTheme() {
  try { return localStorage.getItem("hv-theme"); } catch { return null; }
}

function isDark() {
  const t = document.documentElement.dataset.theme;
  return t ? t === "dark" : matchMedia("(prefers-color-scheme: dark)").matches;
}

function initTheme() {
  const t = storedTheme();
  if (t) document.documentElement.dataset.theme = t;
  document.getElementById("theme-toggle").addEventListener("click", () => {
    const next = isDark() ? "light" : "dark";
    document.documentElement.dataset.theme = next;
    try { localStorage.setItem("hv-theme", next); } catch { /* private mode: theme just isn't remembered */ }
    route();
  });
}

// --- Shared pieces ---

function fmt(v, digits = 0) {
  return v === null || v === undefined ? "–" : Number(v).toFixed(digits);
}

function logo(team, cls = "logo") {
  if (!team) return null;
  const img = h("img", { class: cls, alt: "", loading: "lazy",
    src: `https://assets.nhle.com/logos/nhl/svg/${encodeURIComponent(team)}_${isDark() ? "dark" : "light"}.svg` });
  // Logos come from the NHL's CDN; offline, show the abbreviation instead.
  img.addEventListener("error", () => img.replaceWith(h("span", { class: `${cls} logo-text`, "aria-hidden": "true" }, team)));
  return img;
}

function meter(value) {
  if (value === null || value === undefined) return h("span", { class: "none" }, "–");
  const bar = h("span", {});
  bar.style.width = Math.max(2, Math.min(100, value)) + "%";
  return h("span", { class: "meter", title: `${fmt(value, 1)} (50 = a typical game)` },
    h("span", { class: "bar" }, bar), h("b", {}, fmt(value)));
}

function pill(rec) {
  return h("span", { class: `pill ${rec.action}`, title: rec.reason }, icon(ICONS[rec.action]), rec.label);
}

function trend(t) {
  if (t === null || t === undefined) return h("span", { class: "none" }, "–");
  const cls = t >= 3 ? "delta-up" : t <= -3 ? "delta-down" : "muted";
  return h("span", { class: `num ${cls}` }, `${t > 0 ? "▲ +" : t < 0 ? "▼ " : ""}${fmt(t)}`);
}

function starButton(p, onChange) {
  const btn = h("button", { class: `star${p.favorite ? " on" : ""}`, type: "button",
    "aria-pressed": String(!!p.favorite), "aria-label": `${p.favorite ? "Remove" : "Add"} ${p.name} ${p.favorite ? "from" : "to"} favourites`,
    title: p.favorite ? "Remove from favourites" : "Add to favourites" }, icon(ICONS.star));
  btn.addEventListener("click", async (e) => {
    e.stopPropagation();
    try {
      await api(p.favorite ? "DELETE" : "POST", `/api/players/${p.player_id}/favorite`);
      p.favorite = !p.favorite;
      btn.classList.toggle("on", p.favorite);
      btn.setAttribute("aria-pressed", String(p.favorite));
      btn.title = p.favorite ? "Remove from favourites" : "Add to favourites";
      toast(p.favorite ? `${p.name} added to favourites` : `${p.name} removed from favourites`);
      if (onChange) onChange(p);
    } catch (err) { toast(err.message); }
  });
  return btn;
}

function playerTable(players, opts = {}) {
  if (!players.length) return h("div", { class: "empty" }, opts.empty || "No players.");
  const full = !opts.compact;
  const head = full ? ["", "#", "Player", "Pos", "Form", "Last game", "Trend", "GP", "Call"] : ["", "Player", "Pos", "Form", "Call"];
  if (opts.remove) head.push("");
  const rows = players.map((p) => {
    const tr = h("tr", { class: "clickable", tabindex: "0", onclick: () => go(`#/player/${p.player_id}`),
      onkeydown: (e) => { if (e.key === "Enter") go(`#/player/${p.player_id}`); } },
      h("td", {}, starButton(p, opts.onFavorite)),
      full ? h("td", { class: "num muted" }, p.number ?? "") : null,
      h("td", { class: "name" }, p.name, h("small", {}, p.team_name || p.team || "")),
      h("td", { title: p.position }, POS_SHORT[p.position] || p.position || ""),
      h("td", {}, meter(p.form.form)),
      full ? [h("td", { class: "r num" }, fmt(p.form.last)), h("td", { class: "r" }, trend(p.form.trend)),
        h("td", { class: "r num" }, p.form.games)] : null,
      h("td", {}, pill(p.recommendation)));
    if (opts.remove) {
      tr.append(h("td", {}, h("button", { class: "btn small danger", type: "button",
        onclick: (e) => { e.stopPropagation(); opts.remove(p); } }, "Remove")));
    }
    return tr;
  });
  return h("div", { class: "table-wrap" }, h("table", {},
    h("thead", {}, h("tr", {}, head.map((t, i) => h("th", { class: full && [5, 6, 7].includes(i) ? "r" : null }, t)))),
    h("tbody", {}, rows)));
}

function pageHead(title, sub, ...actions) {
  return h("div", { class: "page-head" },
    h("div", {}, h("h1", {}, title), sub ? h("p", {}, sub) : null),
    actions.length ? h("div", { class: "filters" }, actions) : null);
}

function tile(label, value, sub) {
  return h("div", { class: "tile" }, h("div", { class: "label" }, label), h("div", { class: "value" }, value),
    sub ? h("div", { class: "sub" }, sub) : null);
}

function noGamesHelp() {
  return h("div", { class: "card empty" },
    h("h2", {}, "No games analysed yet"),
    h("p", {}, "Run a game through the pipeline, then import its export folder:"),
    h("p", {}, h("code", {}, "python -m webapp.ingest outputs/<clip>/export --roster rosters/<game>.csv")),
    h("p", {}, "Or try the app with made-up demo games: ", h("code", {}, "python -m webapp.demo")));
}

// --- Charts ---

function scoreChart(games, selectedId, onSelect) {
  const data = games.slice(0, 30).reverse(); // oldest first
  // Drawn at the page's width so the text stays at its real size.
  const W = Math.max(320, Math.min(1180, app.clientWidth - 72)), H = 220, m = { t: 12, r: 8, b: 28, l: 30 };
  const iw = W - m.l - m.r, ih = H - m.t - m.b;
  const step = iw / data.length;
  const bw = Math.max(4, Math.min(28, step - 2));
  const y = (v) => m.t + ih - (v / 100) * ih;
  const svgEl = s("svg", { class: "chart", viewBox: `0 0 ${W} ${H}`, role: "img",
    "aria-label": `Game scores for the last ${data.length} games; the game log table below lists every value.` });
  const grid = s("g", { class: "grid" });
  for (const v of [0, 25, 50, 75, 100]) {
    grid.append(s("line", { x1: m.l, x2: W - m.r, y1: y(v), y2: y(v) }));
    svgEl.append(s("text", { x: m.l - 6, y: y(v) + 4, "text-anchor": "end" }, String(v)));
  }
  svgEl.prepend(grid);
  data.forEach((g, i) => {
    const cx = m.l + step * i + step / 2;
    const top = y(Math.max(g.score, 1));
    const hit = s("rect", { class: "hit", x: m.l + step * i, y: m.t, width: step, height: ih, tabindex: "0",
      "aria-label": `${g.date} ${g.label}: score ${fmt(g.score)}` });
    // Bar: rounded top, square at the baseline.
    const r = Math.min(4, bw / 2);
    const x0 = cx - bw / 2, x1 = cx + bw / 2, base = y(0);
    const bar = s("path", { class: `bar${g.game_id === selectedId ? " selected" : ""}`,
      d: `M${x0},${base} V${top + r} Q${x0},${top} ${x0 + r},${top} H${x1 - r} Q${x1},${top} ${x1},${top + r} V${base} Z` });
    const tip = (e) => showTip(e, `${fmt(g.score)} score`, `${g.label}`, `${g.date} · ${fmt(g.minutes, 1)} min seen (${g.confidence} confidence)`);
    hit.addEventListener("pointermove", tip);
    hit.addEventListener("focus", tip);
    hit.addEventListener("pointerleave", hideTip);
    hit.addEventListener("blur", hideTip);
    hit.addEventListener("click", () => onSelect(g));
    hit.addEventListener("keydown", (e) => { if (e.key === "Enter") onSelect(g); });
    svgEl.append(hit, bar);
    const every = Math.ceil(data.length / 8);
    if (i % every === 0 || i === data.length - 1) {
      svgEl.append(s("text", { x: cx, y: H - 8, "text-anchor": "middle" }, g.date.slice(5)));
    }
  });
  svgEl.append(s("line", { class: "ref", x1: m.l, x2: W - m.r, y1: y(50), y2: y(50) }));
  return h("div", {}, svgEl,
    h("div", { class: "legend" }, h("span", {}, h("i", {}), "Game score (select a bar for its breakdown)"),
      h("span", {}, h("i", { class: "dash" }), "50 = a typical game")));
}

function breakdown(g) {
  if (!g.components.length) return h("p", { class: "muted" }, "No counted plays in this game; the score comes from time on the ice alone.");
  const max = Math.max(...g.components.map((c) => Math.abs(c.points)), 1);
  return h("div", {},
    h("div", { class: "breakdown" }, g.components.map((c) => {
      const span = h("span", { class: c.points >= 0 ? "pos" : "neg" });
      span.style.width = (Math.abs(c.points) / max) * 50 + "%";
      return [h("div", {}, `${c.label} `, h("span", { class: "muted num" }, `×${fmt(c.count, c.count % 1 ? 1 : 0)}`)),
        h("div", { class: "track", title: `${fmt(c.count, 1)} × ${c.weight} = ${fmt(c.points, 1)} pts` }, span),
        h("div", { class: "num r" }, `${c.points > 0 ? "+" : ""}${fmt(c.points, 1)}`)];
    })),
    h("div", { class: "legend" }, h("span", {}, h("i", {}), "Adds to the score"), h("span", {}, h("i", { class: "neg" }), "Takes away")),
    h("p", { class: "help" }, `${fmt(g.points, 1)} impact points in ${fmt(g.minutes, 1)} minutes seen → ${fmt(g.rate, 2)} per minute after blending with a typical game → score ${fmt(g.score)}.`));
}

// --- Pages ---

async function pageHome() {
  const [summary, top, lists] = await Promise.all([get("/api/summary"), get("/api/players?sort=form&rated=1&limit=10"), get("/api/playlists")]);
  const pools = lists.filter((l) => l.is_pool);
  const assessments = await Promise.all(pools.map((p) => get(`/api/playlists/${p.id}/assessment`)));
  const body = [
    pageHead("Overview", "Every NHL player, scored from the games Hockey-Vision has analysed."),
    h("div", { class: "tiles" },
      tile("Teams", summary.teams), tile("Players", summary.players, "active NHL rosters"),
      tile("Games analysed", summary.games, summary.demo_games ? `${summary.demo_games} of them demo games` : null),
      tile("Players rated", summary.rated_players, "with at least one game")),
  ];
  if (!summary.games) body.push(noGamesHelp());
  const poolCard = h("div", { class: "card" }, h("div", { class: "card-head" }, h("h2", {}, "Your pools"),
    h("a", { href: "#/playlists" }, "Manage playlists")));
  if (!pools.length) {
    poolCard.append(h("p", { class: "muted" }, "Make a playlist and mark it as a pool to get trade calls and a strength assessment."));
  } else {
    pools.forEach((p, i) => {
      const a = assessments[i];
      poolCard.append(h("a", { class: "upgrade", href: `#/playlist/${p.id}` },
        h("div", {}, h("b", {}, p.name), h("div", { class: "small muted" }, `${p.players} players`)),
        h("span", { class: "arrow" }, ""),
        h("div", {}, h("div", {}, a.verdict_label),
          h("div", { class: "small muted" }, a.strength !== null ? `Strength ${fmt(a.strength)} (${a.grade}) · urgency ${a.urgency}/100` : "Not rated yet"))));
    });
  }
  const topCard = h("div", { class: "card" }, h("div", { class: "card-head" }, h("h2", {}, "In form"),
    h("a", { href: "#/players?sort=form&rated=1" }, "All players by form")), playerTable(top, { compact: true, empty: "No rated players yet." }));
  body.push(h("div", { class: "grid-2" }, topCard, h("div", { class: "stack" }, poolCard, scoreHelp(summary))));
  return body;
}

function scoreHelp(summary) {
  const t = summary.thresholds;
  return h("div", { class: "card help" }, h("h2", {}, "How scores work"),
    h("dl", {},
      h("dt", {}, "Game score (0–100)"),
      h("dd", {}, "Each play the video picks up is worth impact points (takeaway +2, good pass +1, turnover −2, …). Points per minute the player was seen are compared with every other player-game: 50 is a typical game, 73 one spread better, 88 two."),
      h("dt", {}, "Form"),
      h("dd", {}, "The last 5 game scores, newer games and games the player was seen longer counting more. Trend: last 3 games against the ones before."),
      h("dt", {}, "Trade calls"),
      h("dd", {}, `Trade below ${t.trade_below} form, Watch below ${t.watch_below} or on a 10-point slide, Keep otherwise, Hold until ${t.min_games} games are in.`)));
}

let playersState = { team: "", q: "", group: "", sort: "name", rated: false };

async function pagePlayers(params) {
  Object.assign(playersState, Object.fromEntries(params));
  if (params.has("rated")) playersState.rated = params.get("rated") === "1";
  const teams = await get("/api/teams");
  const list = h("div", {});
  const teamBtns = [{ team: "", team_name: "All teams", players: teams.reduce((n, t) => n + t.players, 0) }, ...teams].map((t) =>
    h("button", { type: "button", class: t.team === playersState.team ? "active" : null, "data-team": t.team,
      onclick: () => { playersState.team = t.team; syncTeams(); load(); } },
      t.team ? logo(t.team) : null, h("span", { class: "team-name" }, t.team_name), h("span", { class: "count" }, t.players)));
  const teamList = h("nav", { class: "card team-list", "aria-label": "Teams" }, teamBtns);
  const syncTeams = () => teamBtns.forEach((b) => b.classList.toggle("active", b.dataset.team === playersState.team));
  const title = h("h1", {});

  const search = h("input", { type: "search", class: "grow", placeholder: "Search players…", value: playersState.q, "aria-label": "Search players" });
  let timer;
  search.addEventListener("input", () => { clearTimeout(timer); timer = setTimeout(() => { playersState.q = search.value; load(); }, 200); });
  const group = h("select", { "aria-label": "Position", onchange: (e) => { playersState.group = e.target.value; load(); } },
    h("option", { value: "" }, "All positions"), Object.entries(GROUPS).map(([k, v]) => h("option", { value: k }, v)));
  group.value = playersState.group;
  const sort = h("select", { "aria-label": "Sort", onchange: (e) => { playersState.sort = e.target.value; load(); } },
    h("option", { value: "name" }, "Sort: name"), h("option", { value: "form" }, "Sort: form"), h("option", { value: "number" }, "Sort: number"));
  sort.value = playersState.sort;
  const rated = h("input", { type: "checkbox", onchange: (e) => { playersState.rated = e.target.checked; load(); } });
  rated.checked = playersState.rated;

  async function load() {
    const q = new URLSearchParams();
    for (const k of ["team", "q", "group", "sort"]) if (playersState[k]) q.set(k, playersState[k]);
    if (playersState.rated) q.set("rated", "1");
    list.classList.add("loading");
    const players = await get(`/api/players?${q}`);
    const team = teams.find((t) => t.team === playersState.team);
    title.replaceChildren(team ? logo(team.team) : "", team ? ` ${team.team_name}` : "All players");
    title.style.display = "flex"; title.style.gap = "10px"; title.style.alignItems = "center";
    list.replaceChildren(h("p", { class: "muted small" }, `${players.length} players`),
      playerTable(players, { empty: "No players match these filters." }));
    list.classList.remove("loading");
  }
  await load();
  return h("div", { class: "layout-side" }, teamList,
    h("div", {}, h("div", { class: "page-head" }, title),
      h("div", { class: "filters" }, search, group, sort, h("label", { class: "check" }, rated, "Rated only")),
      h("div", { class: "card" }, list)));
}

async function pagePlayer(id) {
  const [p, lists] = await Promise.all([get(`/api/players/${id}`), get("/api/playlists")]);
  let selected = p.games[0];
  const chartBox = h("div", {});
  const breakdownBox = h("div", {});
  const breakdownTitle = h("h2", {});
  const select = (g) => {
    selected = g;
    chartBox.replaceChildren(scoreChart(p.games, g.game_id, select));
    breakdownTitle.replaceChildren(`Score breakdown: ${g.label}, ${g.date}`);
    breakdownBox.replaceChildren(h("div", { class: "tiles" }, tile("Score", fmt(g.score), `${g.confidence} confidence`),
      tile("Minutes seen", fmt(g.minutes, 1)), tile("Impact points", fmt(g.points, 1))), breakdown(g));
  };

  const memberOf = new Set(p.playlists.map((l) => l.id));
  const addSelect = h("select", { "aria-label": "Add to playlist" },
    h("option", { value: "" }, lists.length ? "Add to playlist…" : "No playlists yet"),
    lists.filter((l) => !memberOf.has(l.id)).map((l) => h("option", { value: l.id }, `${l.name}${l.is_pool ? " (pool)" : ""}`)));
  addSelect.addEventListener("change", async () => {
    if (!addSelect.value) return;
    const list = lists.find((l) => String(l.id) === addSelect.value);
    await api("POST", `/api/playlists/${list.id}/players`, { player_id: p.player_id });
    toast(`Added to ${list.name}`);
    route();
  });

  const f = p.form;
  const body = [
    h("div", { class: "card" }, h("div", { class: "player-head" },
      logo(p.team, "logo logo-lg"),
      h("div", {}, h("h1", {}, p.name),
        h("div", { class: "muted" }, `${p.position || ""} · `, h("a", { href: `#/players?team=${p.team}` }, p.team_name || p.team || "No team"))),
      h("span", { class: "jersey" }, p.number !== null ? `#${p.number}` : ""),
      h("div", { class: "actions" }, starButton(p), addSelect))),
  ];
  if (p.playlists.length) {
    body[0].append(h("p", { class: "small muted" }, "In: ",
      p.playlists.map((l, i) => [i ? ", " : "", h("a", { href: `#/playlist/${l.id}` }, l.name)])));
  }
  body.push(h("div", { class: "tiles", style: "margin-top:16px" },
    tile("Form", fmt(f.form), "last 5 games, weighted"), tile("Season average", fmt(f.season)),
    tile("Last game", fmt(f.last)), tile("Trend", f.trend === null ? "–" : trend(f.trend), "last 3 vs before"),
    tile("Games", f.games)));
  body.push(h("div", { class: "card rec-card" }, pill(p.recommendation),
    h("div", {}, h("b", {}, "Pool call"), h("p", {}, p.recommendation.reason))));
  if (!p.games.length) {
    body.push(h("div", { class: "card empty" }, "No analysed games for this player yet."));
    return body;
  }
  body.push(h("div", { class: "card" }, h("div", { class: "card-head" }, h("h2", {}, "Game scores")), chartBox));
  body.push(h("div", { class: "card" }, h("div", { class: "card-head" }, breakdownTitle), breakdownBox));
  const goalie = p.games[0].role === "goalie";
  const cols = goalie
    ? [["saves", "Saves"], ["goals_against", "GA"], ["goalie_recoveries", "Recov."]]
    : [["shots", "Shots"], ["shots_on_net", "On net"], ["passes_completed", "Passes"], ["good_passes", "Good"],
       ["takeaways", "TK"], ["interceptions", "INT"], ["defensive_plays", "Def."], ["turnovers", "TO"]];
  const logHead = h("tr", {}, h("th", {}, "Date"), h("th", {}, "Game"), h("th", { class: "r" }, "Score"),
    h("th", { class: "r" }, "Min seen"), cols.map(([, l]) => h("th", { class: "r" }, l)));
  const logRows = p.games.map((g) => h("tr", { class: "clickable",
    onclick: () => { select(g); breakdownTitle.scrollIntoView({ behavior: "smooth", block: "center" }); } },
    h("td", { class: "num" }, g.date),
    h("td", {}, h("a", { href: `#/game/${g.game_id}`, onclick: (e) => e.stopPropagation() }, g.label),
      g.is_demo ? [" ", h("span", { class: "badge" }, "Demo")] : null),
    h("td", { class: "r num" }, h("b", {}, fmt(g.score))),
    h("td", { class: "r num" }, fmt(g.minutes, 1)),
    cols.map(([k]) => h("td", { class: "r num" }, fmt(g.stats[k] || 0)))));
  body.push(h("div", { class: "card" }, h("div", { class: "card-head" }, h("h2", {}, "Game log")),
    h("div", { class: "table-wrap" }, h("table", {}, h("thead", {}, logHead), h("tbody", {}, logRows)))));
  select(selected);
  return body;
}

async function pageGames() {
  const games = await get("/api/games");
  if (!games.length) return [pageHead("Games"), noGamesHelp()];
  return [pageHead("Games", `${games.length} games analysed. Open one to see every player's score.`),
    h("div", { class: "card table-wrap" }, h("table", {},
      h("thead", {}, h("tr", {}, h("th", {}, "Date"), h("th", {}, "Game"), h("th", { class: "r" }, "Players"), h("th", {}, "Top performer"))),
      h("tbody", {}, games.map((g) => h("tr", { class: "clickable", onclick: () => go(`#/game/${g.id}`) },
        h("td", { class: "num" }, g.date),
        h("td", { class: "name" }, logo(g.away), " ", logo(g.home), " ", g.label, g.is_demo ? [" ", h("span", { class: "badge" }, "Demo")] : null),
        h("td", { class: "r num" }, g.players),
        h("td", {}, g.top ? [g.top.name, " ", h("span", { class: "muted num" }, fmt(g.top.score))] : "–"))))))];
}

async function pageGame(id) {
  const g = await get(`/api/games/${id}`);
  const teams = [g.away, g.home].filter(Boolean);
  const byTeam = teams.length ? teams.map((t) => [t, g.players.filter((p) => p.team === t)]) : [["", g.players]];
  const del = h("button", { class: "btn danger", type: "button", onclick: async () => {
    if (!confirm(`Remove ${g.label} (${g.date}) and its scores from the app?`)) return;
    await api("DELETE", `/api/games/${g.id}`);
    toast("Game removed");
    go("#/games");
  } }, "Remove game");
  return [pageHead(g.label, `${g.date}${g.nhl_game_id ? ` · NHL game ${g.nhl_game_id}` : ""}${g.is_demo ? " · demo game (made up)" : ""}`, del),
    h("div", { class: "grid-2" }, byTeam.map(([team, players]) => h("div", { class: "card" },
      h("div", { class: "card-head" }, h("h2", { style: "display:flex;gap:8px;align-items:center" }, logo(team), team || "Players"),
        h("span", { class: "muted small" }, `${players.length} players`)),
      h("div", { class: "table-wrap" }, h("table", {},
        h("thead", {}, h("tr", {}, h("th", {}, "#"), h("th", {}, "Player"), h("th", {}, "Pos"), h("th", {}, "Score"), h("th", { class: "r" }, "Min seen"))),
        h("tbody", {}, players.map((p) => h("tr", { class: "clickable", onclick: () => go(`#/player/${p.player_id}`) },
          h("td", { class: "num muted" }, p.number ?? ""), h("td", { class: "name" }, p.name),
          h("td", {}, POS_SHORT[p.position] || ""), h("td", {}, meter(p.score)),
          h("td", { class: "r num" }, fmt(p.minutes, 1)))))))))) ];
}

async function pageFavorites() {
  const players = await get("/api/players?favorites=1&sort=form");
  const box = h("div", { class: "card" });
  const render = (list) => box.replaceChildren(playerTable(list, {
    empty: "No favourites yet. Star a player anywhere in the app to keep them here.",
    onFavorite: (p) => { if (!p.favorite) render(list.filter((x) => x !== p)); },
  }));
  render(players);
  return [pageHead("Favourites", "Players you've starred, best form first."), box];
}

function playlistForm(initial, onSubmit, submitLabel) {
  const name = h("input", { type: "text", required: true, maxlength: "80", value: initial.name || "", placeholder: "e.g. My pool, Trade targets" });
  const desc = h("input", { type: "text", maxlength: "200", value: initial.description || "", placeholder: "Optional" });
  const pool = h("input", { type: "checkbox" });
  pool.checked = !!initial.is_pool;
  const slot = (k, v) => h("input", { type: "number", min: "0", max: "50", value: String(initial[k] ?? v) });
  const slots = { slots_f: slot("slots_f", 12), slots_d: slot("slots_d", 6), slots_g: slot("slots_g", 2) };
  const slotRow = h("div", { class: "form-row" },
    h("label", {}, "Forwards", slots.slots_f), h("label", {}, "Defense", slots.slots_d), h("label", {}, "Goalies", slots.slots_g));
  const syncSlots = () => { slotRow.style.display = pool.checked ? "" : "none"; };
  pool.addEventListener("change", syncSlots);
  syncSlots();
  const form = h("form", { class: "stack" },
    h("div", { class: "form-row" }, h("label", { class: "grow" }, "Name", name), h("label", { class: "grow" }, "Description", desc)),
    h("label", { class: "check" }, pool, "This is a hockey pool (get trade calls and a strength assessment)"),
    slotRow,
    h("div", {}, h("button", { class: "btn primary", type: "submit" }, submitLabel)));
  form.addEventListener("submit", async (e) => {
    e.preventDefault();
    try {
      await onSubmit({ name: name.value, description: desc.value, is_pool: pool.checked,
        slots_f: +slots.slots_f.value, slots_d: +slots.slots_d.value, slots_g: +slots.slots_g.value });
    } catch (err) { toast(err.message); }
  });
  return form;
}

async function pagePlaylists() {
  const lists = await get("/api/playlists");
  const create = playlistForm({ is_pool: !lists.some((l) => l.is_pool) }, async (body) => {
    const pl = await api("POST", "/api/playlists", body);
    toast(`Created ${pl.name}`);
    go(`#/playlist/${pl.id}`);
  }, "Create playlist");
  return [pageHead("Playlists & pools", "Group players however you like. Mark a playlist as a pool to see who to keep, who to trade and whether to trade at all."),
    h("div", { class: "card" }, h("h2", { style: "margin-bottom:12px" }, "New playlist"), create),
    h("div", { class: "card" }, h("h2", { style: "margin-bottom:12px" }, "Your playlists"),
      lists.length ? h("div", { class: "playlist-grid" }, lists.map((l) => h("a", { class: "card playlist-card", href: `#/playlist/${l.id}` },
        h("h3", {}, l.name, " ", l.is_pool ? h("span", { class: "badge pool" }, "Pool") : null),
        h("p", {}, `${l.players} player${l.players === 1 ? "" : "s"}${l.description ? ` · ${l.description}` : ""}`))))
        : h("p", { class: "muted" }, "None yet."))];
}

function playerPicker(onPick) {
  const input = h("input", { type: "search", placeholder: "Add a player: type a name…", "aria-label": "Find a player to add", style: "width:100%" });
  const results = h("div", { class: "picker-results", hidden: true });
  let timer;
  input.addEventListener("input", () => {
    clearTimeout(timer);
    const q = input.value.trim();
    if (q.length < 2) { results.hidden = true; return; }
    timer = setTimeout(async () => {
      const players = await get(`/api/players?q=${encodeURIComponent(q)}&limit=12`);
      results.replaceChildren(...(players.length ? players.map((p) => h("button", { type: "button", onclick: () => { results.hidden = true; input.value = ""; onPick(p); } },
        logo(p.team), h("span", {}, h("b", {}, p.name), " ", h("span", { class: "muted small" }, `${POS_SHORT[p.position] || ""} · ${p.team} #${p.number}`)),
        h("span", { style: "margin-left:auto" }, meter(p.form.form)))) : [h("div", { class: "muted", style: "padding:10px" }, "No players found")]));
      results.hidden = false;
    }, 180);
  });
  input.addEventListener("keydown", (e) => { if (e.key === "Escape") results.hidden = true; });
  return h("div", { class: "picker" }, input, results);
}

function assessmentView(a) {
  const iconPath = { hold: ICONS.keep, targeted: ICONS.watch, active: ICONS.trade, empty: ICONS.hold }[a.verdict];
  const gaugeMark = h("i", {});
  gaugeMark.style.left = a.urgency + "%";
  const out = [
    h("div", { class: `verdict ${a.verdict}` }, icon(iconPath, "icon"),
      h("div", {}, h("h2", {}, a.verdict_label), h("ul", {}, a.reasons.map((r) => h("li", {}, r))))),
  ];
  if (a.verdict === "empty") return out;
  out.push(h("div", { class: "tiles", style: "margin-top:16px" },
    tile("Pool strength", a.strength === null ? "–" : [fmt(a.strength), h("span", { class: "muted", style: "font-size:1rem" }, ` ${a.grade}`)], "mean form; 50 = typical"),
    tile("Trade urgency", `${a.urgency}`, "0 hold · 100 rebuild"),
    tile("Above league median", a.rated ? `${a.above_median}/${a.rated}` : "–", a.league_median !== null ? `median form ${fmt(a.league_median)}` : null),
    tile("Calls", `${a.counts.keep || 0} · ${a.counts.watch || 0} · ${a.counts.trade || 0}`, "keep · watch · trade")));
  out.push(h("div", { class: "grid-2" },
    h("div", { class: "card" }, h("h2", {}, "Trade urgency"),
      h("div", { class: "gauge", role: "img", "aria-label": `Trade urgency ${a.urgency} of 100` }, gaugeMark),
      h("div", { class: "gauge-scale" }, h("span", {}, "Hold"), h("span", {}, "Targeted trade"), h("span", {}, "Be active")),
      h("h2", { style: "margin-top:18px" }, "Depth"),
      a.depth.map((d) => {
        const boxes = [];
        for (let i = 0; i < Math.max(d.need, d.have); i++) boxes.push(h("span", { class: i < Math.min(d.have, d.need) ? "filled" : i < d.have ? "extra" : null }));
        return h("div", { class: "depth-row" }, h("span", {}, d.name), h("div", { class: "depth-bar", title: `${d.have} of ${d.need}` }, boxes),
          h("span", { class: "num " + (d.gap ? "delta-down" : "muted") }, `${d.have}/${d.need}`));
      })),
    h("div", { class: "card" }, h("h2", { style: "margin-bottom:6px" }, "Suggested upgrades"),
      a.upgrades.length ? a.upgrades.map((u) => h("div", { class: "upgrade" },
        h("div", {}, h("a", { href: `#/player/${u.out.player_id}` }, h("b", {}, u.out.name)),
          h("div", { class: "small muted" }, `${POS_SHORT[u.out.position] || ""} · ${u.out.team} · form ${fmt(u.out.form)}`)),
        h("span", { class: "arrow" }, "→"),
        h("ul", {}, u.targets.map((t) => h("li", {}, h("span", {}, h("a", { href: `#/player/${t.player_id}` }, t.name),
          h("span", { class: "muted small" }, ` ${t.team} · ${fmt(t.form)}`)), h("span", { class: "gain" }, `+${fmt(t.gain)}`))))))
        : h("p", { class: "muted" }, "No clear upgrades: nobody on Watch or Trade has a much better player available at the same position."))));
  return out;
}

async function pagePlaylist(id) {
  const pl = await get(`/api/playlists/${id}`);
  const a = pl.is_pool ? await get(`/api/playlists/${id}/assessment`) : null;
  const editBox = h("div", { class: "card", hidden: true }, h("h2", { style: "margin-bottom:12px" }, "Edit playlist"),
    playlistForm(pl, async (body) => { await api("PATCH", `/api/playlists/${id}`, body); toast("Saved"); route(); }, "Save"));
  const actions = [
    h("button", { class: "btn", type: "button", onclick: () => { editBox.hidden = !editBox.hidden; } }, "Edit"),
    h("button", { class: "btn danger", type: "button", onclick: async () => {
      if (!confirm(`Delete the playlist "${pl.name}"? The players stay in the app.`)) return;
      await api("DELETE", `/api/playlists/${id}`);
      toast("Playlist deleted");
      go("#/playlists");
    } }, "Delete"),
  ];
  const title = [pl.name, " ", pl.is_pool ? h("span", { class: "badge pool" }, "Pool") : null];
  const body = [pageHead(title, pl.description || (pl.is_pool ? "Your hockey pool." : null), ...actions), editBox];
  if (a) body.push(h("div", { class: "stack" }, assessmentView(a)));
  body.push(h("div", { class: "card" },
    h("div", { class: "card-head" }, h("h2", {}, `Players (${pl.players.length})`)),
    playerPicker(async (p) => {
      try {
        await api("POST", `/api/playlists/${id}/players`, { player_id: p.player_id });
        toast(`Added ${p.name}`);
        route();
      } catch (err) { toast(err.message); }
    }),
    h("div", { style: "margin-top:12px" }, playerTable(pl.players, {
      empty: "No players yet. Use the search above, or the “Add to playlist” menu on any player's page.",
      remove: async (p) => { await api("DELETE", `/api/playlists/${id}/players/${p.player_id}`); toast(`Removed ${p.name}`); route(); },
    }))));
  return body;
}

// --- Router ---

const ROUTES = [
  [/^#?\/?$/, "home", pageHome],
  [/^#\/players$/, "players", pagePlayers],
  [/^#\/player\/(\d+)$/, "players", pagePlayer],
  [/^#\/games$/, "games", pageGames],
  [/^#\/game\/(\d+)$/, "games", pageGame],
  [/^#\/favorites$/, "favorites", pageFavorites],
  [/^#\/playlists$/, "playlists", pagePlaylists],
  [/^#\/playlist\/(\d+)$/, "playlists", pagePlaylist],
];

function go(hash) {
  if (location.hash === hash) route();
  else location.hash = hash;
}

let routeToken = 0;
async function route() {
  hideTip();
  const token = ++routeToken;
  const [path, query = ""] = (location.hash || "#/").split("?");
  const match = ROUTES.find(([re]) => re.test(path)) || ROUTES[0];
  const [re, nav, page] = match;
  const arg = path.match(re)?.[1];
  document.querySelectorAll("[data-nav]").forEach((a) => a.classList.toggle("active", a.dataset.nav === nav));
  app.classList.add("loading");
  try {
    const content = await page(arg !== undefined ? Number(arg) : new URLSearchParams(query));
    if (token !== routeToken) return;
    clear(app);
    append(app, [content]);
    const heading = app.querySelector("h1");
    document.title = heading ? `${heading.textContent.trim()} · Hockey-Vision` : "Hockey-Vision";
  } catch (err) {
    if (token !== routeToken) return;
    clear(app).append(h("div", { class: "card empty" }, h("h2", {}, "Something went wrong"), h("p", {}, err.message)));
  }
  app.classList.remove("loading");
}

initTheme();
window.addEventListener("hashchange", () => { route(); window.scrollTo(0, 0); });
matchMedia("(prefers-color-scheme: dark)").addEventListener("change", () => { if (!storedTheme()) route(); });
route();
