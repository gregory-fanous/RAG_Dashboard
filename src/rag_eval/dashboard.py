from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .models import BenchmarkReport, dataclass_to_dict


def build_dashboard_html(report: BenchmarkReport) -> str:
    return build_dashboard_html_from_payload(dataclass_to_dict(report))


def build_dashboard_html_from_payload(payload: dict[str, Any]) -> str:
    data_json = json.dumps(payload)

    template = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>RAG Evaluation Dashboard</title>
  <style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;600;700&family=Azeret+Mono:wght@400;500&display=swap');

    :root {
      --bg-0: #f5f7f2;
      --bg-1: #dfe8c8;
      --ink: #1a2917;
      --ink-soft: #41583d;
      --accent: #d95d39;
      --accent-2: #20736f;
      --card: #ffffff;
      --line: #b9c5a7;
    }

    * {
      box-sizing: border-box;
    }

    body {
      margin: 0;
      font-family: 'Space Grotesk', sans-serif;
      color: var(--ink);
      background: radial-gradient(circle at 0% 0%, var(--bg-1) 0%, var(--bg-0) 48%),
                  radial-gradient(circle at 100% 100%, #f0e3c3 0%, transparent 45%);
      min-height: 100vh;
    }

    .shell {
      max-width: 1160px;
      margin: 0 auto;
      padding: 28px 20px 60px;
    }

    .hero {
      background: linear-gradient(120deg, #233725 0%, #1f594f 48%, #93422a 100%);
      color: #f9fdf4;
      border-radius: 18px;
      padding: 30px;
      position: relative;
      overflow: hidden;
      box-shadow: 0 18px 40px rgba(26, 41, 23, 0.18);
    }

    .hero::after {
      content: "";
      position: absolute;
      right: -40px;
      top: -40px;
      width: 220px;
      height: 220px;
      border-radius: 50%;
      background: rgba(255, 255, 255, 0.12);
    }

    .label {
      display: inline-block;
      letter-spacing: 0.1em;
      font-size: 12px;
      text-transform: uppercase;
      margin-bottom: 10px;
      opacity: 0.92;
    }

    h1 {
      margin: 0;
      font-size: clamp(28px, 3.8vw, 42px);
      line-height: 1.05;
      max-width: 16ch;
    }

    .manifesto {
      margin-top: 16px;
      max-width: 64ch;
      font-size: 16px;
      line-height: 1.45;
      opacity: 0.95;
    }

    .meta {
      margin-top: 18px;
      font-family: 'Azeret Mono', monospace;
      font-size: 12px;
      opacity: 0.85;
    }

    .grid {
      display: grid;
      grid-template-columns: repeat(12, minmax(0, 1fr));
      gap: 14px;
      margin-top: 18px;
    }

    .card {
      background: var(--card);
      border: 1px solid var(--line);
      border-radius: 14px;
      padding: 16px;
    }

    .stat {
      grid-column: span 4;
    }

    .stat small {
      display: block;
      font-family: 'Azeret Mono', monospace;
      text-transform: uppercase;
      color: var(--ink-soft);
      letter-spacing: 0.09em;
      font-size: 11px;
      margin-bottom: 8px;
    }

    .stat strong {
      font-size: clamp(22px, 3vw, 31px);
    }

    .section {
      margin-top: 16px;
    }

    .section h2 {
      margin: 0 0 10px;
      font-size: 20px;
      letter-spacing: 0.02em;
    }

    .table-wrap {
      overflow-x: auto;
    }

    table {
      border-collapse: collapse;
      min-width: 860px;
      width: 100%;
    }

    thead th {
      text-align: left;
      font-size: 12px;
      text-transform: uppercase;
      letter-spacing: 0.08em;
      color: var(--ink-soft);
      padding: 10px 8px;
      border-bottom: 1px solid var(--line);
      font-family: 'Azeret Mono', monospace;
    }

    tbody td {
      padding: 12px 8px;
      border-bottom: 1px solid #e4ead7;
      font-size: 14px;
    }

    tbody tr:hover {
      background: #f6f9ef;
    }

    .feature-tags {
      display: inline-flex;
      flex-wrap: wrap;
      gap: 6px;
    }

    .chip {
      font-size: 11px;
      padding: 2px 8px;
      border-radius: 999px;
      background: #e9f3df;
      border: 1px solid #c7d6ba;
      color: #2e4f30;
      font-family: 'Azeret Mono', monospace;
    }

    #scatter {
      width: 100%;
      min-height: 360px;
    }

    .legend {
      margin-top: 8px;
      font-size: 12px;
      color: var(--ink-soft);
      font-family: 'Azeret Mono', monospace;
      display: flex;
      gap: 12px;
      flex-wrap: wrap;
    }

    .dot {
      display: inline-block;
      width: 10px;
      height: 10px;
      border-radius: 50%;
      margin-right: 5px;
      vertical-align: middle;
    }

    .bars {
      display: grid;
      gap: 10px;
    }

    .bar-row {
      display: grid;
      grid-template-columns: 180px 1fr 120px;
      align-items: center;
      gap: 10px;
    }

    .bar-label {
      font-size: 13px;
      font-weight: 600;
    }

    .bar-track {
      height: 14px;
      background: #e9efdc;
      border: 1px solid #ced9bc;
      border-radius: 999px;
      overflow: hidden;
    }

    .bar-fill {
      height: 100%;
      border-radius: 999px;
      background: linear-gradient(90deg, #20736f 0%, #d95d39 100%);
    }

    .bar-value {
      font-size: 12px;
      font-family: 'Azeret Mono', monospace;
      text-align: right;
    }

    @media (max-width: 840px) {
      .stat {
        grid-column: span 12;
      }

      .bar-row {
        grid-template-columns: 1fr;
        gap: 6px;
      }

      .bar-value {
        text-align: left;
      }
    }
  </style>
</head>
<body>
  <main class="shell">
    <section class="hero">
      <span class="label">RAG Governance Dashboard</span>
      <h1>Most teams measure RAG wrong.</h1>
      <p class="manifesto">This benchmark evaluates retrieval quality, generation faithfulness, and cost-latency economics as one system. If a RAG stack cannot prove these tradeoffs with repeatable numbers, it is not production-ready.</p>
      <div class="meta" id="meta"></div>
    </section>

    <section class="grid" id="stats"></section>

    <section class="section card">
      <h2>Leaderboard</h2>
      <div class="table-wrap">
        <table>
          <thead>
            <tr>
              <th>Strategy</th>
              <th>Features</th>
              <th>Precision@k</th>
              <th>Recall@k</th>
              <th>MRR</th>
              <th>Hallucination Rate</th>
              <th>Quality</th>
              <th>Latency (ms)</th>
              <th>Avg Cost ($)</th>
            </tr>
          </thead>
          <tbody id="leaderboard-body"></tbody>
        </table>
      </div>
    </section>

    <section class="section card">
      <h2>Latency vs Quality Curve</h2>
      <div id="scatter"></div>
      <div class="legend">
        <span><span class="dot" style="background:#d95d39"></span>Strategy Point</span>
        <span><span class="dot" style="background:#20736f"></span>Pareto Frontier</span>
      </div>
    </section>

    <section class="section card">
      <h2>Token Cost Analysis</h2>
      <div class="bars" id="cost-bars"></div>
    </section>
  </main>

  <script>
    const DATA = __DATA__;

    const aggregates = DATA.strategy_results
      .map((item) => item.aggregate)
      .sort((a, b) => b.avg_quality_score - a.avg_quality_score);

    const pct = (value) => `${(value * 100).toFixed(1)}%`;
    const ms = (value) => `${value.toFixed(1)} ms`;
    const usd = (value) => `$${value.toFixed(4)}`;

    function featureTags(row) {
      const tags = [
        `backend:${row.retrieval_backend || 'in_memory'}`,
        `chunk:${row.chunking_strategy}`,
        row.logicrag ? 'LogicRAG' : null,
        row.recursive_retrieval ? 'recursive' : null,
        row.graph_augmentation ? 'graph' : null,
        row.hyde ? 'HyDE' : null,
        row.self_rag ? 'Self-RAG' : null,
      ].filter(Boolean);

      return `<span class="feature-tags">${tags.map((tag) => `<span class="chip">${tag}</span>`).join('')}</span>`;
    }

    function renderMeta() {
      const line = `${DATA.benchmark_name} | run ${DATA.run_id} | ${new Date(DATA.created_at).toLocaleString()}`;
      document.getElementById('meta').textContent = line;
    }

    function renderStats() {
      const best = aggregates[0];
      const fastest = [...aggregates].sort((a, b) => a.avg_latency_ms - b.avg_latency_ms)[0];
      const cheapest = [...aggregates].sort((a, b) => a.avg_token_cost_usd - b.avg_token_cost_usd)[0];

      const cards = [
        { label: 'Top Quality Strategy', value: best ? best.strategy_name : 'n/a', sub: best ? pct(best.avg_quality_score) : '-' },
        { label: 'Fastest Strategy', value: fastest ? fastest.strategy_name : 'n/a', sub: fastest ? ms(fastest.avg_latency_ms) : '-' },
        { label: 'Cheapest Per Query', value: cheapest ? cheapest.strategy_name : 'n/a', sub: cheapest ? usd(cheapest.avg_token_cost_usd) : '-' },
      ];

      const html = cards
        .map((card) => `<article class="card stat"><small>${card.label}</small><strong>${card.value}</strong><div>${card.sub}</div></article>`)
        .join('');

      document.getElementById('stats').innerHTML = html;
    }

    function renderLeaderboard() {
      const body = document.getElementById('leaderboard-body');
      body.innerHTML = aggregates
        .map(
          (row) => `
            <tr>
              <td><strong>${row.strategy_name}</strong></td>
              <td>${featureTags(row)}</td>
              <td>${pct(row.avg_precision_at_k)}</td>
              <td>${pct(row.avg_recall_at_k)}</td>
              <td>${row.mrr.toFixed(3)}</td>
              <td>${pct(row.avg_hallucination_rate ?? (1 - row.avg_hallucination_score))}</td>
              <td><strong>${pct(row.avg_quality_score)}</strong></td>
              <td>${ms(row.avg_latency_ms)}</td>
              <td>${usd(row.avg_token_cost_usd)}</td>
            </tr>
          `
        )
        .join('');
    }

    function paretoFrontier(rows) {
      const sorted = [...rows].sort((a, b) => a.avg_latency_ms - b.avg_latency_ms);
      const frontier = [];
      let maxQuality = -Infinity;
      for (const row of sorted) {
        if (row.avg_quality_score > maxQuality) {
          frontier.push(row);
          maxQuality = row.avg_quality_score;
        }
      }
      return frontier;
    }

    function drawScatter() {
      const container = document.getElementById('scatter');
      const width = container.clientWidth || 900;
      const height = 360;
      const padding = { top: 18, right: 22, bottom: 44, left: 56 };
      const plotW = width - padding.left - padding.right;
      const plotH = height - padding.top - padding.bottom;

      const minLatency = Math.min(...aggregates.map((row) => row.avg_latency_ms));
      const maxLatency = Math.max(...aggregates.map((row) => row.avg_latency_ms));
      const latSpan = Math.max(1, maxLatency - minLatency);

      const minQuality = Math.min(...aggregates.map((row) => row.avg_quality_score));
      const maxQuality = Math.max(...aggregates.map((row) => row.avg_quality_score));
      const qualSpan = Math.max(0.01, maxQuality - minQuality);

      const x = (lat) => padding.left + ((lat - minLatency) / latSpan) * plotW;
      const y = (quality) => padding.top + (1 - (quality - minQuality) / qualSpan) * plotH;

      const svg = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
      svg.setAttribute('viewBox', `0 0 ${width} ${height}`);
      svg.setAttribute('width', '100%');
      svg.setAttribute('height', `${height}`);

      const bg = document.createElementNS('http://www.w3.org/2000/svg', 'rect');
      bg.setAttribute('x', '0');
      bg.setAttribute('y', '0');
      bg.setAttribute('width', `${width}`);
      bg.setAttribute('height', `${height}`);
      bg.setAttribute('fill', '#fcfef8');
      svg.appendChild(bg);

      for (let i = 0; i <= 4; i += 1) {
        const yVal = padding.top + (plotH * i) / 4;
        const line = document.createElementNS('http://www.w3.org/2000/svg', 'line');
        line.setAttribute('x1', `${padding.left}`);
        line.setAttribute('x2', `${padding.left + plotW}`);
        line.setAttribute('y1', `${yVal}`);
        line.setAttribute('y2', `${yVal}`);
        line.setAttribute('stroke', '#dde6cf');
        line.setAttribute('stroke-width', '1');
        svg.appendChild(line);
      }

      const xAxis = document.createElementNS('http://www.w3.org/2000/svg', 'line');
      xAxis.setAttribute('x1', `${padding.left}`);
      xAxis.setAttribute('x2', `${padding.left + plotW}`);
      xAxis.setAttribute('y1', `${padding.top + plotH}`);
      xAxis.setAttribute('y2', `${padding.top + plotH}`);
      xAxis.setAttribute('stroke', '#8ea58a');
      xAxis.setAttribute('stroke-width', '1.4');
      svg.appendChild(xAxis);

      const yAxis = document.createElementNS('http://www.w3.org/2000/svg', 'line');
      yAxis.setAttribute('x1', `${padding.left}`);
      yAxis.setAttribute('x2', `${padding.left}`);
      yAxis.setAttribute('y1', `${padding.top}`);
      yAxis.setAttribute('y2', `${padding.top + plotH}`);
      yAxis.setAttribute('stroke', '#8ea58a');
      yAxis.setAttribute('stroke-width', '1.4');
      svg.appendChild(yAxis);

      const xLabel = document.createElementNS('http://www.w3.org/2000/svg', 'text');
      xLabel.textContent = 'Average Latency (ms)';
      xLabel.setAttribute('x', `${padding.left + plotW / 2}`);
      xLabel.setAttribute('y', `${height - 10}`);
      xLabel.setAttribute('text-anchor', 'middle');
      xLabel.setAttribute('fill', '#41583d');
      xLabel.setAttribute('font-size', '12');
      xLabel.setAttribute('font-family', 'Azeret Mono, monospace');
      svg.appendChild(xLabel);

      const yLabel = document.createElementNS('http://www.w3.org/2000/svg', 'text');
      yLabel.textContent = 'Composite Quality Score';
      yLabel.setAttribute('x', '16');
      yLabel.setAttribute('y', `${padding.top + plotH / 2}`);
      yLabel.setAttribute('text-anchor', 'middle');
      yLabel.setAttribute('transform', `rotate(-90 16 ${padding.top + plotH / 2})`);
      yLabel.setAttribute('fill', '#41583d');
      yLabel.setAttribute('font-size', '12');
      yLabel.setAttribute('font-family', 'Azeret Mono, monospace');
      svg.appendChild(yLabel);

      const frontier = paretoFrontier(aggregates);
      if (frontier.length > 1) {
        const points = frontier.map((row) => `${x(row.avg_latency_ms)},${y(row.avg_quality_score)}`).join(' ');
        const polyline = document.createElementNS('http://www.w3.org/2000/svg', 'polyline');
        polyline.setAttribute('points', points);
        polyline.setAttribute('fill', 'none');
        polyline.setAttribute('stroke', '#20736f');
        polyline.setAttribute('stroke-width', '2.3');
        svg.appendChild(polyline);
      }

      for (const row of aggregates) {
        const cx = x(row.avg_latency_ms);
        const cy = y(row.avg_quality_score);

        const point = document.createElementNS('http://www.w3.org/2000/svg', 'circle');
        point.setAttribute('cx', `${cx}`);
        point.setAttribute('cy', `${cy}`);
        point.setAttribute('r', '5.5');
        point.setAttribute('fill', '#d95d39');
        point.setAttribute('stroke', '#7a2d19');
        point.setAttribute('stroke-width', '1');
        svg.appendChild(point);

        const label = document.createElementNS('http://www.w3.org/2000/svg', 'text');
        label.textContent = row.strategy_name;
        label.setAttribute('x', `${cx + 8}`);
        label.setAttribute('y', `${cy - 8}`);
        label.setAttribute('fill', '#1f3d1f');
        label.setAttribute('font-size', '11');
        label.setAttribute('font-family', 'Azeret Mono, monospace');
        svg.appendChild(label);
      }

      container.innerHTML = '';
      container.appendChild(svg);
    }

    function renderCostBars() {
      const container = document.getElementById('cost-bars');
      const sorted = [...aggregates].sort((a, b) => b.total_token_cost_usd - a.total_token_cost_usd);
      const maxCost = Math.max(...sorted.map((row) => row.total_token_cost_usd), 1e-9);

      container.innerHTML = sorted
        .map((row) => {
          const width = (row.total_token_cost_usd / maxCost) * 100;
          return `
            <div class="bar-row">
              <div class="bar-label">${row.strategy_name}</div>
              <div class="bar-track"><div class="bar-fill" style="width:${width.toFixed(1)}%"></div></div>
              <div class="bar-value">Total ${usd(row.total_token_cost_usd)} | ${row.total_tokens.toLocaleString()} toks</div>
            </div>
          `;
        })
        .join('');
    }

    renderMeta();
    renderStats();
    renderLeaderboard();
    drawScatter();
    renderCostBars();
    window.addEventListener('resize', drawScatter);
  </script>
</body>
</html>
"""

    return template.replace("__DATA__", data_json)


def write_dashboard(report: BenchmarkReport, output_path: str | Path) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(build_dashboard_html(report), encoding="utf-8")
