### 1. Core Argument

|  Slide Decks (Google Slides)      |  [[Unidash]] (Dashboard)                          |
| --------------------------------- | ------------------------------------------------- |
| Static snapshot; rebuild each WBR | Live, filterable, hover details                   |
| Manual copy‑paste → version drift | Reads directly from governed Hive/Manifold tables |
| DS hours on layout tweaks         | DS hours on new hypotheses, more markets          |
| Max 1‑2 drill levels              | Unlimited drill (metric → L4 → L8)                |
| Exec consumption only             | Self‑serve for AMs, Sales Ops, VPs                |

---

### 2. Why stakeholders care? 

| Stakeholder             | What’s in it for them?                                                                                                                          |
| ----------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------- |
| **DS Manager**          | *200 DS hrs/yr saved*; focus on scaling to new markets, deeper hypotheses, DFS traversal; LLM integration becomes trivial because data is live. |
| **DE**                  | Can own refresh pipeline; no slide‑layout maintenance; single governed source; easier ACL.                                                      |
| **SSPO / WBR owners**   | Still get a crisp exec summary slide; can screenshot any Unidash chart; "Manager View" dashboards come for free.                                |
| **Sales Leaders & AMs** | Filter by team/metric/date; ask Metamate: "Compare week 3 vs 9" on‑the‑fly.                                                                     |

---

### 3. New Capabilities Enabled by Unidash + Metamate

1. **LLM Q\&A panel** – Metamate reads call‑outs & payload; answers “persistent issues across all L4s?”
2. **Hover → preview card** – mini‑waterfall + AM quote pops up, no extra slide.
3. **Topic extraction** – Metamate tags AM closed‑lost comments; dashboard shows tag cloud linked to hypothesis.
4. **Cross‑time compare toggle** – ridge plot week‑over‑week; one click screenshot.
5. **What‑if mix simulation slider** – shift 5 % pitches to best‑fit product and see Δ win‑rate in real time.

---

### Ranked reasons to shift **root-cause automation** from pure Slides to an **Unidash-first model**

| Rank  | Impact | Rationale                                                                                                                                             | Who benefits                  |
| ----- | ------ | ----------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------- |
| **1** | 🔴🔴🔴 | **Self-serve diagnostic filters (region, team, metric, date)** replace 1:1 deck requests; DS time goes to model improvements instead of slide re-cuts | SSPO frontline, Sales leaders |
| **2** | 🔴🔴   | **LLM (Metamate) layer on top of live data**: *“Compare week 3 vs 9”* / *“Show verticals common to all lagging L4s”* / surfacing AM comment excerpts  | All leadership tiers          |
| **3** | 🔴🔴   | **Drill-to-evidence hover** (preview card with mini-waterfall + AM quotes) removes 5–10 slide hops per question                                       | VP reviews, QBR live demos    |
| **4** | 🔴🔴   | **One canonical data source** (Hive / Manifold behind Unidash) → DE can own orchestration; no copy-paste drift                                        | DE, Compliance                |
| **5** | 🔴     | **Manager-view dashboards** reading the same table power future SSPO roadmap without extra DS coding                                                  | SSPO analytics                |
| **6** | 🔴     | **Screenshot → Slide** workflow still possible for weekly WBR deck; DS keeps only 3 static summary pages                                              | WBR owner                     |

---

### What **stays in Slides** (low-touch)

1. Exec one-pager: KPI heatmap & CTA bullets
2. 1 PNG per flagged KPI (pulled from Unidash “export”)
3. Narrative auto-filled from payload (still Jinja → Slides API)

*→ Slides remain the “storybook”; Unidash is the evidence warehouse.*

---

### Feature ideas **filtered & prioritised**

| Priority | Feature                          | Notes / blockers                                       |
| -------- | -------------------------------- | ------------------------------------------------------ |
| P0       | Region / team filter, export PNG | Unidash native; critical path for adoption             |
| P0       | Drill hover preview card         | Quick win: tooltip JSON → mini-HTML                    |
| P0       | Metamate Q\&A panel              | Quick win: ensure data is stored properly              |
| P1       | DFS auto-traverse button         | Requires recursive hypothesis engine but no UI blocker |
| P2       | “What-if mix” simulator slider   | Use JS front-end; low DS effort after payload served   |
| P2       | Cross-quarter change detector    | Needs historical tables; can piggyback pacing calc     |
Not yet put to the table / more details
* Interactive heatmaps (product × vertical)
* Stage-flow alluvial with hover counts
* Pacing curves & velocity tables
* Hypothesis ranking table with sort / search
* CI topic uplift matrix
