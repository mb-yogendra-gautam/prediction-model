# Product Requirements Document (PRD) - MB Parallax

### Product: Business Simulator Platform (Studios)

---

## 1. Problem Statement

Studio owners and managers often struggle to understand **which operational levers most influence revenue growth** whether itâ€™s membership retention, intro offer conversions, or upselling services. While Mindbody provides robust analytics, it lacks a **scenario simulation layer** that helps businesses plan proactively: What if I improved my retention by 3% or What happens if I increase my average ticket by $5

**Pain Points:**

- Decisions are reactive, not predictive.
- Owners lack clarity on how KPIs (sales, retention, utilization) interrelate.
- No sandbox environment to test what-if growth strategies.

---

## 2. Product Vision

Empower every Mindbody studio to **simulate, predict, and plan their business growth** by adjusting controllable leversturning insight into foresight.

**Vision Statement:**

> Help studios see the future of their business, not just the past through an intuitive simulator that models growth based on actionable levers.

---

## 3. Goals & Success Metrics

| **Goal**                               | **Metric / Success Criteria**                                       |
| -------------------------------------- | ------------------------------------------------------------------- |
| Help studios project business outcomes | 80% of test users can simulate at least one 3 month growth scenario |
| Identify top levers driving change     | Model shows clear revenue/retention impact by lever                 |
| Improve engagement with Insights       | +25% increase in feature usage (week-over-week)                     |
| Deliver shareable insights             | Users export or share simulations with staff                        |

---

## 4. Target Users & Personas

**Primary:** Studio Owners & Managers

- Use Case: Plan growth targets (sales + retention).
- Example: How can I grow my revenue by 5% in 3 months?
- Example: What will be my revenue if I increase member count by 100 and class count by 10?

**Secondary:** Consultants / Regional Managers

- Use Case: Compare performance across studios, test strategies before rollout.

---

## 5. Core Experience

### User Story Flow

1. **Adjust Levers** Membership, class, retention, intro offers, staff utilization, upsell rate, avg ticket.
2. **Simulate** Predicts future performance using historical data + modeled impact.
3. **See Outcomes** Visual summary of projected revenue, class fills, member count.
4. **Save Scenario** Save simulation as a plan, share to Slack/Email, or export CSV.

---

## 6. Key Features

| **Feature**               | **Description**                                                                       |
| ------------------------- | ------------------------------------------------------------------------------------- |
| **Lever Panel**           | Adjustable sliders: retention rate, avg ticket, new members, utilization, class %.    |
| **Simulator Engine**      | Predicts future KPIs using regression based on past 12 months of Mindbody data.       |
| **Suggested Steps**       | Auto-generated actions (Focus on reactivating lapsed members, Offer add-on packages). |
| **Scenario Save & Share** | Save named scenarios; export or share to team (Slack)/**Email/CSV/PNG**.              |

---

## 7. System Architecture (Hackathon MVP)

**Stack:** Next.js + FastAPI + Postgres + OpenAI (for text summary + suggestions) + ML prediction models

**Flow:**

1. Store baseline metrics for last 12 months.
2. When user adjusts levers, engine applies weighted formulas to project outcomes.
3. AI module summarizes predicted results + actionable next steps.
4. Display results visually and enable export/share.

---

## 8. Sample Lever Logic (Simplified)

| **Lever**         | **Effect on Sales**    | **Weight (Hackathon Default)** |
| ----------------- | ---------------------- | ------------------------------ |
| Retention         | +1% +0.8% sales growth | 0.8                            |
| New Memberships   | +1% +1% sales growth   | 1.0                            |
| Avg Ticket        | +$1 +0.5% sales growth | 0.5                            |
| Staff Utilization | +1% +0.4% sales growth | 0.4                            |

---

## 9. Design Concepts

- **Dashboard Layout:** Left = levers; Right = projection graph.
- **Simulation Graph:** Line chart comparing baseline vs. simulated projection.
- **Next Best Action:** Suggests 3 most impactful actions based on inputs.

---

## 10. Hackathon Execution Plan (5 Days)

| **Day** | **Focus**                   | **Deliverables**                                |
| ------- | --------------------------- | ----------------------------------------------- |
| Day 1   | Kickoff & Schema            | Pull KPI APIs, DB setup, base UI scaffold       |
| Day 2   | Lever UI + Simulation Model | Build lever panel, static regression model      |
| Day 3   | AI Suggestions              | Integrate GPT API for recommendation text       |
| Day 4   | Save/Share                  | Scenario save, Slack integration, charts polish |
| Day 5   | Demo                        | End-to-end run: input simulate suggest share    |

---

## 11. Risks & Mitigations

| **Risk**                             | **Mitigation**                                       |
| ------------------------------------ | ---------------------------------------------------- |
| Inaccurate model due to limited data | Use static regression from anonymized sample dataset |
| Scope creep (AI/UX)                  | Fix MVP to single flow per user type                 |
| Over-complex UI                      | Prioritize intuitive sliders and short feedback loop |

---

## 12. Stretch Goals

- Compare Scenarios mode for A/B planning.
- Industry benchmarks overlay (Yoga vs. Pilates vs. Wellness).
- Auto-Optimize button to recommend ideal lever mix.

---

## 13. Demo Narrative

1. Adjust levers (e.g., +2% retention, +1 new member/day).
2. See projected growth graph + AI-suggested actions.
3. Save and share scenario to Slack.

---

## 14. Definition of Done

Studio can set goals and adjust levers.

Simulator projects outcome with visual summary.

AI suggests actionable next steps.

Scenario can be saved/shared.

Demo shows tangible what-if business planning.

---

**Outcome:**

An intuitive business simulator that transforms static analytics into **interactive growth planning**, enabling Mindbody customers to understand _how_ to hit their business goals, not just _what_ they are.
