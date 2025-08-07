### **ElevatorInsight - Product Brief**



##### **Problem**

Unplanned elevator downtime in high-rise buildings causes tenant frustration, safety risks, and expensive emergency call outs. Existing maintenance schedules are either fixed-interval (wasteful) or reactive (too late). Raw sensor logs- door-cycle counts, vibration, motor current- are plentiful but unreadable for non-data-savvy facility managers



##### **Solution**

**ElevatorInsight** is a lightweight, cloud-hosted service that converts real-time elevator telemetry into **actionable, plain-language alerts** delivered straight to Slack or Microsoft Teams. An XGBoost early-warning model scores each door system every few minutes; a GPT-4-powered agent turns the numeric risk output and SHAP explanations into a concise recommendation (ex. "Adjust door rollers on Car 2 within 5 days to avoid failure"). A browser dashboard shows live risk percentages and recent interventions 



##### **Key Features**

* **Predictive scoring API -** scores streamed sensor rows in < 200ms
* **Agentic alert generator -** explains top failure drivers in plain English
* **Slack integration -** one-click "Create work order" button with timestamp
* **Read-only dashboard -** React/Vite front-end for status at a glance
* **Explainability -** SHAP waterfall visual per alert for technician trust



##### **Target Users**

Building-operations managers, 3rd-party elevator service contractors, and asset-portfolio owners seeking to cut downtime and maintenance costs 



##### **Tech Stack**

Python 3 + Pandas, XGBoost, SHAP · FastAPI on Replit · LangChain + OpenAI GPT-4o · Slack Webhooks · React/Vite dashboard deployed to Netlify.



##### **Success Metrics (MVP)**

* **Model performance:** F1 ≥ 0.75 on 7-day door-failure prediction.
* **Operational latency:** end-to-end inference ≤ 200 ms (sensor row → risk JSON).



ElevatorInsight turns elevators from passive assets into proactively managed, data-driven systems- keeping passengers moving and budgets in check.

