# Design system

| Token | Hex | Use |
|---|---|---|
| Navy | `#0B1F3A` | text, headers, observed CGM line, twin-state emphasis |
| Teal | `#0F9D8F` | primary actions, selected model, twin forecast |
| Background | `#F6F8FB` | app background |
| EHR purple | `#5E4691` / light `#EFE9F7` | historical EHR stream tags |
| Sensor blue | `#24497D` / light `#E5EEF9` | dynamic sensor stream tags |
| Amber | `#C77700` / `#9A5B00` | event threshold, simulated / warning labels |
| Red | `#C0392B` | elevated band, alert threshold (always with a text label) |

Model colours in report charts are fixed per model (never re-ordered): gradient boosting teal `#0F9D8F`,
logistic regression blue `#2F5D9E`, random forest amber `#D08C2E`, rule baseline grey dashed `#8A94A6`.
The categorical palette was checked with a colour-vision-deficiency validator: all checks pass; amber is
below 3:1 contrast on white, so every chart that uses it also has a legend and direct value labels.

Rules: one y-axis per chart; observed data solid, forecast window shaded only; colour never carries meaning
alone (labels and legends always present).
