# DSL coverage over the libero-pro dev task set (兼容性 / compatibility map)

How many tasks can the declarative `judge_dsl` relations actually judge? Mapping every base
task of the dev suites (the `_swap` / `_task` variants are perturbations of these same goals)
to a relation. This is the quantitative case for "a SMALL set of geometric relations is
general" — the agent only ever picks from this short menu.

Relations: `place_in · on_top_of · opened · closed · lifted · grasped · next_to · removed_from`
(+ `all_of` to AND-combine for multi-condition goals).

## libero_object (10/10 ✅) — all are `place_in`
Every task is "pick up X and place it in the basket":
`J.judge(ctx, "place_in", target="<X>", reference="basket")`. The object name X is the only
thing that varies (alphabet soup, cream cheese, ketchup, milk, ...).

## libero_spatial (10/10 ✅) — all are `on_top_of`
Every task is "pick up the black bowl [spatial qualifier] and place it on the plate":
`J.judge(ctx, "on_top_of", target="black bowl", reference="plate")`. The spatial qualifier
("between the plate and the ramekin", "next to the cookie box", "on the stove") only
disambiguates WHICH bowl to grasp — that is one-time **grounding** done by the execution code,
NOT the per-turn state judgment, which is identical across all 10.

## libero_goal (8/10 ✅, 2 residual)
| # | instruction | relation |
|---|---|---|
| 0 | open the middle drawer of the cabinet | `opened(target="drawer handle")` ✅ |
| 1 | put the bowl on the stove | `on_top_of(bowl, stove)` ✅ |
| 2 | put the wine bottle on top of the cabinet | `on_top_of(wine bottle, cabinet)` ✅ |
| 3 | open the top drawer and put the bowl inside | `all_of(opened(drawer handle), place_in(bowl, drawer))` ✅ |
| 4 | put the bowl on top of the cabinet | `on_top_of(bowl, cabinet)` ✅ |
| 5 | push the plate to the front of the stove | `next_to(plate, stove)` ⚠️ approximate (proximity, not exact "front") |
| 6 | put the cream cheese in the bowl | `place_in(cream cheese, bowl)` ✅ |
| 7 | **turn on the stove** | ❌ **non-spatial state** — see residual |
| 8 | put the bowl on the plate | `on_top_of(bowl, plate)` ✅ |
| 9 | put the wine bottle on the rack | `on_top_of(wine bottle, rack)` ✅ |

## Tally
**28 / 30 dev tasks** are judged by `place_in`, `on_top_of`, `opened` (+ `all_of`). Two
residuals, both anticipated as the geometry boundary:
- **goal[5] "push to the front of the stove"** — a *position/proximity* goal, not a relation
  between two whole objects. `next_to(plate, stove)` approximates it; a dedicated
  `moved_to_region` (against a sensed landmark) would be exact. Low priority (1 task).
- **goal[7] "turn on the stove"** — a *non-spatial* state (burner on/off), the one category
  pure 3D geometry cannot read. The honest options are (a) a knob-ROTATION test if "on" maps to
  a knob turn (geometric, needs the knob's angular signal), or (b) a *sparse* appearance check
  (NOT the per-turn VDM — a single local check). Documented as the known residual; not forced
  into geometry. This is exactly the boundary stated in the design discussion (geometry judges
  spatial/articulated state; non-spatial state needs an appearance signal).

**Takeaway:** the per-turn LLM state judgment (the VDM) is removable for ~93% of dev tasks with
THREE relations; the residue is one proximity goal and one genuinely non-spatial state — and
neither needs a *per-turn* VLM.
