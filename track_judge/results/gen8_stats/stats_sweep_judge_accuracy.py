"""Statistical sweep: geometric judge accuracy vs GT over many sampled states.
on_top_of(bowl, plate) on libero_goal task8, across seeds x placements. Measures the
judge-vs-GT confusion (the thing the VDM gets wrong: missed-done / false-done) and the
judge's run-to-run determinism. Runtime LLM not involved; pure judge accuracy."""
import os, sys, collections
import numpy as np
sys.path.insert(0, os.getcwd())
from capx.envs.simulators.libero import FrankaLiberoEnv
import capx.integrations  # noqa
from track_judge.algo.track_judge import TrackJudge
from track_judge.algo import judge_dsl as J

SEEDS = [1, 2, 3]
# placements relative to plate: (dx, dy, dz, settle, intent)
PLACE = [
    (0.00, 0.00, 0.015, True,  "on"),
    (0.025,0.00, 0.015, True,  "on"),
    (0.00, 0.025,0.015, True,  "on"),
    (0.20, 0.00, 0.00,  False, "off"),
    (-0.15,0.05, 0.00,  False, "off"),
    (0.00, 0.18, 0.00,  False, "off"),
    (0.00, 0.00, 0.12,  False, "hover"),  # over plate but high -> not seated
]
judge = TrackJudge(socket_path=None, viz_dir=None)
B = slice(9, 16)

def run_state(env, sim, plate, place):
    dx, dy, dz, settle, intent = place
    env.enable_dense_rgbd_capture(True, clear=True)
    quat = sim.data.qpos[B][3:].copy()
    goal = np.array([plate[0]+dx, plate[1]+dy, plate[2]+dz])
    p0 = sim.data.qpos[B][:3].copy()
    for i in range(14):
        a = min(1.0, i/10)
        sim.data.qpos[B] = np.concatenate([(1-a)*p0 + a*goal, quat]); sim.forward(); env._record_frame()
    if settle:
        rq = sim.data.qpos[0:9].copy()
        for _ in range(50):
            sim.step(); sim.data.qpos[0:9]=rq; sim.data.qvel[0:9]=0; sim.forward(); env._record_frame()
    rgb, depth = env.get_rgbd_frames_range(0, 10**9)
    idx = np.linspace(0, len(rgb)-1, 12).round().astype(int)
    cam = env.camera_params()
    v = judge.judge_turn(rgb=np.ascontiguousarray(rgb[idx],np.uint8),
                         depth=np.ascontiguousarray(depth[idx],np.float32),
                         K=cam["K"], world_to_cam=cam["world_to_cam"],
                         task_description="bowl on plate", judge_fn=lambda c: J.judge(c,"on_top_of",target="bowl",reference="plate"),
                         state={}, viz_tag="s")
    return bool(v["done"]), bool(env.task_completed()), intent

conf = collections.Counter(); rows=[]
for seed in SEEDS:
    env = FrankaLiberoEnv("libero_goal", 8, privileged=False, max_steps=30000, control_freq=20, enable_render=True)
    env.reset(seed=seed); sim = env.handle.env.sim
    plate = sim.data.xpos[sim.model.body_name2id("plate_1_main")].copy()
    for place in PLACE:
        env.reset(seed=seed); sim = env.handle.env.sim
        plate = sim.data.xpos[sim.model.body_name2id("plate_1_main")].copy()
        jd, gt, intent = run_state(env, sim, plate, place)
        key = ("done" if jd else "notdone", "GTdone" if gt else "GTnot")
        conf[key]+=1; rows.append((seed, intent, jd, gt, jd==gt))
        print(f"  seed{seed} {intent:<5} judge_done={jd} GT={gt} {'OK' if jd==gt else 'MISMATCH'}")

n=len(rows); agree=sum(r[4] for r in rows)
td=conf[("done","GTdone")]; fd=conf[("done","GTnot")]; md=conf[("notdone","GTdone")]; tn=conf[("notdone","GTnot")]
print(f"\n=== geometric judge vs GT over N={n} sampled states ({len(SEEDS)} seeds x {len(PLACE)} placements) ===")
print(f"  accuracy (judge==GT): {agree}/{n} = {100*agree/n:.1f}%")
print(f"  confusion: true_done={td} false_done={fd} missed_done={md} true_notdone={tn}")
prec = td/(td+fd) if td+fd else float('nan'); rec = td/(td+md) if td+md else float('nan')
print(f"  precision={prec:.2f}  recall={rec:.2f}  (false_done={fd}, missed_done={md})")
