"""Select actual measured successes and failures."""
import json
from pathlib import Path
root=Path("results")
files=[root/name/"test_id_trajectories.jsonl" for name in
       ("qwen_grpo_A","qwen_grpo_B","qwen_grpo_C","qwen_sft","qwen_zero","baselines")
       if (root/name/"test_id_trajectories.jsonl").exists()]
successes=[]; failures=[]
for path in files:
    chosen_here={True:False,False:False}
    for line in path.open():
        row=json.loads(line); rec=row["record"]
        if not row["history"]: continue
        pool=successes if rec["success"] else failures
        if (len(pool)<3 and (not chosen_here[rec["success"]] or path.parent.name=="baselines")
            and not any(r["record"]["problem_id"]==rec["problem_id"] for r in pool)):
            pool.append(row); chosen_here[rec["success"]]=True
        if path.parent.name!="baselines" and (chosen_here[True] or len(successes)>=3) and (chosen_here[False] or len(failures)>=3): break
    if len(successes)>=3 and len(failures)>=3: break
Path("report").mkdir(exist_ok=True)
Path("report/case_studies.json").write_text(json.dumps({"successes":successes,"failures":failures},indent=2))
lines=["# Measured trajectory case studies","","Examples prefer one measured ID success and one failure from each reward variant A/B/C, selecting the first distinct problem IDs in file order. Other models/baselines are fallbacks only. These are illustrative cases, not a random sample for estimating rates.",""]
for label,rows in [("Success",successes),("Failure",failures)]:
    for j,row in enumerate(rows,1):
        r=row["record"]; history=row["history"]
        lines += [f"## {label} {j}: {r['method']} / {r['problem_id']}",
                  f"Seed {r['seed']}; reason {r['reason']}; {r['actions']} actions; invalid={r['invalid_actions']}; cycle={r['cycle']}.",
                  f"Final normalized residuals: stationarity={r['stationarity']:.3g}, primal={r['primal']:.3g}, dual={r['dual']:.3g}, complementarity={r['complementarity']:.3g}.","",
                  "| Step | Action | Active set | KKT merit |","|---|---|---|---|"]
        for h in history: lines.append(f"| {h['iteration']} | {h.get('action','initial')} | {h['active']} | {h['residuals']['merit']:.4g} |")
        if r["reason"]=="incorrect_stop": comment="The policy stopped before the independent verifier accepted all four residuals."
        elif r["invalid_actions"]: comment="Rejected actions consumed the decision budget without changing the active set."
        elif r["cycle"]: comment="Repeated active sets reveal cycling in this repair heuristic; no global convergence is claimed."
        elif r["success"]: comment="The final explicit STOP produced a valid primal-dual certificate; exact oracle-set equality was not required."
        else: comment="The policy exhausted its budget before issuing a certified STOP."
        merits=[h["residuals"]["merit"] for h in history]
        increases=sum(b>a+1e-10 for a,b in zip(merits,merits[1:]))
        drops=sum(h.get("action","").strip().startswith("DROP(") for h in history)
        lines += ["",comment,
                  f"Merit increased on {increases} transitions; DROP actions={drops}. "
                  f"Initial merit={merits[0]:.4g}, final merit={merits[-1]:.4g}. "
                  "An intermediate increase is not itself a failure: feasibility and multiplier repair can trade off across steps.",
                  f"Generated tokens={r['output_tokens']}; model inference share={r['inference_seconds']:.4f}s; "
                  f"numerical solve time={r['solver_seconds']:.4f}s; episode latency={r['wall_seconds']:.4f}s.",""]
Path("report/case_studies.md").write_text("\n".join(lines))
print("Selected",len(successes),"successes and",len(failures),"failures")
