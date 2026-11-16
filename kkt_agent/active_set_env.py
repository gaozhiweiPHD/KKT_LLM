"""Heuristic add/drop repair; no classical global convergence guarantee."""
from collections import Counter
from time import perf_counter
import re
import numpy as np
from .qp_solver import solve_active,NumericalError
from .kkt_verifier import verify

def parse_action(text):
    if not isinstance(text,str): raise ValueError("action must be text")
    match=re.fullmatch(r"\s*(STOP|(?:ADD|DROP)\((0|[1-9][0-9]*)\))\s*",text)
    if not match: raise ValueError("expected exactly ADD(i), DROP(i), or STOP")
    return ("STOP",None) if match[1]=="STOP" else (match[1].split("(")[0],int(match[2]))

class ActiveSetEnv:
    def __init__(self,qp,active=(),max_steps=20,tolerance=1e-5):
        if not 1<=max_steps<=20: raise ValueError("max_steps must be 1..20")
        self.qp,self.active=qp,sorted(active); self.max_steps,self.tolerance=max_steps,tolerance
        self.steps=self.invalid=0; self.done=self.success=self.cycle=False
        self.reason,self.error="running",""
        self.solver_time=self.inference_time=0.; self.input_tokens=self.output_tokens=0
        self.started=perf_counter(); self.visits=Counter({tuple(self.active):1}); self.history=[]
        self._solve(); self.initial_merit=self.certificate.merit; self.history.append(self.snapshot())

    def _solve(self):
        self.solution=solve_active(self.qp,self.active); self.solver_time+=self.solution.seconds
        self.certificate=verify(self.qp,self.solution.x,self.solution.lam,self.tolerance)

    def snapshot(self):
        return {"iteration":self.steps,"active":list(self.active),"residuals":self.certificate.to_dict(),
                "reason":self.reason,"error":self.error,"success":self.success}

    def step(self,text):
        if self.done: raise RuntimeError("episode terminated")
        self.steps+=1; self.error=""
        try:
            action,index=parse_action(text)
            if action=="STOP":
                self._solve(); self.success=self.certificate.success; self.done=True
                self.reason="kkt_success" if self.success else "incorrect_stop"
            else:
                if index>=self.qp.m: raise ValueError("index out of range")
                if action=="ADD" and index in self.active: raise ValueError("duplicate ADD")
                if action=="DROP" and index not in self.active: raise ValueError("DROP of inactive constraint")
                candidate=sorted(self.active+[index]) if action=="ADD" else [i for i in self.active if i!=index]
                start=perf_counter()
                try: sol=solve_active(self.qp,candidate)
                finally: self.solver_time+=perf_counter()-start
                self.active,self.solution=candidate,sol
                self.certificate=verify(self.qp,sol.x,sol.lam,self.tolerance)
                self.visits[tuple(candidate)]+=1; self.cycle|=self.visits[tuple(candidate)]>1
                if self.visits[tuple(candidate)]>=4: self.done,self.reason=True,"cycle"
        except (ValueError,NumericalError) as exc:
            self.invalid+=1; self.error=str(exc)
            try: self._solve()
            except NumericalError: self.done,self.reason=True,"numerical_error"
        if not self.done and self.steps>=self.max_steps: self.done,self.reason=True,"max_steps"
        state=self.snapshot(); state["action"],state["invalid"]=text,bool(self.error); self.history.append(state)
        return state

    def legal_actions(self):
        actions=["STOP"]+[f"DROP({i})" for i in self.active]
        if len(self.active)<self.qp.n:
            for i in range(self.qp.m):
                if i in self.active: continue
                try: solve_active(self.qp,sorted(self.active+[i])); actions.append(f"ADD({i})")
                except NumericalError: pass
        return actions

    def record(self):
        return {"problem_id":self.qp.problem_id,"success":self.success,"reason":self.reason,"actions":self.steps,
                "invalid_actions":self.invalid,"invalid_action_rate":self.invalid/max(1,self.steps),
                "incorrect_stop":self.reason=="incorrect_stop","cycle":self.cycle,
                "inference_seconds":self.inference_time,"solver_seconds":self.solver_time,"wall_seconds":perf_counter()-self.started,
                "input_tokens":self.input_tokens,"output_tokens":self.output_tokens,
                **{k:v for k,v in self.certificate.to_dict().items() if k!="success"},"kkt_candidate":self.certificate.success}
