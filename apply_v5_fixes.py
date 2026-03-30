import os
import glob
import shutil

def emergency_fixes():
    # 1. Fix pnl_engine.py double return
    if os.path.exists("pnl_engine.py"):
        with open("pnl_engine.py", "r", encoding="utf-8", errors="ignore") as f:
            c = f.read()
        c = c.replace("return int((return float(", "return int((float(")
        with open("pnl_engine.py", "w", encoding="utf-8") as f:
            f.write(c)
        print("[+] Fixed pnl_engine.py syntax")

    # 2. Fix alerts_engine.py commented bracket
    if os.path.exists("alerts_engine.py"):
        with open("alerts_engine.py", "r", encoding="utf-8", errors="ignore") as f:
            c = f.read()
        c = c.replace("isoformat() # BUG FIX 3: Prevent Tz-Naive crash", "isoformat()")
        with open("alerts_engine.py", "w", encoding="utf-8") as f:
            f.write(c)
        print("[+] Fixed alerts_engine.py syntax")

    # 3. Restore and cleanly patch rl_trainer.py
    baks = sorted(glob.glob("rl_trainer.py.*.bak"))
    if baks:
        shutil.copy(baks[-1], "rl_trainer.py")
        with open("rl_trainer.py", "r", encoding="utf-8", errors="ignore") as f:
            c = f.read()
        
        # Apply the fix safely without messing up indentation
        c = c.replace(
            "expected_dim = int(PolyTradeEnv().observation_space.shape[0])",
            "expected_dim = int(LiveReplayDatasetEnv(df).observation_space.shape[0])"
        )
        with open("rl_trainer.py", "w", encoding="utf-8") as f:
            f.write(c)
        print("[+] Restored and safely patched rl_trainer.py")
    else:
        print("[-] Could not find rl_trainer.py backup to restore.")

if __name__ == "__main__":
    emergency_fixes()