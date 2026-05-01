#!/usr/bin/env python3
# ============================================================
# run_portfolio_optimizer.py
# AI Stock Screener — Portfolio Optimizer (Day 15)
#
# Run after run_weekly.py:
#   python run_portfolio_optimizer.py
#
# Or add to end of run_weekly.py:
#   import subprocess
#   subprocess.Popen(['python',
#       r'E:\...\run_portfolio_optimizer.py'])
#
# Generated from day15_portfolio_optimizer.ipynb
# ============================================================

# ────────────────────────────────────────────────────────────
# INSTRUCTIONS TO COMPLETE THIS FILE
# ────────────────────────────────────────────────────────────

# ── HOW TO GENERATE THIS SCRIPT ──────────────────────────────
#
# Option A — Export from Jupyter (recommended):
#   File → Download as → Python (.py)
#   Then remove the cell markers (# In[N]:)
#
# Option B — The cells in order are:
#   Cell 1 : Imports, paths, constants
#   Cell 2 : Data load, helpers, menu
#   Cell 3 : 4-layer pipeline engine
#   Cell 4 : Auto-tester engine
#   Cell 5 : Actual portfolio
#   Cell 6 : State save (this file)
#
# Entry point (add at bottom after all cells):

if __name__ == '__main__':
    import sys
    if '--auto' in sys.argv:
        # Called automatically from run_weekly.py
        # Run auto-tester update on all active testers
        print("\nAuto-tester update triggered by run_weekly.py")
        testers = _list_testers()
        if testers:
            for tname in testers:
                print(f"\nUpdating tester: {tname}")
                _run_tester_update(tname)
        else:
            print("No active testers.")
        save_optimizer_state()
        check_new_qualifiers()
    else:
        # Interactive mode — show menu
        run_optimizer_menu()

# ────────────────────────────────────────────────────────────
# Add the __main__ block below after
# pasting all cell contents above it:
# ────────────────────────────────────────────────────────────


# ── ENTRY POINT ───────────────────────────────────────────────
if __name__ == '__main__':
    import sys

    # Save state on every run
    save_optimizer_state()
    check_new_qualifiers()

    if '--auto' in sys.argv:
        # Triggered by run_weekly.py — update all testers
        print("\nAuto-update: checking all testers...")
        testers = _list_testers()
        if testers:
            for tname in testers:
                print(f"\n── Tester: {tname} ──")
                _run_tester_update(tname)
        else:
            print("No active testers.")
        print("\nAuto-update complete.")
    else:
        # Interactive — show menu
        run_optimizer_menu()
