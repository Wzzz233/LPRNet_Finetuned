# Police BlueBase CVStrict — Plan & Progress

## Status: STAGE A RUNNING (10 epochs, ~15 min)

### Step 1: Experiment Directory
✅ Created: experiments/police_bluebase_cvstrict_20260603/

### Step 2: Blue Candidate Check
Checked 4 candidates:
1. v7_stageC_Final — 68-class blue plate expert. Strong backbone. **SELECTED as primary**
2. v7_stageC_iter8000 — checkpoint, similar to Final
3. first_char_guard — province-focused, collapses on real dumps
4. first_board_baseline — empty output on police images

### Step 3: Init Checkpoint
✅ Inherited 54/62 layers from blue v7_stageC_Final
✅ Saved: init_from_bluebase_police_keys.pth (767KB)
✅ Verified loadable

### Step 4: Police v2 Data Audit
✅ Train: 15,500 (31x500), Val: 3,100 (31x100), 100% correct format, 0 I/O
