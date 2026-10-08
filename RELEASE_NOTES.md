Release v2.0.1 corrects documentation and one figure; the code and every result are unchanged from v2.0.

- `RUN_REGISTER.csv` gains a `role` column: 88 reported (every result in Section 7 of the revised manuscript), 11 supplementary,
  15 diagnostic, 85 superseded, 1 excluded. v2.0's era column wrongly implied that runs 111-200 were the reported set.
- `README.md` and `REPRODUCE.md`: the run-range sentences now point to the roles.
- `make_figures.py`: Fig. 4 (`fig7A_separation`) no longer labels the M = 3 point "capacity" (the cap there comes from the
  learner's placement, not from capacity); "within-pool" is now "within-class"; Table IV (`T_mode0b`) adds the fleet-level
  rows; captions match the manuscript; PDFs no longer embed a creation date.
- Version metadata updated to 2.0.1.
