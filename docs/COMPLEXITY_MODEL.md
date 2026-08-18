# Active-Cost Model

The manuscript uses a normalized **active adaptation cost proxy** to compare branch activation. The reference convention is:

- LMS: `C_E = 1`;
- Sign-Error: `C_S = 2`;
- Huber: `C_Hub = 4`;
- correntropy / DMCC: `C_H = 10`.

For a gated robust branch with empirical hard-route rate `Gamma`, the reported proxy averages the cost of the branch actually executed. It is intended to isolate conditional execution of the adaptation rule.

## The proxy does not include

- gate evaluation overhead;
- memory traffic;
- Python/interpreter overhead;
- vectorization and BLAS effects;
- cache behavior;
- operating-system scheduling;
- hardware-dependent instruction cost.

For this reason, the manuscript reports wall-clock results separately and explicitly notes that observed runtime savings are smaller than the idealized active-cost savings.
