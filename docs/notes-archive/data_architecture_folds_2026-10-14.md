# DATA_ARCHITECTURE.md, the fold paragraph (superseded by the validation split, moved here 2026-10-15)

**Folds.** `fold = md5(dj_key) % 5` on every play, and on every seam through its mix, so every mix of a
DJ and every mix two sources share sit in one fold. Fold 0 is validation (`VAL_FOLD` in the loader).
On 2026-10-14 fold 0 holds 388 mixes, old and Raveform only, because the seven DJ-data DJs hashed into
folds 1 to 4; 7 % of tracks appear in more than one fold. A balanced assignment is open.

