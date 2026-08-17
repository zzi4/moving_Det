# Task 5 Fix Round 1 — Formal Web Adapter

## Boundary

- Fix baseline: `156a67a3a042027a23dc5cda83981a9a73065667` (verified before edits).
- Worktree only: `/home/stu1/Projects/moving_Det/.worktrees/mg-vtod-formal-execution`.
- Preserved Vinext/Vite/Sites capability architecture, npm/package-lock, and LAN flow.
- Did not open a browser, deploy, read real formal outputs, or create real formal results. All new tests use temporary producer-compatible fixtures.
- Did not modify `.superpowers/sdd/2026-08-17-mg-vtod-formal-comparison/progress.md`.

## TDD Evidence

Every requested behavior was introduced with a focused failing check before its implementation was changed.

- Canonical producer GPU fixture initially produced 5 failures because the adapter accepted only the abbreviated name; GREEN accepts exactly two `NVIDIA RTX A6000` values and rejects `RTX A6000`.
- Provenance/benchmark/state/probability group initially produced 4 focused failures; GREEN enforces exact Baseline, MG Full, Motion-Off, and optional MG Frozen semantics, binds the comparison benchmark SHA to preflight, rejects failed-training/comparison contradictions, and bounds report probabilities to `[0, 1]`.
- Stable-reader test initially failed because `readStableBoundedFile` did not exist; GREEN uses a single `O_NOFOLLOW` FD, reads no more than limit + 1, and rejects a controlled real-file growth between pre/post `fstat`.
- Wrong-media tests initially failed in both completion and allowlist paths; GREEN hashes every declared MP4/PNG and compares the actual bytes with `demo.json`.
- Range/identity tests initially failed because no secure formal-serving API existed; GREEN covers single-range `206`, invalid/multi-range `416`, Range HEAD, file replacement, parent replacement, changed bytes, symlinks, encoded/traversal misses, and same-FD streaming.
- Cache tests initially failed because no formal cache API existed; GREEN covers in-flight deduplication, reuse within 15 seconds, signature invalidation, and fixed plus demo-declared artifact coverage.
- UI/adapter tests initially failed for missing video names, pending gate styling, refresh clearing, negative threshold/probability acceptance, and duplicate case `src`; all are GREEN.

## Review Finding Evidence

1. Preflight GPU names now match the producer contract exactly: `("NVIDIA RTX A6000", "NVIDIA RTX A6000")`.
2. Stable JSON reads walk non-symlink parents, open once with `O_NOFOLLOW`, compare lstat/fstat identity, read at most limit + 1 from that FD, verify exact byte count plus dev/inode/size/mtime before and after, and parse/hash the returned same-FD buffer.
3. Comparison can no longer overwrite a failed Baseline or MG Full stage; the contradictory tree fails closed.
4. Run provenance is label-specific, and comparison `human_benchmark_sha256` must equal verified preflight.
5. All declared demo bytes are hashed before completion and allowlisting. The allowlist retains verified parent/file dev/inode/size/mtime plus the declared/verified SHA.
6. Formal media serving reopens with `O_NOFOLLOW`, rechecks cached parent/file identity, rehashes the same FD, and streams ranges from that same FD. It supports GET/HEAD, `Accept-Ranges`, `206`, `416`, `Content-Range`, and exact `Content-Length`.
7. The formal status reader is process-shared by configured `formalRoot`, uses a 15-second cache, checks all fixed and demo-declared artifact stat signatures, and deduplicates the whole refresh including slow requests.
8. Client polling schedules the next 15-second refresh only after the current request settles. A refresh failure replaces the previous report with the empty fail-closed report, clearing gate, videos, and cases while showing the matching error text.
9. Server and typed adapter constrain thresholds and probability metrics to `[0, 1]`; signed gate deltas remain valid. Case `src` is unique and is the React key, null gates use a distinct pending class/CSS, and each video has an accessible name.

## Verification

Fresh final commands after the last source edit:

```text
npm test
exit 0
```

The command passed status 8/8, evidence 8/8, formal status 15/15, formal adapter 6/6, formal view 1/1, client/pipeline/LAN 11/11, TypeScript, the production build, and rendered HTML 3/3.

```text
npm run build
exit 0
```

All five Vinext build phases completed. The only emitted advisory was the environment's proxy notice.

```text
npm run lint
exit 0
```

ESLint completed with no warnings or errors.

```text
git diff --check
exit 0
```

## Residual Risk

- The formal filesystem endpoints intentionally remain LAN-only Vite middleware and are not cloud-hosting endpoints.
- Serving a formal media request revalidates the complete declared demo allowlist and rehashes the requested FD. This favors fail-closed local evidence integrity over minimum disk I/O.

# Task 5 Fix Round 2 — Raw Targets and Process-Wide Evidence Cache

## Boundary

- Fix baseline: `0e48db82e27f2edd8b3318d87c4259532530fb01` (verified before edits).
- Worktree only: `/home/stu1/Projects/moving_Det/.worktrees/mg-vtod-formal-execution`.
- Preserved the Vinext/Vite/Sites and LAN-only middleware architecture.
- Did not open a browser, deploy, or read/create any real formal output. All tests use temporary producer-compatible fixtures.
- Did not modify `.superpowers/sdd/2026-08-17-mg-vtod-formal-comparison/progress.md`.

## TDD Evidence

- A real Node HTTP middleware test first failed because no independently testable local API adapter existed. GREEN sends raw request lines over TCP and rejects traversal, mixed-case encoded dot segments, double encoding, backslashes, duplicate slashes, dot segments, NUL encoding, and queries before `new URL`, while a canonical Range request returns `206` with the expected bytes.
- Absolute `run_dir` coverage first failed for empty and relative values. GREEN uses `node:path.isAbsolute` and retains the canonical positive producer fixture.
- MP4/PNG size-limit tests first failed on missing constants; strengthened pre-hash assertions then failed until size inspection occurred through `O_NOFOLLOW` handles before any hash. GREEN enforces 256 MiB per MP4, 16 MiB per PNG, and 1 GiB total declared media.
- Semaphore tests first failed on missing bounded scheduling, and manifest concurrency coverage then failed while hashes were sequential. GREEN bounds hashing to four process-wide and two per formal root while validating declared media concurrently.
- Cache tests first failed on missing process-wide reuse and in-flight deduplication. GREEN reuses the formal-root entry while the manifest identity is unchanged, performs no repeated media hashes for Range/HEAD, deduplicates concurrent initial validation, and invalidates/rebuilds on ctime/manifest replacement.
- Revalidation coverage initially returned `404` after a target identity change. GREEN invalidates the root and fully rebuilds the allowlist once within the same request, then streams only from a newly identity-matched `O_NOFOLLOW` FD.
- FD lifecycle tests cover both size and hash failures and observe exactly one close for each opened handle.
- Final audit added an ETag assertion, observed RED with `"undefined"`, and GREEN now sources the ETag from the allowlisted SHA-256 while streaming from the current identity-matched FD.

## Review Finding Evidence

1. Formal evidence routing examines the raw request target before URL construction. Only literal ASCII `/formal-evidence/...` paths with canonical safe segments are passed to the allowlist; percent encoding, backslashes, duplicate slashes, dot segments, queries, control characters, and noncanonical prefixes fail closed.
2. The module-level evidence cache is process-wide and keyed by `formalRoot`. Each entry retains the stable manifest identity and SHA-256 plus all verified media identities and hashes. Concurrent cold requests share one in-flight validation promise.
3. Cache hits perform only an `O_NOFOLLOW` manifest identity match. Unchanged Range and HEAD requests do not rehash the manifest or any media.
4. Manifest or target identity changes invalidate the whole root and immediately reread the manifest and rehash every declaration. Identity records include parent paths and dev/inode/size/mtime/ctime; rebuilt entries are served only after exact declared hashes pass.
5. Each response opens the target with `O_NOFOLLOW`, compares its cached parent/file identity, and streams the full or ranged response from that exact current FD. Every success and tested error path closes its handle.
6. Declared files are size-inspected before hashing, checked again during verification, and bounded by per-type and aggregate media limits. Hash work is bounded to four globally and two per formal root.
7. Producer run references require a nonempty absolute `run_dir` and retain label-specific canonical provenance validation.

## Verification

Fresh final commands after the last source edit:

```text
npm test
exit 0
```

The command passed 62/62 tests: status 8/8, evidence 16/16, raw local API 1/1, formal status 16/16, formal adapter 6/6, formal view 1/1, client/pipeline/LAN 11/11, and rendered HTML 3/3. Its embedded TypeScript check and production build also completed.

```text
npm run build
exit 0
```

All five independent Vinext build phases completed. The only emitted advisory was the environment's proxy notice.

```text
npm run lint
exit 0
```

ESLint completed with no warnings or errors.

```text
git diff --check
exit 0
```

## Residual Risk

- The formal filesystem endpoints remain intentionally limited to the local/LAN Vite middleware and are not cloud-hosting endpoints.
- Identity-based hot-path reuse assumes the filesystem reports dev/inode/size/mtime/ctime changes reliably; any observed identity change deliberately triggers the more expensive full manifest/media verification.

# Task 5 Fix Round 3 — Manifest and Preflight Identity Barriers

## Boundary

- Fix baseline: `94f6ce4b4bc1cd7d32b9a9aab5f5fa74cb6bf55c` (verified before edits).
- Changed only the formal evidence consistency path and its declarations/tests.
- Did not open a browser, deploy, or read/create real formal output. All tests use temporary fixtures.
- Did not modify `.superpowers/sdd/2026-08-17-mg-vtod-formal-comparison/progress.md`.

## TDD Evidence

- The first focused RED run had 16 passing and 5 failing evidence tests. The manifest group showed that an in-hash manifest replacement still published and streamed the old `200` response, and that a response-before-write hook was ignored. The preflight group showed missing `expectedVerification`, no rejection when one inspected file changed, and 21 hash calls after a 21-file preflight changed to an oversized aggregate.
- The preflight GREEN retains each inspected parent/file dev/inode/size/mtime/ctime identity, accumulates total bytes only from those bound identities, checks every identity again before starting any hash, and supplies that same `expectedVerification` to every hash open. A one-file identity change starts zero hashes; the 21-file change rebuilds once, observes the new aggregate above 1 GiB, and starts zero hashes/byte reads.
- The manifest GREEN performs a stable `O_NOFOLLOW` identity barrier after all media hashes and immediately before cache publication. The old allowlist is never inserted if the original manifest parent/file identity changed.
- A cache-hit response opens and identity-matches the target FD, then repeats the original manifest identity barrier before any response header or body. Failure closes the target, invalidates the root, and allows one consistency rebuild only. A second manifest change fails closed with no evidence headers or old bytes.

## Verification

Fresh final commands after the last source edit:

```text
npm test
exit 0
```

The command passed 67/67 tests: status 8/8, evidence 21/21, raw local API 1/1, formal status 16/16, formal adapter 6/6, formal view 1/1, client/pipeline/LAN 11/11, and rendered HTML 3/3. Its embedded TypeScript check and production build also completed.

```text
npm run build
exit 0
```

All five independent Vinext build phases completed. The only advisory was the environment proxy notice.

```text
npm run lint
exit 0
```

ESLint completed with no warnings or errors.

```text
git diff --check
exit 0
```

## Residual Risk

- The two consistency barriers deliberately add `O_NOFOLLOW` identity opens on cold cache publication and on each formal media response; unchanged requests still avoid full media rehashing.
- Fail-closed identity guarantees depend on the filesystem's dev/inode/size/mtime/ctime reporting, consistent with the existing local formal evidence trust boundary.

# Task 5 Fix Round 4 — Aggregate Request Consistency Budget

## Boundary

- Fix baseline: `2c72cfea575198acf1ea69be70f8ee618657792e` (verified before edits).
- Changed only the formal evidence cache/route consistency accounting, its declaration, its focused behavior test, and this report.
- Did not open a browser, deploy, or read/create real formal output. The new test uses a temporary producer-compatible fixture.
- Did not modify `.superpowers/sdd/2026-08-17-mg-vtod-formal-comparison/progress.md`.

## RED Evidence

The new test catches the production mutation where `createFormalEvidenceCache.getFiles()` swallows a cache-hit manifest mismatch and rebuilds without charging `serveFormalEvidenceRoute()`'s per-request consistency budget. It combines that mismatch with one pre-response manifest change and requires fail-closed behavior before any evidence header or old byte is written.

```text
node --test server/evidence.test.mjs
exit 1
tests 22; pass 21; fail 1
formal evidence counts a cache-hit manifest rebuild against the request budget
AssertionError: Expected values to be strictly equal: 200 !== 404
```

## GREEN and Final Verification

`getFiles()` now reports a cache-hit consistency rebuild to every caller sharing the process-wide in-flight task. The route aggregates that rebuild with its own target/manifest barrier retries; after one rebuild has been consumed, the next identity failure is rethrown before evidence headers and bytes. The test proves the combined request performs exactly two manifest reads (warm cache plus one rebuild), calls the response barrier once, returns `404`, emits none of `Accept-Ranges`, `Content-Type`, or `ETag`, and does not return the old `site19-day` bytes.

```text
npm run test:evidence
exit 0
tests 22; pass 22; fail 0
```

```text
npm test
exit 0
tests 68; pass 68; fail 0
TypeScript completed; the embedded production build completed all five Vinext phases; rendered HTML passed 3/3.
```

```text
npm run build
exit 0
All five Vinext build phases completed. The only advisory was the environment proxy notice.
```

```text
npm run lint
exit 0
ESLint completed with no warnings or errors.
```

```text
git diff --check
exit 0
```

## Changed Files

- `progress-report-web/server/evidence.mjs`
- `progress-report-web/server/evidence.d.mts`
- `progress-report-web/server/evidence.test.mjs`
- `.superpowers/sdd/2026-08-17-mg-vtod-formal-comparison/task-5-report.md`

## Residual Risk

- Callers that supply a custom `FormalEvidenceCache` implementation must honor the optional consistency-rebuild callback for the route to aggregate cache-internal rebuilds; the process-wide cache used by production does so.
- The existing filesystem identity trust boundary remains based on dev/inode/size/mtime/ctime reporting and `O_NOFOLLOW`; this fix changes only per-request rebuild accounting.

# Task 5 Fix Round 5 — Shared Route-Rebuild Provenance

## Boundary

- Fix baseline: `e3d8ee4a2c0edbbc92fa22f6a5733a366ddbab3e` (verified before edits).
- Worktree and branch only: `/home/stu1/Projects/moving_Det/.worktrees/mg-vtod-formal-execution` on `codex/mg-vtod-formal-execution`.
- Changed only the formal evidence cache/route consistency accounting, its declaration and focused tests, and this report.
- Did not open a browser, deploy, access/create any real formal output, or touch training logic. All new behavior tests use a temporary producer-compatible fixture.
- Did not modify `.superpowers/sdd/2026-08-17-mg-vtod-formal-comparison/progress.md`.

## RED Evidence

The concurrent HTTP regression test catches the production mutation where a route retry creates a shared in-flight rebuild without marking its provenance. Specifically, omitting the route's consumed-budget state leaves the route-originated `pendingEntry.rebuilt` false, so a concurrent joiner receives rebuilt files without its consistency-rebuild callback and incorrectly spends a second rebuild after the next manifest change.

```text
node --test server/evidence.test.mjs
exit 1
tests 23; pass 22; fail 1
formal evidence charges a concurrent joiner for a route-originated rebuild
AssertionError: Expected values to be strictly equal: 200 !== 404
```

The completion-timing mutation was then isolated: retaining a resolved task in `inFlight` while awaiting its creator's callback makes a post-completion request look like a concurrent rebuild joiner.

```text
node --test server/evidence.test.mjs
exit 1
tests 24; pass 23; fail 1
formal evidence cache detaches a completed task before rebuild notification
AssertionError: Expected values to be strictly equal: 1 !== 0
```

## GREEN Implementation and Concurrency Proof

`serveFormalEvidenceRoute()` now tells `getFiles()` when the current request has already consumed its route-level rebuild. A manifest read created by that retry marks the shared in-flight entry as rebuilt, while notification is suppressed for callers that have already consumed their budget. Every uncharged caller that actually awaits the successful shared task is notified exactly once. Notifications are awaited, occur only after task success, and the task is removed from `inFlight` before a creator-specific callback runs, so callback delay/failure cannot retain or mutate shared in-flight state or leak a detached rejection.

The real HTTP/cache test warms the cache, changes the first request's response barrier, blocks its route-originated manifest rebuild, and starts a second request. The observer resolves only after that second request has called the real cache and joined the pending task. After release, the joiner receives one rebuild notification while the initiating retry receives zero; a second manifest change at the joiner's response barrier returns `404` before `Accept-Ranges`, `Content-Type`, `ETag`, or old `site19-day` bytes. Exactly three route cache calls and two manifest reads prove that the joiner performs no second rebuild. A separate timing test holds the creator's async notification and proves a request arriving after task completion uses the published cache entry with zero rebuild notifications.

```text
npm run test:evidence
exit 0
tests 24; pass 24; fail 0
```

## Complete Verification

Fresh commands after the last source edit:

```text
npm test
exit 0
70/70 tests passed across status 8/8, evidence 24/24, raw local API 1/1,
formal status 16/16, formal adapter 6/6, formal view 1/1,
client/pipeline/LAN 11/11, and rendered HTML 3/3.
TypeScript completed; the embedded production build completed all five Vinext phases.
```

```text
npm run build
exit 0
All five Vinext build phases completed. The only advisory was the environment proxy notice.
```

```text
npm run lint
exit 0
ESLint completed with no warnings or errors.
```

```text
git diff --check
exit 0
```

## Changed Files

- `progress-report-web/server/evidence.mjs`
- `progress-report-web/server/evidence.d.mts`
- `progress-report-web/server/evidence.test.mjs`
- `.superpowers/sdd/2026-08-17-mg-vtod-formal-comparison/task-5-report.md`

## Residual Risk

- The production process-wide cache implements shared route-rebuild provenance. Existing custom cache objects remain runtime-compatible because the new consumed-budget option is optional; a custom cache that performs its own in-flight deduplication must propagate equivalent provenance to provide the same aggregate guarantee.
- The unchanged local filesystem trust boundary still relies on dev/inode/size/mtime/ctime identity reporting and `O_NOFOLLOW`.
