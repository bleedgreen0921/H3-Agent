# Community Agent Skills

Install the optional dependencies into your existing environment using your normal
project setup (`.[skills]`), then add `[skills]` to the application TOML. Framework
versions are unchanged. Without this section no directory scanning or skill tools
are enabled. The package implements the [Agent Skills format](https://agentskills.io/specification).

```toml
[skills]
roots = []
dependency_install = true
npm_project = "."
dependency_timeout_seconds = 120
```

Roots include workspace `.agents/skills`, user `~/.agents/skills`, and additional
TOML-relative directories. Skills are immediately discoverable. A CLI process owns
a fixed snapshot, including across `/new`; restart it after installs or updates.
Root entry links are supported, internal links are copied only when they stay in
the package, and special files are rejected. Resource directories have arbitrary
names, including `examples/`. Invalid packages appear as diagnostics in `skill list`.
YAML uses a bounded safe loader with duplicate-key and recursive-alias rejection.
Unknown metadata survives parsing; client-specific fields and `allowed-tools` do
not grant permissions. Directory/name differences generate compatibility hints.

```bash
python -m ai_assistant skill discover anthropics/skills --config assistant.toml
python -m ai_assistant skill install anthropics/skills --path skills/pdf --ref main --config assistant.toml
python -m ai_assistant skill install /absolute/local-skill --config assistant.toml
python -m ai_assistant skill list --config assistant.toml
python -m ai_assistant skill inspect SKILL_ID --config assistant.toml
python -m ai_assistant skill check SKILL_ID --config assistant.toml
python -m ai_assistant skill update SKILL_ID --config assistant.toml
python -m ai_assistant skill remove SKILL_ID --config assistant.toml
```

Installation defaults to the workspace; `install --global` selects the user root.
Use full IDs for ambiguous names. GitHub URLs must identify repositories; supply
subdirectories and refs separately, without `/tree/...` URLs. The default branch
is resolved when omitted, and actual commits are recorded and downloaded without
Git. Updates follow the recorded ref; SHA refs stay pinned unless `--ref` changes
explicitly. Local updates reread the original directory. Management never runs
scripts or hooks. The object store and provenance are in the sibling hidden
`.assistant-skills` directory. Entry links publish versions atomically under a
nonblocking file lock. Older objects are retained. Community-managed entries can
be read but cannot be removed or updated by this installer.

`/skills` lists the current snapshot; `/skill ID request` uses the same turn loop
as normal text. API: `await session.run_turn(text, skill_ids=[id], on_event=...)`.
`AgentTurn.skill_ids` reports explicit/tool activations and valid remaining active identities. Normal language can use
`search_skills`, `load_skill`, `unload_skill`, and `read_skill_resource`. Activation
is a native state reducer and Command update; parallel loads merge. Only IDs and
content digests are checkpointed. Failed turns do not advance the successful
checkpoint. Reopening validates digests and expires stale activations. Disabling
Skills disables all skill permissions, even with historical activations.

Full bodies are injected once per model request, after summarization/editing and
before the final input check, and never stored in loading tool results. Large
catalogs switch entirely to paginated search rather than omitting entries. Bodies
remain protected; exceeding the input budget fails explicitly. Unload skills or
use `/new`. Summaries cannot become the authority for skill versions.

`run_command` remains the only script execution route and requires configured
Bubblewrap terminal access. Active snapshots mount read-only below
`/opt/assistant-skills/<stable-id-hash>`; `load_skill` reports the exact path,
workspace, and output location. Existing Node is provided when available.
Registered npm dependencies mount below each skill for CommonJS and ESM lookup.
Dependency links must remain inside their registered tree. In Skills mode, all
terminal profiles mount the current virtual environment and existing npm trees
and manifests read-only; maintenance still permits networking and workspace
outputs. With Skills disabled the original maintenance behavior is preserved.

Controlled dependency installation is enabled by default and may be disabled:

```bash
python -m ai_assistant skill deps install SKILL_ID --manager pip --requirement pypdf --config assistant.toml
python -m ai_assistant skill deps install SKILL_ID --manager npm --requirement PACKAGE --config assistant.toml
```

The agent equivalent, `install_skill_dependencies`, requires a currently active
matching skill. It accepts structured requirements, never shell commands. Each
transaction uses the operation lock plus nonblocking environment/project file
locks. Resolver subprocesses run in Bubblewrap with read-only runtimes and only
transaction staging writable; controlled installation therefore requires Linux and
Bubblewrap even when the general terminal tool is disabled. Pip constrains all installed versions, resolves with dry-run/report,
downloads hash-matched wheels, stages outside the environment, rejects collisions
and startup hooks, and only adds audited files. No virtual environment is created,
no source builds or uninstall/reinstall operations run. Consistency and framework/
CLI subprocess checks run before commit. Npm requires an existing workspace
project and v2/v3 lockfile, stages native installation with scripts disabled,
rejects changed existing lock entries/files/requirements and packages declaring
installation hooks, and commits only new files and manifest deltas. Internal npm
lockfiles are transactional manifests. Existing user npm configuration and model
credentials are not inherited. Node/system packages, workspaces, alternative
package managers, URLs, VCS sources, and private repository authentication are
outside this implementation.

A durable JSON journal records new paths/digests, new directories, and original
npm manifests. Cancellation kills subprocess groups and waits for rollback before
releasing locks. The next controlled install or Skills terminal execution recovers
unfinished journals. Recovery deletes only matching new files and restores only
matching manifests; external modification stops recovery with a diagnostic.
Keep these journals for manual inspection when recovery reports a conflict.
The protocol coordinates application operations; arbitrary external package
managers must not concurrently mutate the same environments.

Run offline regression with `python -m pytest tests/skills`. Pip tests use a
disposable test venv and local wheels; npm uses local tarballs; GitHub uses HTTP
mocks; model tests use scripted responses. No paid model is called. Run
`scripts/check_skills_live.py --help` for explicit real GitHub/package/model
acceptance. The two real community cases are deliberately opt-in and their
results are not implied by offline tests. Static `check` validates declarations
and declared requirements files, not every import or third-party behavior.
