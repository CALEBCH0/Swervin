---
name: changelog
description: Inspect the current git diff, classify the change type, and append a structured entry to CHANGELOG.md. Invoke manually with /changelog [hint].
argument-hint: "[optional hint, e.g. image generator fix]"
disable-model-invocation: true
---

Summarize the current changes and append a structured entry to CHANGELOG.md.

Hint provided by user: `$ARGUMENTS` (may be empty — base the entry on what the diff actually shows regardless).

## Step 1 — Inspect changes

Run these and read the output:
1. `git diff HEAD` — unstaged changes
2. `git diff --staged` — staged changes
3. `git status` — untracked files
4. `git log --oneline -10` — recent commits for context

If the tree is clean and there are no recent unpublished commits, tell the user there is nothing to log and stop.

## Step 2 — Classify the change

Pick exactly one type based on the primary purpose of the change:

| Type | Use when |
|---|---|
| `bug fix` | Existing behavior was broken and is now corrected |
| `feature` | New user-visible behavior was added |
| `refactor` | Internal structure changed, no intended behavior change |
| `cleanup` | Dead code removed, renamed things, formatting |
| `performance` | Faster runtime, lower memory, fewer calls |
| `test` | Tests added or updated |
| `docs` | Documentation or comments changed |
| `config` | Build, environment, CI, linting, package settings |
| `dependency` | Package added, removed, or upgraded |
| `data/schema` | File format, API payload, model structure changed |
| `security` | Permissions, validation, auth, injection risk |
| `UX` | User flow, error messages, interface behavior |

## Step 3 — Write the entry

Do not invent causes, files, or behavior. If the cause is unclear from the diff, write `Cause: Not confirmed from available changes.` If no behavior change is intended, write `Behavior change: None intended.`

**Bug fix:**
```
#### <title>
Date: <YYYY-MM-DD>
**Type:** Bug Fix
**Context:** <what was broken and when it manifested>
**Cause:** <root cause visible in the diff, or "Not confirmed from available changes.">
**Fix:** <what was changed to correct it>
**Files changed:**
- `<path>`
**Impact:** <what is restored or unblocked>
```

**Feature:**
```
#### <title>
Date: <YYYY-MM-DD>
**Type:** Feature
**Context:** <why this was needed>
**Change:** <what was added>
**Behavior:** <how the system now behaves>
**Files changed:**
- `<path>`
**Impact:** <user-visible effect>
```

**Refactor / Cleanup:**
```
#### <title>
Date: <YYYY-MM-DD>
**Type:** Refactor
**Context:** <why the old structure was a problem>
**Change:** <what was reorganized or removed>
**Behavior change:** None intended.
**Files changed:**
- `<path>`
**Impact:** <maintainability improvement>
```

**Performance:**
```
#### <title>
Date: <YYYY-MM-DD>
**Type:** Performance
**Context:** <where the inefficiency was>
**Change:** <what was optimized>
**Result:** <measured or expected improvement>
**Files changed:**
- `<path>`
**Impact:** <effect on runtime or resource use>
```

**Config / Dependency / Data/Schema / Security / UX / Docs / Test:** use the most relevant fields from the above. Always include: Date, Type, Context, Change, Files changed, Impact.

## Step 4 — Update CHANGELOG.md

If CHANGELOG.md does not exist, create it with this structure first:

```
# Changelog

## Unreleased

### Bug Fixes

### Features

### Refactors

### Performance

### Tests

### Documentation

### Configuration

### Dependencies

### Data and Schema

### Security

### UX
```

Section mapping:
- `bug fix` → `### Bug Fixes`
- `feature` → `### Features`
- `refactor` or `cleanup` → `### Refactors`
- `performance` → `### Performance`
- `test` → `### Tests`
- `docs` → `### Documentation`
- `config` → `### Configuration`
- `dependency` → `### Dependencies`
- `data/schema` → `### Data and Schema`
- `security` → `### Security`
- `UX` → `### UX`

Append the entry under the correct section inside `## Unreleased`. Add the section header if it is missing.

## Step 5 — Show the entry

Print the full entry you added so the user can review it.
