# Course Project Workflow

## Canonical working tree

- Google account: dh892127@gmail.com. Verify Drive profile before connector writes.
- Windows: G:\我的云端硬盘\umich-dlcv-wi2022.
- Colab: /content/drive/MyDrive/umich-dlcv-wi2022.
- Remote: https://github.com/Citnarfeng/umich-dlcv-wi2022.git.
- Read docs/WORKFLOW.md before work. Drive is the only development working copy; GitHub stores formal history. Local Downloads copies are emergency backups only.
- Do not treat colab/a1 as editable: those retained notebooks document the superseded self-contained workflow. Develop assignments/a1 .py and .ipynb files directly.

## Start and finish

1. Confirm cwd is inside the canonical Drive project and verify git rev-parse --show-toplevel, git status, git branch --show-current, and git remote -v.
2. Inspect uncommitted/untracked work. Fetch remote updates; when clean, safely pull --rebase if needed. Never reset, delete, or overwrite work to pull.
3. Read code, make minimal assignment-required changes, run relevant tests, debug, retest, inspect diff, and check for unintended files or credentials.
4. Commit and push meaningful verified stages directly: the user's 2026-09-08 instruction supersedes the old commit/push hold. Do not create meaningless relocation commits.
5. Never force push, rewrite published history, delete branches, reset --hard, delete uncommitted files, or overwrite changes without explicit authorization.

## Course and runtime rules

- Preserve course interfaces, tests, cell order, instructions, and helper code. Prefer student TODO blocks. Do not clear all notebook output by default.
- Use Drive for code and retained results. /content is only runtime/cache/staging; copy large data there when useful and copy selected results back.
- Git ignores should be small and specific. Do not globally ignore .pt/.pth because assignments may require model artifacts; inspect assignment requirements.
- Use Colab and Windows sequentially for Git operations. Wait for Drive sync before switching writers; do not run two Git processes from two devices against the same .git directory.
- Keep README concise and English for the user's own instructor. Credit course starter code; do not claim University of Michigan enrollment or unverified completion.
- Keep Codex intermediates under hidden .codex_internal. Preserve source backups and archived pre-migration Drive files; they are not working copies.
